"""Unit tests for the eval runner module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests


class TestLoadEndpoints:
    """Loading model endpoints from deepsafe_config.json."""

    def test_extracts_localhost_urls_from_docker_hostnames(self, tmp_path):
        """Docker-internal URLs are converted to localhost:{port}/predict."""
        from deepsafe_eval.runner import load_endpoints

        config = {
            "media_types": {
                "image": {
                    "model_endpoints": {
                        "npr_deepfakedetection": "http://npr_deepfakedetection:5001/predict",
                        "c2pa_checker": "http://c2pa-checker:9001/predict",
                    },
                },
                "audio": {
                    "model_endpoints": {
                        "shiftyspeech_detection": "http://shiftyspeech_detection:8001/predict",
                    },
                },
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        endpoints = load_endpoints(config_path)

        assert (
            endpoints["image"]["npr_deepfakedetection"]
            == "http://localhost:5001/predict"
        )
        assert endpoints["image"]["c2pa_checker"] == "http://localhost:9001/predict"
        assert (
            endpoints["audio"]["shiftyspeech_detection"]
            == "http://localhost:8001/predict"
        )

    def test_groups_by_media_type(self, tmp_path):
        """Endpoints are grouped by media type key from config."""
        from deepsafe_eval.runner import load_endpoints

        config = {
            "media_types": {
                "image": {
                    "model_endpoints": {
                        "npr_deepfakedetection": "http://npr_deepfakedetection:5001/predict",
                    },
                },
                "video": {
                    "model_endpoints": {
                        "fakestormer": "http://fakestormer_detection:7001/predict",
                    },
                },
                "audio": {
                    "model_endpoints": {
                        "sonics_detection": "http://sonics_detection:8003/predict",
                    },
                },
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        endpoints = load_endpoints(config_path)

        assert set(endpoints.keys()) == {"image", "video", "audio"}

    def test_missing_config_raises(self):
        """Non-existent config path raises FileNotFoundError."""
        from deepsafe_eval.runner import load_endpoints

        with pytest.raises(FileNotFoundError):
            load_endpoints(Path("/nonexistent/config.json"))


class TestCallModel:
    """HTTP model calls with retry and analytics."""

    def test_successful_call_returns_response_and_analytics(self):
        """Successful POST returns wrapper with response and analytics."""
        from deepsafe_eval.runner import call_model

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"probability": 0.87, "prediction": 1}
        mock_resp.raise_for_status = MagicMock()

        with patch("deepsafe_eval.runner.requests.post", return_value=mock_resp):
            result = call_model(
                url="http://localhost:5001/predict",
                b64_data="dGVzdA==",
                payload_key="image_data",
                timeout=120,
            )

        assert result["response"] == {"probability": 0.87, "prediction": 1}
        assert result["status"] == "success"
        assert result["http_status_code"] == 200
        assert result["error_message"] is None
        assert result["attempts"] == 1
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0

    def test_retries_once_on_failure_then_returns_error(self):
        """On two consecutive failures, returns error wrapper after retry."""
        from deepsafe_eval.runner import call_model

        with (
            patch(
                "deepsafe_eval.runner.requests.post",
                side_effect=requests.exceptions.ConnectionError("refused"),
            ) as mock_post,
            patch("deepsafe_eval.runner.time.sleep"),
        ):
            result = call_model(
                url="http://localhost:5001/predict",
                b64_data="dGVzdA==",
                payload_key="image_data",
                timeout=120,
            )

        assert result["response"] is None
        assert result["status"] == "connection_error"
        assert result["error_message"] is not None
        assert "refused" in result["error_message"]
        assert result["attempts"] == 2
        assert mock_post.call_count == 2

    def test_retries_once_then_succeeds(self):
        """If first call fails but retry succeeds, returns success."""
        from deepsafe_eval.runner import call_model

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"probability": 0.42}
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "deepsafe_eval.runner.requests.post",
                side_effect=[ConnectionError("refused"), mock_resp],
            ),
            patch("deepsafe_eval.runner.time.sleep"),
        ):
            result = call_model(
                url="http://localhost:5001/predict",
                b64_data="dGVzdA==",
                payload_key="image_data",
                timeout=120,
            )

        assert result["response"] == {"probability": 0.42}
        assert result["status"] == "success"
        assert result["attempts"] == 2

    def test_timeout_classified_correctly(self):
        """Timeout errors are classified as 'timeout' status."""
        from deepsafe_eval.runner import call_model

        with (
            patch(
                "deepsafe_eval.runner.requests.post",
                side_effect=requests.exceptions.Timeout("timed out"),
            ),
            patch("deepsafe_eval.runner.time.sleep"),
        ):
            result = call_model(
                url="http://localhost:5001/predict",
                b64_data="dGVzdA==",
                payload_key="image_data",
                timeout=120,
            )

        assert result["status"] == "timeout"
        assert result["response"] is None

    def test_http_error_classified_correctly(self):
        """HTTP 500 errors are classified as 'http_error'."""
        from deepsafe_eval.runner import call_model

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )

        with (
            patch("deepsafe_eval.runner.requests.post", return_value=mock_resp),
            patch("deepsafe_eval.runner.time.sleep"),
        ):
            result = call_model(
                url="http://localhost:5001/predict",
                b64_data="dGVzdA==",
                payload_key="image_data",
                timeout=120,
            )

        assert result["status"] == "http_error"
        assert result["http_status_code"] == 500


class TestFanOutSample:
    """Fan out to all models for a single sample."""

    def test_calls_all_models_and_collects_results(self):
        """All models for a modality are called and results collected."""
        from deepsafe_eval.runner import fan_out_sample

        def mock_call(url, b64_data, payload_key, timeout):
            port = url.split(":")[2].split("/")[0]
            return {
                "response": {"probability": float(port) / 10000},
                "latency_ms": 100,
                "status": "success",
                "http_status_code": 200,
                "error_message": None,
                "attempts": 1,
            }

        endpoints = {
            "npr_deepfakedetection": "http://localhost:5001/predict",
            "fsd_detection": "http://localhost:5005/predict",
        }

        with patch("deepsafe_eval.runner.call_model", side_effect=mock_call):
            results = fan_out_sample(
                b64_data="dGVzdA==",
                endpoints=endpoints,
                payload_key="image_data",
                timeout=120,
            )

        assert results["npr_deepfakedetection"]["response"][
            "probability"
        ] == pytest.approx(0.5001)
        assert results["fsd_detection"]["response"]["probability"] == pytest.approx(
            0.5005
        )

    def test_handles_partial_failures(self):
        """If one model fails, others still return results."""
        from deepsafe_eval.runner import fan_out_sample

        def mock_call(url, b64_data, payload_key, timeout):
            if "5001" in url:
                return {
                    "response": None,
                    "latency_ms": 2500,
                    "status": "timeout",
                    "http_status_code": None,
                    "error_message": "timed out",
                    "attempts": 2,
                }
            return {
                "response": {"probability": 0.75},
                "latency_ms": 100,
                "status": "success",
                "http_status_code": 200,
                "error_message": None,
                "attempts": 1,
            }

        endpoints = {
            "npr_deepfakedetection": "http://localhost:5001/predict",
            "fsd_detection": "http://localhost:5005/predict",
        }

        with patch("deepsafe_eval.runner.call_model", side_effect=mock_call):
            results = fan_out_sample(
                b64_data="dGVzdA==",
                endpoints=endpoints,
                payload_key="image_data",
                timeout=120,
            )

        assert results["npr_deepfakedetection"]["response"] is None
        assert results["npr_deepfakedetection"]["status"] == "timeout"
        assert results["fsd_detection"]["response"]["probability"] == 0.75


def _wrap(
    response, status="success", latency_ms=100, http_code=200, error=None, attempts=1
):
    """Helper to build a call_model wrapper dict for tests."""
    return {
        "response": response,
        "latency_ms": latency_ms,
        "status": status,
        "http_status_code": http_code,
        "error_message": error,
        "attempts": attempts,
    }


class TestBuildPrediction:
    """Build a prediction dict from raw model results."""

    def test_separates_detection_and_provenance(self):
        """Detection probs go to model_probs, provenance goes to provenance."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "npr_deepfakedetection": _wrap({"probability": 0.7, "prediction": 1}),
            "fsd_detection": _wrap({"probability": 0.9, "prediction": 1}),
            "c2pa_checker": _wrap({"probability": 0.95, "prediction": 1}),
            "sdxl_watermark_detector": _wrap({"probability": 0.1, "prediction": 0}),
        }

        pred = build_prediction(
            sample_id="img_00001",
            label=1,
            generator="dalle_3",
            raw_results=raw_results,
            media_type="image",
        )

        assert pred["model_probs"]["npr"] == 0.7
        assert pred["model_probs"]["fsd"] == 0.9
        assert "c2pa_checker" not in pred["model_probs"]

        assert pred["provenance"]["c2pa_checker"]["probability"] == 0.95
        assert pred["provenance"]["sdxl_watermark_detector"]["probability"] == 0.1

    def test_null_for_failed_models(self):
        """Failed models produce null in model_probs."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "npr_deepfakedetection": _wrap(
                None,
                status="timeout",
                http_code=None,
                error="timed out",
                attempts=2,
            ),
            "fsd_detection": _wrap({"probability": 0.9}),
        }

        pred = build_prediction(
            sample_id="img_00002",
            label=0,
            generator="real",
            raw_results=raw_results,
            media_type="image",
        )

        assert pred["model_probs"]["npr"] is None
        assert pred["model_probs"]["fsd"] == 0.9

    def test_ensemble_prob_is_computed(self):
        """ensemble_prob is present and is a float."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "npr_deepfakedetection": _wrap({"probability": 0.7}),
            "fsd_detection": _wrap({"probability": 0.9}),
        }

        pred = build_prediction(
            sample_id="img_00003",
            label=1,
            generator="stable_diffusion_xl",
            raw_results=raw_results,
            media_type="image",
        )

        assert isinstance(pred["ensemble_prob"], float)
        assert 0.0 <= pred["ensemble_prob"] <= 1.0

    def test_all_detection_failed_gives_null_ensemble(self):
        """If all detection models fail, ensemble_prob is None."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "npr_deepfakedetection": _wrap(
                None,
                status="connection_error",
                http_code=None,
                error="refused",
                attempts=2,
            ),
            "fsd_detection": _wrap(
                None,
                status="timeout",
                http_code=None,
                error="timed out",
                attempts=2,
            ),
        }

        pred = build_prediction(
            sample_id="img_00004",
            label=1,
            generator="dalle_3",
            raw_results=raw_results,
            media_type="image",
        )

        assert pred["ensemble_prob"] is None

    def test_analytics_present_in_output(self):
        """Prediction includes per-model analytics."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "npr_deepfakedetection": _wrap(
                {"probability": 0.7},
                latency_ms=234,
            ),
            "fsd_detection": _wrap(
                None,
                status="timeout",
                http_code=None,
                error="timed out",
                latency_ms=120000,
                attempts=2,
            ),
        }

        pred = build_prediction(
            sample_id="img_00005",
            label=1,
            generator="dalle_3",
            raw_results=raw_results,
            media_type="image",
            file_size_bytes=54321,
        )

        analytics = pred["analytics"]
        assert analytics["file_size_bytes"] == 54321
        assert analytics["n_models_succeeded"] == 1
        assert analytics["n_models_failed"] == 1

        npr_stats = analytics["per_model"]["npr_deepfakedetection"]
        assert npr_stats["latency_ms"] == 234
        assert npr_stats["status"] == "success"
        assert npr_stats["http_status_code"] == 200
        assert npr_stats["attempts"] == 1

        fsd_stats = analytics["per_model"]["fsd_detection"]
        assert fsd_stats["latency_ms"] == 120000
        assert fsd_stats["status"] == "timeout"
        assert fsd_stats["error_message"] == "timed out"
        assert fsd_stats["attempts"] == 2

    def test_analytics_defaults_file_size_zero(self):
        """file_size_bytes defaults to 0 when not provided."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "npr_deepfakedetection": _wrap({"probability": 0.5}),
        }

        pred = build_prediction(
            sample_id="img_00006",
            label=0,
            generator="real",
            raw_results=raw_results,
            media_type="image",
        )

        assert pred["analytics"]["file_size_bytes"] == 0


class TestPartialFile:
    """Incremental write and resume from partial predictions file."""

    def test_write_and_read_back(self, tmp_path):
        """Predictions written incrementally can be read back."""
        from deepsafe_eval.runner import PartialFile

        path = tmp_path / ".partial_predictions.json"
        models_list = ["npr", "fsd"]

        pf = PartialFile(path, models_list)
        pf.append({"id": "img_00001", "label": 1, "ensemble_prob": 0.87})
        pf.append({"id": "img_00002", "label": 0, "ensemble_prob": 0.12})
        pf.close()

        pf2 = PartialFile(path, models_list)
        assert pf2.completed_ids == {"img_00001", "img_00002"}
        assert len(pf2.predictions) == 2

    def test_resume_skips_completed(self, tmp_path):
        """On resume, completed_ids is populated from existing file."""
        from deepsafe_eval.runner import PartialFile

        path = tmp_path / ".partial_predictions.json"
        models_list = ["npr", "fsd"]

        pf = PartialFile(path, models_list)
        pf.append({"id": "img_00001", "label": 1, "ensemble_prob": 0.87})
        pf.close()

        pf2 = PartialFile(path, models_list)
        assert "img_00001" in pf2.completed_ids
        assert "img_00002" not in pf2.completed_ids

    def test_config_mismatch_raises(self, tmp_path):
        """Different model list on resume raises ValueError."""
        from deepsafe_eval.runner import PartialFile

        path = tmp_path / ".partial_predictions.json"

        pf = PartialFile(path, ["npr", "fsd"])
        pf.append({"id": "img_00001", "label": 1, "ensemble_prob": 0.87})
        pf.close()

        with pytest.raises(ValueError, match="config mismatch"):
            PartialFile(path, ["npr", "fsd", "aide"])

    def test_empty_file_starts_fresh(self, tmp_path):
        """Non-existent file starts with empty state."""
        from deepsafe_eval.runner import PartialFile

        path = tmp_path / ".partial_predictions.json"
        pf = PartialFile(path, ["npr"])
        assert pf.completed_ids == set()
        assert pf.predictions == []


class TestRunInference:
    """End-to-end run_inference with mocked HTTP calls."""

    def _make_dataset(self, tmp_path):
        """Create a minimal master_eval structure."""
        master = tmp_path / "master_eval"
        # Create image files
        img_real_dir = master / "images" / "real" / "coco"
        img_real_dir.mkdir(parents=True)
        (img_real_dir / "img_00001.jpg").write_bytes(b"\xff\xd8\xff\xe0fake_jpg")

        img_fake_dir = master / "images" / "fake" / "dalle_3"
        img_fake_dir.mkdir(parents=True)
        (img_fake_dir / "img_00002.jpg").write_bytes(b"\xff\xd8\xff\xe0fake_jpg")

        # Write metadata.json
        metadata = [
            {
                "id": "img_00001",
                "path": "images/real/coco/img_00001.jpg",
                "modality": "images",
                "label": "real",
                "generator": "coco",
                "format": "jpg",
            },
            {
                "id": "img_00002",
                "path": "images/fake/dalle_3/img_00002.jpg",
                "modality": "images",
                "label": "fake",
                "generator": "dalle_3",
                "format": "jpg",
            },
        ]
        (master / "metadata.json").write_text(json.dumps(metadata))

        # Write config
        config = {
            "media_types": {
                "image": {
                    "model_endpoints": {
                        "npr_deepfakedetection": "http://npr:5001/predict",
                    },
                },
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        return tmp_path, config_path

    def test_produces_predictions_for_all_samples(self, tmp_path):
        """run_inference returns one prediction per metadata entry."""
        from deepsafe_eval.runner import run_inference

        dataset_root, config_path = self._make_dataset(tmp_path)
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"probability": 0.65}
        mock_resp.raise_for_status = MagicMock()

        with patch("deepsafe_eval.runner.requests.post", return_value=mock_resp):
            predictions = run_inference(
                config_path=config_path,
                dataset_root=dataset_root,
                output_dir=output_dir,
            )

        assert len(predictions) == 2
        ids = {p["id"] for p in predictions}
        assert ids == {"img_00001", "img_00002"}

    def test_skips_completed_on_resume(self, tmp_path):
        """If partial file exists, completed samples are skipped."""
        from deepsafe_eval.runner import PartialFile, run_inference

        dataset_root, config_path = self._make_dataset(tmp_path)
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        # Pre-populate partial file with one completed sample
        models = ["npr_deepfakedetection"]
        pf = PartialFile(
            output_dir / ".partial_predictions.json",
            models,
        )
        pf.append(
            {
                "id": "img_00001",
                "label": 0,
                "generator": "coco",
                "ensemble_prob": 0.3,
                "model_probs": {"npr": 0.3},
                "provenance": {},
            }
        )
        pf.close()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"probability": 0.65}
        mock_resp.raise_for_status = MagicMock()

        with patch("deepsafe_eval.runner.requests.post", return_value=mock_resp) as mp:
            predictions = run_inference(
                config_path=config_path,
                dataset_root=dataset_root,
                output_dir=output_dir,
            )

        # Only img_00002 should have been processed (1 model call)
        assert mp.call_count == 1
        assert len(predictions) == 2  # both in final output


class TestPredictionModality:
    """Prediction dicts include modality for downstream routing."""

    def test_modality_in_output(self):
        """build_prediction includes the modality field."""
        from deepsafe_eval.runner import build_prediction

        raw_results = {
            "shiftyspeech_detection": _wrap({"probability": 0.8}),
        }

        pred = build_prediction(
            sample_id="aud_00001",
            label=1,
            generator="elevenlabs",
            raw_results=raw_results,
            media_type="audio",
        )

        assert pred["modality"] == "audio"
