"""Tests for single-phase parallel dispatch (provenance + models together)."""

import time
from unittest.mock import patch

from services.detection import _fan_out_to_services


class TestSinglePhaseDispatch:
    """Validate that _fan_out_to_services dispatches all services in parallel
    and handles partial failures gracefully."""

    @patch("services.detection.query_model_api")
    def test_all_services_dispatched_in_parallel(self, mock_query):
        """All services should start nearly simultaneously, not sequentially."""
        call_times = {}

        def fake_query(name, *args, **kwargs):
            call_times[name] = time.time()
            time.sleep(0.05)
            return {"probability": 0.8, "model_name": name}

        mock_query.side_effect = fake_query

        services = ["model_a", "model_b", "model_c", "provenance_x"]
        results = _fan_out_to_services(
            service_names=services,
            media_type="image",
            encoded_media="base64data",
            threshold=0.5,
            request_id="test-123",
            max_workers=12,
        )
        assert len(results) == 4
        for name in services:
            assert name in results
            assert "error" not in results[name]
        times = list(call_times.values())
        assert max(times) - min(times) < 0.1, "Services were not dispatched in parallel"

    @patch("services.detection.query_model_api")
    def test_partial_failure_returns_all_results(self, mock_query):
        """A failing service should not prevent other services from returning."""

        def fake_query(name, *args, **kwargs):
            if name == "bad_model":
                return {"error": "Service unreachable"}
            return {"probability": 0.7, "model_name": name}

        mock_query.side_effect = fake_query

        services = ["good_model", "bad_model", "another_good"]
        results = _fan_out_to_services(
            service_names=services,
            media_type="image",
            encoded_media="base64data",
            threshold=0.5,
            request_id="test-456",
            max_workers=12,
        )
        assert len(results) == 3
        assert "error" not in results["good_model"]
        assert "error" in results["bad_model"]
        assert "error" not in results["another_good"]
