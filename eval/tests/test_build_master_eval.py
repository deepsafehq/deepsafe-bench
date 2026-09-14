"""Tests for master evaluation set builder."""

import json
import os
import shutil
import tempfile

import pytest

from eval.build_master_eval import (
    TIER_CONFIGS,
    TIER_OUTPUT_DIRS,
    build_master_eval,
    gather_files,
    stratified_sample,
)


class TestStratifiedSample:
    """Tests for the stratified_sample function."""

    def test_balances_generators(self):
        """Sampling distributes evenly across generators."""
        files_by_generator = {
            "gen_a": [f"file_{i}.jpg" for i in range(100)],
            "gen_b": [f"file_{i}.jpg" for i in range(200)],
            "gen_c": [f"file_{i}.jpg" for i in range(50)],
        }
        result = stratified_sample(files_by_generator, total=90, seed=42)
        assert len(result) == 90
        counts = {}
        for _, gen in result:
            counts[gen] = counts.get(gen, 0) + 1
        assert counts["gen_a"] == 30
        assert counts["gen_b"] == 30
        assert counts["gen_c"] == 30

    def test_handles_small_generators(self):
        """Small generators contribute all they have, rest redistributed."""
        files_by_generator = {
            "gen_a": [f"file_{i}.jpg" for i in range(100)],
            "gen_b": [f"file_{i}.jpg" for i in range(5)],
        }
        result = stratified_sample(files_by_generator, total=50, seed=42)
        assert len(result) == 50
        counts = {}
        for _, gen in result:
            counts[gen] = counts.get(gen, 0) + 1
        assert counts["gen_b"] == 5  # All of gen_b
        assert counts["gen_a"] == 45  # Remainder from gen_a

    def test_deterministic(self):
        """Same seed produces same result."""
        files = {"g": [f"f_{i}" for i in range(100)]}
        r1 = stratified_sample(files, total=10, seed=42)
        r2 = stratified_sample(files, total=10, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        files = {"g": [f"f_{i}" for i in range(100)]}
        r1 = stratified_sample(files, total=10, seed=42)
        r2 = stratified_sample(files, total=10, seed=99)
        assert r1 != r2

    def test_total_zero_returns_empty(self):
        """Requesting zero samples returns empty list."""
        files = {"g": [f"f_{i}" for i in range(10)]}
        result = stratified_sample(files, total=0, seed=42)
        assert result == []

    def test_empty_generators_dict(self):
        """Empty input returns empty list."""
        result = stratified_sample({}, total=10, seed=42)
        assert result == []

    def test_total_exceeds_available(self):
        """When total exceeds available files, return all available."""
        files_by_generator = {
            "gen_a": [f"file_{i}.jpg" for i in range(5)],
            "gen_b": [f"file_{i}.jpg" for i in range(3)],
        }
        result = stratified_sample(files_by_generator, total=100, seed=42)
        assert len(result) == 8  # All available

    def test_returns_tuples_of_file_and_generator(self):
        """Each result item is a (file_path, generator_name) tuple."""
        files = {"gen_x": ["a.jpg", "b.jpg"]}
        result = stratified_sample(files, total=2, seed=42)
        assert len(result) == 2
        for item in result:
            assert len(item) == 2
            file_path, gen_name = item
            assert gen_name == "gen_x"
            assert file_path in ("a.jpg", "b.jpg")

    def test_multiple_small_generators_redistribute(self):
        """Multiple small generators exhaust, remainder goes to large."""
        files_by_generator = {
            "big": [f"f_{i}" for i in range(1000)],
            "tiny_a": [f"f_{i}" for i in range(2)],
            "tiny_b": [f"f_{i}" for i in range(3)],
        }
        result = stratified_sample(files_by_generator, total=30, seed=42)
        assert len(result) == 30
        counts = {}
        for _, gen in result:
            counts[gen] = counts.get(gen, 0) + 1
        assert counts["tiny_a"] == 2
        assert counts["tiny_b"] == 3
        assert counts["big"] == 25


class TestGatherFiles:
    """Tests for the gather_files function."""

    def test_gathers_from_subdirectories(self, tmp_path):
        """Finds files organized in source subdirectories."""
        modality_dir = tmp_path / "images" / "fake"
        gen_a = modality_dir / "gen_a"
        gen_b = modality_dir / "gen_b"
        gen_a.mkdir(parents=True)
        gen_b.mkdir(parents=True)

        (gen_a / "img1.jpg").touch()
        (gen_a / "img2.jpg").touch()
        (gen_b / "img3.jpg").touch()

        result = gather_files(str(tmp_path), "images", "fake")
        assert "gen_a" in result
        assert "gen_b" in result
        assert len(result["gen_a"]) == 2
        assert len(result["gen_b"]) == 1
        # All paths should be absolute
        for paths in result.values():
            for p in paths:
                assert os.path.isabs(p)

    def test_skips_hidden_and_system_files(self, tmp_path):
        """Hidden files and system files like .DS_Store are skipped."""
        modality_dir = tmp_path / "images" / "real"
        src = modality_dir / "source1"
        src.mkdir(parents=True)

        (src / "img1.jpg").touch()
        (src / ".DS_Store").touch()
        (src / ".hidden_file").touch()

        result = gather_files(str(tmp_path), "images", "real")
        assert len(result["source1"]) == 1

    def test_empty_directory_excluded(self, tmp_path):
        """Empty source directories are excluded from results."""
        modality_dir = tmp_path / "audio" / "fake"
        empty_gen = modality_dir / "empty_gen"
        empty_gen.mkdir(parents=True)
        has_files = modality_dir / "has_files"
        has_files.mkdir(parents=True)
        (has_files / "audio1.wav").touch()

        result = gather_files(str(tmp_path), "audio", "fake")
        assert "empty_gen" not in result
        assert "has_files" in result

    def test_nonexistent_path_returns_empty(self, tmp_path):
        """Non-existent modality/label path returns empty dict."""
        result = gather_files(str(tmp_path), "images", "fake")
        assert result == {}


class TestTierConfig:
    """Tests for tier configuration constants."""

    def test_tier_configs_exist(self):
        """All three tiers are defined."""
        assert "small" in TIER_CONFIGS
        assert "medium" in TIER_CONFIGS
        assert "full" in TIER_CONFIGS

    def test_tier_config_structure(self):
        """Each tier has images/audio/video with real/fake keys."""
        for tier_name, cfg in TIER_CONFIGS.items():
            for modality in ("images", "audio", "video"):
                assert modality in cfg, f"{tier_name} missing {modality}"
                assert "real" in cfg[modality], f"{tier_name}.{modality} missing 'real'"
                assert "fake" in cfg[modality], f"{tier_name}.{modality} missing 'fake'"

    def test_small_tier_targets(self):
        """Small tier totals between 150 and 250."""
        cfg = TIER_CONFIGS["small"]
        total = sum(v for m in cfg.values() for v in m.values())
        assert 150 <= total <= 250, f"small total={total}"

    def test_medium_tier_targets(self):
        """Medium tier totals between 10K and 20K."""
        cfg = TIER_CONFIGS["medium"]
        total = sum(v for m in cfg.values() for v in m.values())
        assert 10_000 <= total <= 20_000, f"medium total={total}"

    def test_full_tier_targets(self):
        """Full tier totals between 30K and 60K."""
        cfg = TIER_CONFIGS["full"]
        total = sum(v for m in cfg.values() for v in m.values())
        assert 30_000 <= total <= 60_000, f"full total={total}"

    def test_tier_output_dirs(self):
        """Output directory names match expected mapping."""
        assert TIER_OUTPUT_DIRS["small"] == "master_eval_small"
        assert TIER_OUTPUT_DIRS["medium"] == "master_eval"
        assert TIER_OUTPUT_DIRS["full"] == "master_eval_full"


class TestBuildMasterEval:
    """Tests for the build_master_eval orchestrator."""

    def _create_dataset(self, root: str) -> None:
        """Create a minimal fake dataset for testing.

        Args:
            root: Temporary directory to populate.
        """
        # Images
        for gen in ["gen_a", "gen_b"]:
            d = os.path.join(root, "images", "fake", gen)
            os.makedirs(d, exist_ok=True)
            for i in range(10):
                with open(os.path.join(d, f"img_{i}.jpg"), "w") as f:
                    f.write(f"fake_{gen}_{i}")

        real_img = os.path.join(root, "images", "real", "source1")
        os.makedirs(real_img, exist_ok=True)
        for i in range(10):
            with open(os.path.join(real_img, f"real_{i}.jpg"), "w") as f:
                f.write(f"real_{i}")

        # Audio
        for gen in ["vocoder_a"]:
            d = os.path.join(root, "audio", "fake", gen)
            os.makedirs(d, exist_ok=True)
            for i in range(10):
                with open(os.path.join(d, f"audio_{i}.wav"), "w") as f:
                    f.write(f"fake_audio_{i}")

        real_audio = os.path.join(root, "audio", "real", "ljspeech")
        os.makedirs(real_audio, exist_ok=True)
        for i in range(10):
            with open(os.path.join(real_audio, f"real_{i}.wav"), "w") as f:
                f.write(f"real_audio_{i}")

        # Video
        for gen in ["faceswap"]:
            d = os.path.join(root, "video", "fake", gen)
            os.makedirs(d, exist_ok=True)
            for i in range(10):
                with open(os.path.join(d, f"vid_{i}.mp4"), "w") as f:
                    f.write(f"fake_video_{i}")

        real_vid = os.path.join(root, "video", "real", "youtube")
        os.makedirs(real_vid, exist_ok=True)
        for i in range(10):
            with open(os.path.join(real_vid, f"real_{i}.mp4"), "w") as f:
                f.write(f"real_video_{i}")

    def test_dry_run_does_not_create_files(self, tmp_path):
        """Dry run writes no files to output directory."""
        root = str(tmp_path)
        self._create_dataset(root)

        build_master_eval(
            root=root,
            tier="small",
            seed=42,
            dry_run=True,
        )

        out_dir = os.path.join(
            root,
            TIER_OUTPUT_DIRS["small"],
        )
        assert not os.path.exists(out_dir)

    def test_creates_metadata_json(self, tmp_path):
        """Real run creates metadata.json with correct structure."""
        root = str(tmp_path)
        self._create_dataset(root)

        metadata = build_master_eval(
            root=root,
            tier="small",
            seed=42,
            dry_run=False,
        )

        out_dir = TIER_OUTPUT_DIRS["small"]
        metadata_path = os.path.join(
            root,
            out_dir,
            "metadata.json",
        )
        assert os.path.exists(metadata_path)

        with open(metadata_path) as f:
            on_disk = json.load(f)

        assert isinstance(on_disk, list)
        assert len(on_disk) == len(metadata)

        # Check required fields in each entry.
        required_fields = {
            "id",
            "path",
            "modality",
            "label",
            "generator",
            "source_file",
            "format",
        }
        for entry in on_disk:
            assert required_fields.issubset(entry.keys()), (
                f"Missing fields in metadata entry: "
                f"{required_fields - entry.keys()}"
            )

    def test_copies_files_to_output_dir(self, tmp_path):
        """Real run copies files into the tier output directory."""
        root = str(tmp_path)
        self._create_dataset(root)

        build_master_eval(
            root=root,
            tier="small",
            seed=42,
            dry_run=False,
        )

        master_dir = os.path.join(
            root,
            TIER_OUTPUT_DIRS["small"],
        )
        assert os.path.isdir(
            os.path.join(master_dir, "images", "real"),
        )
        assert os.path.isdir(
            os.path.join(master_dir, "images", "fake"),
        )
        assert os.path.isdir(
            os.path.join(master_dir, "audio", "real"),
        )
        assert os.path.isdir(
            os.path.join(master_dir, "audio", "fake"),
        )
        assert os.path.isdir(
            os.path.join(master_dir, "video", "real"),
        )
        assert os.path.isdir(
            os.path.join(master_dir, "video", "fake"),
        )

    def test_metadata_paths_exist(self, tmp_path):
        """Every path in metadata.json points to an actual file."""
        root = str(tmp_path)
        self._create_dataset(root)

        build_master_eval(
            root=root,
            tier="small",
            seed=42,
            dry_run=False,
        )

        master_dir = os.path.join(
            root,
            TIER_OUTPUT_DIRS["small"],
        )
        metadata_path = os.path.join(
            master_dir,
            "metadata.json",
        )
        with open(metadata_path) as f:
            metadata = json.load(f)

        for entry in metadata:
            full_path = os.path.join(
                master_dir,
                entry["path"],
            )
            assert os.path.exists(full_path), f"File not found: {full_path}"

    def test_id_format_by_modality(self, tmp_path):
        """IDs follow modality-specific prefix conventions."""
        root = str(tmp_path)
        self._create_dataset(root)

        metadata = build_master_eval(
            root=root,
            tier="small",
            seed=42,
            dry_run=False,
        )

        for entry in metadata:
            if entry["modality"] == "images":
                assert entry["id"].startswith("img_")
            elif entry["modality"] == "audio":
                assert entry["id"].startswith("aud_")
            elif entry["modality"] == "video":
                assert entry["id"].startswith("vid_")

    def test_full_tier_uses_symlinks(self, tmp_path):
        """Full tier creates symlinks instead of copying files."""
        root = str(tmp_path)
        self._create_dataset(root)

        metadata = build_master_eval(
            root=root,
            tier="full",
            seed=42,
            dry_run=False,
        )

        master_dir = os.path.join(
            root,
            TIER_OUTPUT_DIRS["full"],
        )

        # At least one file should have been created.
        assert len(metadata) > 0

        for entry in metadata:
            full_path = os.path.join(
                master_dir,
                entry["path"],
            )
            assert os.path.islink(full_path), f"Expected symlink: {full_path}"
            assert os.path.exists(full_path), f"Broken symlink: {full_path}"

    def test_invalid_tier_raises(self, tmp_path):
        """Unknown tier raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tier"):
            build_master_eval(
                root=str(tmp_path),
                tier="nonexistent",
                seed=42,
                dry_run=True,
            )
