"""Tests for dataset source configuration."""

import pytest


class TestDatasetConfig:
    def test_all_sources_not_empty(self):
        from scripts.dataset_config import ALL_SOURCES

        assert len(ALL_SOURCES) > 0

    def test_all_sources_count(self):
        """ALL_SOURCES contains all 9 registered datasets."""
        from scripts.dataset_config import ALL_SOURCES

        assert len(ALL_SOURCES) == 9

    def test_get_sources_by_priority_returns_sorted(self):
        from scripts.dataset_config import get_sources_by_priority

        sources = get_sources_by_priority()
        priorities = [s.priority for s in sources]
        assert priorities == sorted(priorities)

    def test_get_sources_by_priority_filters(self):
        from scripts.dataset_config import get_sources_by_priority

        p1 = get_sources_by_priority(max_priority=1)
        assert all(s.priority <= 1 for s in p1)
        assert len(p1) > 0

    def test_get_source_by_name_found(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("synthbuster")
        assert src is not None
        assert src.name == "synthbuster"

    def test_get_source_by_name_not_found(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("nonexistent_dataset")
        assert src is None

    def test_all_sources_have_required_fields(self):
        from scripts.dataset_config import ALL_SOURCES

        for src in ALL_SOURCES:
            assert src.name
            assert src.modality in ("image", "audio", "video")
            assert src.download_method in (
                "zenodo",
                "huggingface",
                "wget",
                "gdown",
                "kaggle",
            )
            assert src.url
            assert src.priority >= 1
            assert src.est_size_gb > 0
            assert src.access_type in ("open", "gated", "application")

    def test_no_duplicate_names(self):
        """Each dataset source has a unique name."""
        from scripts.dataset_config import ALL_SOURCES

        names = [s.name for s in ALL_SOURCES]
        assert len(names) == len(set(names))


class TestLibriSpeechClean100:
    """Tests for the LIBRISPEECH_CLEAN_100 dataset source."""

    def test_exists_in_all_sources(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("librispeech_clean_100")
        assert src is not None

    def test_modality_is_audio(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("librispeech_clean_100")
        assert src.modality == "audio"

    def test_download_method_is_wget(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("librispeech_clean_100")
        assert src.download_method == "wget"

    def test_priority_is_1(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("librispeech_clean_100")
        assert src.priority == 1

    def test_no_generator_mapping(self):
        """Real-only dataset has empty generator mapping."""
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("librispeech_clean_100")
        assert src.generator_mapping == {}


class TestCocoVal2017:
    """Tests for the COCO_VAL2017 dataset source."""

    def test_exists_in_all_sources(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("coco_val2017")
        assert src is not None

    def test_modality_is_image(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("coco_val2017")
        assert src.modality == "image"

    def test_download_method_is_wget(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("coco_val2017")
        assert src.download_method == "wget"

    def test_priority_is_1(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("coco_val2017")
        assert src.priority == 1

    def test_no_generator_mapping(self):
        """Real-only dataset has empty generator mapping."""
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("coco_val2017")
        assert src.generator_mapping == {}


class TestGenImage:
    """Tests for the GENIMAGE dataset source."""

    def test_exists_in_all_sources(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("genimage")
        assert src is not None

    def test_modality_is_image(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("genimage")
        assert src.modality == "image"

    def test_download_method_is_huggingface(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("genimage")
        assert src.download_method == "huggingface"

    def test_priority_is_2(self):
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("genimage")
        assert src.priority == 2

    def test_generator_mapping_has_8_entries(self):
        """GenImage has 8 generator mappings."""
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("genimage")
        assert len(src.generator_mapping) == 8

    def test_generator_mapping_values(self):
        """Spot-check key generator mappings."""
        from scripts.dataset_config import get_source_by_name

        src = get_source_by_name("genimage")
        assert src.generator_mapping["biggan"] == "biggan"
        assert src.generator_mapping["midjourney"] == "midjourney_v5"
        assert src.generator_mapping["ADM"] == "adm"
        assert src.generator_mapping["wukong"] == "wukong"
