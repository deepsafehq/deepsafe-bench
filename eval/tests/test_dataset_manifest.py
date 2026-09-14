"""Tests for the dataset manifest SQLite module."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database path."""
    return tmp_path / "test.db"


class TestManifestCreation:
    def test_create_manifest_creates_db_file(self, tmp_db):
        from scripts.dataset_manifest import create_manifest

        create_manifest(tmp_db)
        assert tmp_db.exists()

    def test_create_manifest_has_files_table(self, tmp_db):
        from scripts.dataset_manifest import create_manifest

        create_manifest(tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "files" in tables

    def test_files_table_has_required_columns(self, tmp_db):
        from scripts.dataset_manifest import create_manifest

        create_manifest(tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("PRAGMA table_info(files)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "path",
            "label",
            "generator",
            "source_dataset",
            "modality",
            "sha256",
        }
        assert expected.issubset(columns)


class TestManifestInsert:
    def test_insert_file_record(self, tmp_db):
        from scripts.dataset_manifest import create_manifest, insert_file

        create_manifest(tmp_db)
        insert_file(
            tmp_db,
            path="images/fake/midjourney_v6/img_001.jpg",
            label="fake",
            generator="midjourney_v6",
            source_dataset="openfake",
            modality="image",
        )
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        conn.close()
        assert count == 1

    def test_insert_batch(self, tmp_db):
        from scripts.dataset_manifest import create_manifest, insert_batch

        create_manifest(tmp_db)
        records = [
            {
                "path": f"images/fake/sdxl/img_{i:04d}.jpg",
                "label": "fake",
                "generator": "stable_diffusion_xl",
                "source_dataset": "genimage",
                "modality": "image",
            }
            for i in range(100)
        ]
        insert_batch(tmp_db, records)
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        conn.close()
        assert count == 100


class TestManifestQuery:
    def test_list_generators(self, tmp_db):
        from scripts.dataset_manifest import (
            create_manifest,
            insert_batch,
            list_generators,
        )

        create_manifest(tmp_db)
        records = [
            {
                "path": f"fake/sdxl/{i}.jpg",
                "label": "fake",
                "generator": "sdxl",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(5)
        ] + [
            {
                "path": f"fake/mj/{i}.jpg",
                "label": "fake",
                "generator": "midjourney_v6",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(3)
        ]
        insert_batch(tmp_db, records)
        generators = list_generators(tmp_db)
        assert set(generators) == {"sdxl", "midjourney_v6"}

    def test_get_files_by_generator(self, tmp_db):
        from scripts.dataset_manifest import create_manifest, get_files, insert_batch

        create_manifest(tmp_db)
        records = [
            {
                "path": f"fake/sdxl/{i}.jpg",
                "label": "fake",
                "generator": "sdxl",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(10)
        ] + [
            {
                "path": f"fake/mj/{i}.jpg",
                "label": "fake",
                "generator": "midjourney_v6",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(5)
        ]
        insert_batch(tmp_db, records)
        sdxl_files = get_files(tmp_db, generator="sdxl")
        assert len(sdxl_files) == 10

    def test_get_files_by_label(self, tmp_db):
        from scripts.dataset_manifest import create_manifest, get_files, insert_batch

        create_manifest(tmp_db)
        records = [
            {
                "path": f"real/{i}.jpg",
                "label": "real",
                "generator": "none",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(20)
        ] + [
            {
                "path": f"fake/sdxl/{i}.jpg",
                "label": "fake",
                "generator": "sdxl",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(30)
        ]
        insert_batch(tmp_db, records)
        real_files = get_files(tmp_db, label="real")
        assert len(real_files) == 20

    def test_get_files_with_sample_size(self, tmp_db):
        from scripts.dataset_manifest import create_manifest, get_files, insert_batch

        create_manifest(tmp_db)
        records = [
            {
                "path": f"fake/sdxl/{i}.jpg",
                "label": "fake",
                "generator": "sdxl",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(100)
        ]
        insert_batch(tmp_db, records)
        sampled = get_files(tmp_db, label="fake", sample_size=10)
        assert len(sampled) == 10

    def test_get_stats(self, tmp_db):
        from scripts.dataset_manifest import create_manifest, get_stats, insert_batch

        create_manifest(tmp_db)
        records = [
            {
                "path": f"real/{i}.jpg",
                "label": "real",
                "generator": "none",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(20)
        ] + [
            {
                "path": f"fake/sdxl/{i}.jpg",
                "label": "fake",
                "generator": "sdxl",
                "source_dataset": "test",
                "modality": "image",
            }
            for i in range(30)
        ]
        insert_batch(tmp_db, records)
        stats = get_stats(tmp_db)
        assert stats["total"] == 50
        assert stats["real"] == 20
        assert stats["fake"] == 30
        assert "sdxl" in stats["generators"]
