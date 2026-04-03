"""Tests für DocSort — Classifier, Organizer und Pipeline-Hilfsfunktionen."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docsort.classifier import Classification, _extract_json, _resolve_date, sanitize_short_info
from docsort.config import Config
from docsort.organizer import OrganizeResult, build_target_path, organize, resolve_duplicate
from docsort.pipeline import collect_files


# ============================================================
# sanitize_short_info
# ============================================================

class TestSanitizeShortInfo:
    """Tests für die Kurzinfo-Bereinigung."""

    def test_umlaute_lowercase(self):
        assert sanitize_short_info("Stühle für Büro") == "Stuehle-fuer-Buero"

    def test_umlaute_uppercase(self):
        assert sanitize_short_info("Ärztliche Überweisung") == "Aerztliche-Ueberweisung"

    def test_eszett(self):
        assert sanitize_short_info("Straße") == "Strasse"

    def test_special_chars_removed(self):
        assert sanitize_short_info("Rechnung #123 (final)") == "Rechnung-123-final"

    def test_multiple_spaces(self):
        assert sanitize_short_info("Wort   mit   Abstand") == "Wort-mit-Abstand"

    def test_max_length(self):
        long = "A" * 60
        result = sanitize_short_info(long)
        assert len(result) <= 50

    def test_empty_string(self):
        assert sanitize_short_info("") == ""

    def test_underscores_to_hyphens(self):
        assert sanitize_short_info("foo_bar_baz") == "foo-bar-baz"

    def test_leading_trailing_hyphens(self):
        assert sanitize_short_info("-Test-") == "Test"

    def test_multiple_hyphens_collapsed(self):
        assert sanitize_short_info("A---B") == "A-B"


# ============================================================
# _extract_json
# ============================================================

class TestExtractJson:
    """Tests für die JSON-Extraktion aus LLM-Antworten."""

    def test_direct_json(self):
        raw = '{"doc_type": "Rechnung", "short_info": "Test", "doc_date": "2026-01-01", "confidence": 0.9}'
        result = _extract_json(raw)
        assert result["doc_type"] == "Rechnung"

    def test_markdown_codeblock(self):
        raw = 'Hier ist die Analyse:\n```json\n{"doc_type": "Vertrag", "short_info": "Mietvertrag"}\n```'
        result = _extract_json(raw)
        assert result["doc_type"] == "Vertrag"

    def test_markdown_codeblock_no_lang(self):
        raw = '```\n{"doc_type": "Brief"}\n```'
        result = _extract_json(raw)
        assert result["doc_type"] == "Brief"

    def test_json_embedded_in_text(self):
        raw = 'Das Ergebnis ist: {"doc_type": "Mahnung", "confidence": 0.8} fertig.'
        result = _extract_json(raw)
        assert result["doc_type"] == "Mahnung"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Kein valides JSON"):
            _extract_json("Das ist kein JSON")

    def test_json_with_whitespace(self):
        raw = '  \n  {"doc_type": "Quittung"}  \n  '
        result = _extract_json(raw)
        assert result["doc_type"] == "Quittung"


# ============================================================
# _resolve_date
# ============================================================

class TestResolveDate:
    """Tests für die Datums-Fallback-Logik."""

    def test_valid_date(self):
        assert _resolve_date("2026-03-15") == "2026-03-15"

    def test_null_string_fallback_to_today(self):
        result = _resolve_date("null")
        assert result == date.today().strftime("%Y-%m-%d")

    def test_none_fallback_to_today(self):
        result = _resolve_date(None)
        assert result == date.today().strftime("%Y-%m-%d")

    def test_empty_string_fallback(self):
        result = _resolve_date("")
        assert result == date.today().strftime("%Y-%m-%d")

    def test_file_mtime_fallback(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("dummy")
        result = _resolve_date(None, source_path=f)
        # Sollte Datei-Datum zurückgeben (heute, da gerade erstellt)
        assert len(result) == 10  # YYYY-MM-DD format


# ============================================================
# build_target_path
# ============================================================

class TestBuildTargetPath:
    """Tests für den Zielpfad-Aufbau."""

    def test_basic_path(self):
        config = Config(output_dir=Path("/output"))
        classification = Classification(
            doc_type="Rechnung",
            short_info="Strom-Januar",
            doc_date="2026-01-15",
            confidence=0.9,
        )
        source = Path("/input/scan001.pdf")
        result = build_target_path(source, classification, config)
        assert result == Path("/output/Rechnung/2026/2026-01-15_Rechnung-Strom-Januar.pdf")

    def test_extension_preserved(self):
        config = Config(output_dir=Path("/output"))
        classification = Classification(
            doc_type="Brief",
            short_info="Kuendigung",
            doc_date="2025-06-01",
            confidence=0.8,
        )
        source = Path("/input/doc.docx")
        result = build_target_path(source, classification, config)
        assert result.suffix == ".docx"

    def test_year_from_date(self):
        config = Config(output_dir=Path("/output"))
        classification = Classification(
            doc_type="Vertrag",
            short_info="Mietvertrag",
            doc_date="2023-11-30",
            confidence=0.95,
        )
        source = Path("/input/contract.pdf")
        result = build_target_path(source, classification, config)
        assert "2023" in str(result)

    def test_uppercase_extension_normalized(self):
        config = Config(output_dir=Path("/output"))
        classification = Classification(
            doc_type="Rechnung",
            short_info="Test",
            doc_date="2026-01-01",
            confidence=0.9,
        )
        source = Path("/input/scan.PDF")
        result = build_target_path(source, classification, config)
        assert result.suffix == ".pdf"


# ============================================================
# resolve_duplicate
# ============================================================

class TestResolveDuplicate:
    """Tests für Duplikat-Auflösung."""

    def test_no_collision(self, tmp_path):
        target = tmp_path / "test.pdf"
        assert resolve_duplicate(target) == target

    def test_first_collision(self, tmp_path):
        target = tmp_path / "test.pdf"
        target.write_text("existing")
        result = resolve_duplicate(target)
        assert result == tmp_path / "test_2.pdf"

    def test_multiple_collisions(self, tmp_path):
        target = tmp_path / "test.pdf"
        target.write_text("existing")
        (tmp_path / "test_2.pdf").write_text("existing2")
        result = resolve_duplicate(target)
        assert result == tmp_path / "test_3.pdf"


# ============================================================
# organize (mit tmp_path)
# ============================================================

class TestOrganize:
    """Tests für die Datei-Organisation."""

    def test_dry_run(self, tmp_path):
        source = tmp_path / "input" / "scan.pdf"
        source.parent.mkdir(parents=True)
        source.write_text("dummy")

        config = Config(output_dir=tmp_path / "output", dry_run=True)
        classification = Classification(
            doc_type="Rechnung",
            short_info="Test",
            doc_date="2026-01-01",
            confidence=0.9,
        )

        result = organize(source, classification, config)
        assert result.success
        assert result.action == "dry-run"
        assert not result.target.exists()

    def test_copy_mode(self, tmp_path):
        source = tmp_path / "input" / "scan.pdf"
        source.parent.mkdir(parents=True)
        source.write_text("dummy content")

        config = Config(output_dir=tmp_path / "output", mode="copy")
        classification = Classification(
            doc_type="Rechnung",
            short_info="Strom",
            doc_date="2026-01-15",
            confidence=0.9,
        )

        result = organize(source, classification, config)
        assert result.success
        assert result.action == "copy"
        assert result.target.exists()
        assert source.exists()  # Original bleibt erhalten

    def test_move_mode(self, tmp_path):
        source = tmp_path / "input" / "scan.pdf"
        source.parent.mkdir(parents=True)
        source.write_text("dummy content")

        config = Config(output_dir=tmp_path / "output", mode="move")
        classification = Classification(
            doc_type="Vertrag",
            short_info="Mietvertrag",
            doc_date="2025-06-01",
            confidence=0.85,
        )

        result = organize(source, classification, config)
        assert result.success
        assert result.action == "move"
        assert result.target.exists()
        assert not source.exists()  # Original wurde verschoben


# ============================================================
# collect_files
# ============================================================

class TestCollectFiles:
    """Tests für die Dateisammlung."""

    def test_collect_from_directory(self, tmp_path):
        (tmp_path / "a.pdf").write_text("pdf")
        (tmp_path / "b.docx").write_text("docx")
        (tmp_path / "c.txt").write_text("ignored")

        config = Config()
        files = collect_files(tmp_path, config)
        names = [f.name for f in files]
        assert "a.pdf" in names
        assert "b.docx" in names
        assert "c.txt" not in names

    def test_collect_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.pdf").write_text("pdf")
        (tmp_path / "top.pdf").write_text("pdf")

        config = Config()
        files = collect_files(tmp_path, config)
        assert len(files) == 2

    def test_collect_single_file(self, tmp_path):
        f = tmp_path / "single.pdf"
        f.write_text("pdf")

        config = Config()
        files = collect_files(f, config)
        assert len(files) == 1
        assert files[0] == f

    def test_collect_unsupported_file(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("text")

        config = Config()
        files = collect_files(f, config)
        assert len(files) == 0

    def test_collect_empty_directory(self, tmp_path):
        config = Config()
        files = collect_files(tmp_path, config)
        assert len(files) == 0
