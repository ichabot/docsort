"""Pipeline: Orchestriert Extract → Classify → Organize."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from docsort.classifier import Classification, classify
from docsort.config import Config
from docsort.extractor import ExtractedDoc, extract_text
from docsort.organizer import OrganizeResult, organize

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Ergebnis der Verarbeitung einer einzelnen Datei."""

    source: Path
    classification: Classification | None = None
    organize_result: OrganizeResult | None = None
    error: str | None = None
    success: bool = False

    @property
    def target(self) -> Path | None:
        """Zielpfad (falls vorhanden)."""
        if self.organize_result:
            return self.organize_result.target
        return None


def collect_files(input_path: Path, config: Config) -> list[Path]:
    """Sammelt alle unterstützten Dateien aus einem Verzeichnis.

    Args:
        input_path: Einzelne Datei oder Verzeichnis.
        config: DocSort-Konfiguration.

    Returns:
        Sortierte Liste der gefundenen Dateipfade.
    """
    if input_path.is_file():
        if input_path.suffix.lower() in config.supported_extensions:
            return [input_path]
        return []

    if not input_path.is_dir():
        return []

    files: list[Path] = []
    for item in sorted(input_path.rglob("*")):
        if item.is_file() and item.suffix.lower() in config.supported_extensions:
            files.append(item)

    return files


def process_file(file_path: Path, config: Config) -> ProcessResult:
    """Verarbeitet eine einzelne Datei: Extract → Classify → Organize.

    Args:
        file_path: Pfad zur Quelldatei.
        config: DocSort-Konfiguration.

    Returns:
        ProcessResult mit Ergebnis aller Schritte.
    """
    try:
        # 1. Text extrahieren
        doc = extract_text(file_path, config)
        logger.debug("Extrahiert: %s (%d Zeichen)", file_path.name, len(doc.text))

        # 2. Klassifizieren
        classification = classify(doc, config)
        logger.debug(
            "Klassifiziert: %s → %s (%s)",
            file_path.name,
            classification.doc_type,
            classification.short_info,
        )

        # 3. Organisieren
        result = organize(file_path, classification, config)

        return ProcessResult(
            source=file_path,
            classification=classification,
            organize_result=result,
            success=result.success,
        )

    except Exception as exc:
        logger.error("Fehler bei %s: %s", file_path.name, exc)
        return ProcessResult(
            source=file_path,
            error=str(exc),
            success=False,
        )


def process_directory(
    input_dir: Path,
    config: Config,
    progress_callback: Callable[[int, int, ProcessResult], Any] | None = None,
) -> list[ProcessResult]:
    """Verarbeitet alle unterstützten Dateien in einem Verzeichnis.

    Args:
        input_dir: Quellverzeichnis.
        config: DocSort-Konfiguration.
        progress_callback: Optionaler Callback(current, total, result) für Fortschritt.

    Returns:
        Liste aller ProcessResults.
    """
    files = collect_files(input_dir, config)

    if not files:
        logger.warning("Keine unterstützten Dateien in %s gefunden.", input_dir)
        return []

    logger.info("Verarbeite %d Datei(en) aus %s", len(files), input_dir)
    results: list[ProcessResult] = []

    for i, file_path in enumerate(files, 1):
        result = process_file(file_path, config)
        results.append(result)

        if progress_callback:
            progress_callback(i, len(files), result)

    # Zusammenfassung
    ok = sum(1 for r in results if r.success)
    fail = len(results) - ok
    logger.info("Fertig: %d/%d erfolgreich, %d Fehler.", ok, len(results), fail)

    return results
