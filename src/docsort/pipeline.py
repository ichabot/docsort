"""Pipeline: Orchestriert Extract → Classify → Organize."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    low_confidence: bool = False
    ocr_quality: str = "ok"        # "ok", "low", "empty"
    ocr_quality_info: str = ""
    duration_seconds: float = 0.0

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


def _extract_and_classify(
    file_path: Path, config: Config
) -> tuple[Path, ExtractedDoc | None, Classification | None, str | None, float]:
    """Extrahiert Text und klassifiziert eine Datei (parallelisierbar).

    Returns:
        Tuple von (file_path, doc, classification, error, duration).
    """
    t0 = time.monotonic()
    try:
        doc = extract_text(file_path, config)
        logger.debug("Extrahiert: %s (%d Zeichen)", file_path.name, len(doc.text))

        classification = classify(doc, config)
        logger.debug(
            "Klassifiziert: %s → %s (%s, %.0f%%)",
            file_path.name,
            classification.doc_type,
            classification.short_info,
            classification.confidence * 100,
        )
        duration = time.monotonic() - t0
        return (file_path, doc, classification, None, duration)

    except Exception as exc:
        duration = time.monotonic() - t0
        logger.error("Fehler bei %s: %s", file_path.name, exc)
        return (file_path, None, None, str(exc), duration)


def process_file(file_path: Path, config: Config) -> ProcessResult:
    """Verarbeitet eine einzelne Datei: Extract → Classify → Organize.

    Args:
        file_path: Pfad zur Quelldatei.
        config: DocSort-Konfiguration.

    Returns:
        ProcessResult mit Ergebnis aller Schritte.
    """
    try:
        t0 = time.monotonic()
        # 1. Text extrahieren
        doc = extract_text(file_path, config)
        logger.debug("Extrahiert: %s (%d Zeichen)", file_path.name, len(doc.text))

        # 2. Klassifizieren (mit Retry-Logik)
        classification = classify(doc, config)
        logger.debug(
            "Klassifiziert: %s → %s (%s, %.0f%%)",
            file_path.name,
            classification.doc_type,
            classification.short_info,
            classification.confidence * 100,
        )

        # Confidence-Check
        low_confidence = classification.confidence < config.confidence_threshold

        # 3. Organisieren
        result = organize(file_path, classification, config)

        duration = time.monotonic() - t0
        return ProcessResult(
            source=file_path,
            classification=classification,
            organize_result=result,
            success=result.success,
            low_confidence=low_confidence,
            ocr_quality=doc.ocr_quality,
            ocr_quality_info=doc.ocr_quality_info,
            duration_seconds=duration,
        )

    except Exception as exc:
        duration = time.monotonic() - t0
        logger.error("Fehler bei %s: %s", file_path.name, exc)
        return ProcessResult(
            source=file_path,
            error=str(exc),
            success=False,
            duration_seconds=duration,
        )


def process_directory(
    input_dir: Path,
    config: Config,
    progress_callback: Callable[[int, int, ProcessResult], Any] | None = None,
) -> list[ProcessResult]:
    """Verarbeitet alle unterstützten Dateien in einem Verzeichnis.

    Wenn config.max_workers > 1, werden Extraktion und Klassifizierung
    parallel ausgeführt (LLM-Aufrufe sind unabhängige HTTP-Requests).
    Die Organisierung (Dateikopie/-verschiebung) erfolgt stets sequentiell,
    um Race-Conditions bei der Duplikat-Auflösung zu vermeiden.

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

    total = len(files)
    logger.info("Verarbeite %d Datei(en) aus %s", total, input_dir)

    max_workers = getattr(config, "max_workers", 1)

    if max_workers <= 1:
        # --- Sequentieller Modus (bisheriges Verhalten) ---
        results: list[ProcessResult] = []
        for i, file_path in enumerate(files, 1):
            result = process_file(file_path, config)
            results.append(result)
            if progress_callback:
                progress_callback(i, total, result)
    else:
        # --- Paralleler Modus ---
        logger.info("Parallele Verarbeitung mit %d Workern.", max_workers)

        # Phase 1: Extract + Classify parallel
        # Dict preserves insertion order in Python 3.7+; we also keep an
        # ordered list so that results come back in file order.
        ec_results: dict[Path, tuple[ExtractedDoc | None, Classification | None, str | None, float]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(_extract_and_classify, fp, config): fp
                for fp in files
            }
            for future in as_completed(future_to_path):
                fp, doc, classification, error, duration = future.result()
                ec_results[fp] = (doc, classification, error, duration)

        # Phase 2: Organize sequentiell (in ursprünglicher Dateireihenfolge)
        results = []
        for i, file_path in enumerate(files, 1):
            doc, classification, error, duration = ec_results[file_path]

            if error is not None or classification is None:
                result = ProcessResult(
                    source=file_path,
                    error=error or "Klassifizierung fehlgeschlagen",
                    success=False,
                    duration_seconds=duration,
                )
            else:
                try:
                    t0 = time.monotonic()
                    low_confidence = classification.confidence < config.confidence_threshold
                    org_result = organize(file_path, classification, config)
                    duration += time.monotonic() - t0

                    result = ProcessResult(
                        source=file_path,
                        classification=classification,
                        organize_result=org_result,
                        success=org_result.success,
                        low_confidence=low_confidence,
                        ocr_quality=doc.ocr_quality if doc else "ok",
                        ocr_quality_info=doc.ocr_quality_info if doc else "",
                        duration_seconds=duration,
                    )
                except Exception as exc:
                    logger.error("Fehler beim Organisieren von %s: %s", file_path.name, exc)
                    result = ProcessResult(
                        source=file_path,
                        classification=classification,
                        error=str(exc),
                        success=False,
                        duration_seconds=duration,
                    )

            results.append(result)
            if progress_callback:
                progress_callback(i, total, result)

    # Zusammenfassung
    ok = sum(1 for r in results if r.success)
    fail = len(results) - ok
    low_conf = sum(1 for r in results if r.low_confidence)
    logger.info(
        "Fertig: %d/%d erfolgreich, %d Fehler, %d mit niedriger Konfidenz.",
        ok, len(results), fail, low_conf,
    )

    return results
