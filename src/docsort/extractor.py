"""Text-Extraktion aus Dokumenten — schneller Pfad via PyMuPDF, OCR-Fallback via Docling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docsort.config import Config

logger = logging.getLogger(__name__)

# Globaler Cache für den DocumentConverter (Modelle nur einmal laden)
_converter_cache: dict[str, Any] = {}

# Minimale Textmenge ab der ein PDF als "digital" gilt (hat brauchbaren Text-Layer)
_MIN_DIGITAL_TEXT_LENGTH = 50


@dataclass
class ExtractedDoc:
    """Ergebnis der Textextraktion."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path = field(default_factory=lambda: Path("."))
    num_pages: int = 0
    ocr_quality: str = "ok"       # "ok", "low", "empty"
    ocr_quality_info: str = ""    # Beschreibung des Problems
    extraction_method: str = ""   # "pymupdf", "docling-ocr"


def _extract_pdf_fast(file_path: Path, max_pages: int = 0) -> tuple[str, int]:
    """Extrahiert Text aus PDF via PyMuPDF (schnell, kein OCR).

    Args:
        file_path: Pfad zur PDF-Datei.
        max_pages: Maximale Seitenzahl (0 = alle).

    Returns:
        Tuple (text, num_pages). Text ist leer wenn kein Text-Layer vorhanden.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(file_path))
    num_pages = len(doc)
    pages_to_read = min(num_pages, max_pages) if max_pages > 0 else num_pages
    pages_text: list[str] = []

    for i in range(pages_to_read):
        pages_text.append(doc[i].get_text())

    doc.close()

    if max_pages > 0 and num_pages > max_pages:
        logger.info(
            "Seitenlimit: %d von %d Seiten gelesen (max_pages=%d).",
            pages_to_read, num_pages, max_pages,
        )

    return "\n".join(pages_text), num_pages


def _get_ocr_converter(config: Config) -> Any:
    """Erstellt oder gibt den gecachten Docling DocumentConverter zurück (nur OCR, kein Layout)."""
    cache_key = f"gpu={config.gpu}_ocr={config.ocr_batch_size}"

    if cache_key in _converter_cache:
        return _converter_cache[cache_key]

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = False
    pipeline_options.allow_external_plugins = True

    # GPU-Beschleunigung
    if config.gpu:
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )

            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                num_threads=4,
            )
            logger.info("GPU-Beschleunigung (CUDA) für OCR aktiviert.")
        except Exception:
            logger.warning("CUDA nicht verfügbar — Fallback auf CPU.")

    # OnnxTR OCR Plugin (optional)
    try:
        from docling_ocr_onnxtr import OnnxtrOcrOptions

        pipeline_options.ocr_options = OnnxtrOcrOptions()
        logger.info("OnnxTR OCR-Engine aktiviert.")
    except ImportError:
        logger.debug("docling-ocr-onnxtr nicht installiert — Standard-OCR wird genutzt.")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    _converter_cache[cache_key] = converter
    return converter


def _extract_with_docling(file_path: Path, config: Config) -> tuple[str, int, dict[str, Any]]:
    """Extrahiert Text via Docling (OCR-Pfad für gescannte Dokumente).

    Bei PDFs mit mehr Seiten als config.max_pages wird nur ein Ausschnitt verarbeitet.

    Args:
        file_path: Pfad zur Quelldatei.
        config: DocSort-Konfiguration.

    Returns:
        Tuple (text, num_pages, metadata).
    """
    import tempfile

    actual_file = file_path
    total_pages = 0
    truncated = False

    # PDF auf max_pages begrenzen um OCR-Zeit zu sparen
    if config.max_pages > 0 and file_path.suffix.lower() == ".pdf":
        try:
            import fitz

            src = fitz.open(str(file_path))
            total_pages = len(src)

            if total_pages > config.max_pages:
                truncated = True
                tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                dst = fitz.open()
                dst.insert_pdf(src, from_page=0, to_page=config.max_pages - 1)
                dst.save(tmp.name)
                dst.close()
                actual_file = Path(tmp.name)
                logger.info(
                    "OCR-Seitenlimit: verarbeite %d von %d Seiten.",
                    config.max_pages, total_pages,
                )

            src.close()
        except Exception as exc:
            logger.warning("Seitenlimit-Trim fehlgeschlagen (%s) — verarbeite komplettes PDF.", exc)

    converter = _get_ocr_converter(config)
    result = converter.convert(str(actual_file))

    text = result.document.export_to_markdown()
    num_pages = total_pages if total_pages > 0 else (getattr(result.document, "num_pages", 0) or 0)

    metadata: dict[str, Any] = {}
    if hasattr(result.document, "metadata"):
        meta_obj = result.document.metadata
        if isinstance(meta_obj, dict):
            metadata = meta_obj
        elif hasattr(meta_obj, "__dict__"):
            metadata = {k: v for k, v in meta_obj.__dict__.items() if not k.startswith("_")}

    # Temp-Datei aufräumen
    if truncated and actual_file != file_path:
        try:
            actual_file.unlink()
        except OSError:
            pass

    return text, num_pages, metadata


def _assess_quality(text: str, file_path: Path) -> tuple[str, str]:
    """Prüft die Qualität des extrahierten Texts.

    Returns:
        Tuple (quality, quality_info) — quality ist "ok", "low" oder "empty".
    """
    text_stripped = text.strip()
    text_len = len(text_stripped)

    if text_len == 0:
        logger.warning("OCR-Qualität LEER: %s — kein Text extrahiert.", file_path.name)
        return "empty", "Kein Text extrahiert — Dokument leer oder OCR komplett fehlgeschlagen."

    if text_len < _MIN_DIGITAL_TEXT_LENGTH:
        logger.warning("OCR-Qualität NIEDRIG: %s — nur %d Zeichen.", file_path.name, text_len)
        return "low", f"Sehr wenig Text extrahiert ({text_len} Zeichen) — OCR möglicherweise fehlerhaft."

    # Prüfe Anteil unleserlicher Zeichen (Ersetzungszeichen, Kästchen etc.)
    garbage_chars = sum(1 for c in text_stripped if c in "\ufffd\x00\x01\x02\x03\x04\x05")
    if text_len > 0 and (garbage_chars / text_len) > 0.1:
        logger.warning("OCR-Qualität NIEDRIG: %s — %d/%d Garbage-Zeichen.", file_path.name, garbage_chars, text_len)
        return "low", f"Hoher Anteil unleserlicher Zeichen ({garbage_chars}/{text_len}) — OCR-Qualität fraglich."

    return "ok", ""


def extract_text(file_path: Path, config: Config) -> ExtractedDoc:
    """Extrahiert Text aus einer Datei — schneller Pfad für digitale PDFs, OCR für Scans.

    Strategie:
        1. PDF mit Text-Layer → PyMuPDF (schnell, <1s)
        2. Gescanntes PDF / Bilder → Docling OCR (langsamer, GPU-beschleunigt)
        3. Andere Formate (DOCX, XLSX etc.) → Docling

    Args:
        file_path: Pfad zur Quelldatei.
        config: DocSort-Konfiguration.

    Returns:
        ExtractedDoc mit extrahiertem Text und Metadaten.
    """
    suffix = file_path.suffix.lower()
    extraction_method = ""
    text = ""
    num_pages = 0
    metadata: dict[str, Any] = {}

    # Schneller Pfad: PDF mit Text-Layer via PyMuPDF
    if suffix == ".pdf":
        try:
            text, num_pages = _extract_pdf_fast(file_path, max_pages=config.max_pages)

            if len(text.strip()) >= _MIN_DIGITAL_TEXT_LENGTH:
                extraction_method = "pymupdf"
                logger.info(
                    "PDF hat Text-Layer — PyMuPDF genutzt (%d Zeichen, %d Seiten).",
                    len(text.strip()), num_pages,
                )
            else:
                # Kein oder zu wenig Text → gescanntes PDF → OCR
                logger.info(
                    "PDF hat keinen brauchbaren Text-Layer (%d Zeichen) — starte OCR.",
                    len(text.strip()),
                )
                text, num_pages, metadata = _extract_with_docling(file_path, config)
                extraction_method = "docling-ocr"

        except Exception as exc:
            logger.warning("PyMuPDF fehlgeschlagen (%s) — Fallback auf Docling.", exc)
            text, num_pages, metadata = _extract_with_docling(file_path, config)
            extraction_method = "docling-ocr"

    # Bilder → immer OCR
    elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
        logger.info("Bild-Datei erkannt — starte OCR: %s", file_path.name)
        text, num_pages, metadata = _extract_with_docling(file_path, config)
        extraction_method = "docling-ocr"

    # Andere Formate (DOCX, XLSX, etc.) → Docling ohne OCR
    else:
        logger.info("Nicht-PDF/Bild-Format — nutze Docling: %s", file_path.name)
        text, num_pages, metadata = _extract_with_docling(file_path, config)
        extraction_method = "docling"

    # Qualitäts-Check
    ocr_quality, ocr_quality_info = _assess_quality(text, file_path)

    return ExtractedDoc(
        text=text,
        metadata=metadata,
        source_path=file_path,
        num_pages=num_pages,
        ocr_quality=ocr_quality,
        ocr_quality_info=ocr_quality_info,
        extraction_method=extraction_method,
    )
