"""Text-Extraktion aus Dokumenten mittels Docling und OCR."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docsort.config import Config

logger = logging.getLogger(__name__)

# Globaler Cache für den DocumentConverter (Modelle nur einmal laden)
_converter_cache: dict[str, Any] = {}


@dataclass
class ExtractedDoc:
    """Ergebnis der Textextraktion."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path = field(default_factory=lambda: Path("."))
    num_pages: int = 0
    ocr_quality: str = "ok"       # "ok", "low", "empty"
    ocr_quality_info: str = ""    # Beschreibung des Problems


def _get_converter(config: Config) -> Any:
    """Erstellt oder gibt den gecachten DocumentConverter zurück."""
    cache_key = f"gpu={config.gpu}_ocr={config.ocr_batch_size}_layout={config.layout_batch_size}"

    if cache_key in _converter_cache:
        return _converter_cache[cache_key]

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
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
            logger.info("GPU-Beschleunigung (CUDA) aktiviert.")
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


def extract_text(file_path: Path, config: Config) -> ExtractedDoc:
    """Extrahiert Text aus einer Datei mittels Docling.

    Args:
        file_path: Pfad zur Quelldatei.
        config: DocSort-Konfiguration.

    Returns:
        ExtractedDoc mit extrahiertem Text und Metadaten.
    """
    converter = _get_converter(config)
    result = converter.convert(str(file_path))

    text = result.document.export_to_markdown()
    num_pages = getattr(result.document, "num_pages", 0) or 0

    metadata: dict[str, Any] = {}
    if hasattr(result.document, "metadata"):
        meta_obj = result.document.metadata
        if isinstance(meta_obj, dict):
            metadata = meta_obj
        elif hasattr(meta_obj, "__dict__"):
            metadata = {k: v for k, v in meta_obj.__dict__.items() if not k.startswith("_")}

    # OCR-Qualitäts-Check
    ocr_quality = "ok"
    ocr_quality_info = ""

    text_stripped = text.strip()
    text_len = len(text_stripped)

    if text_len == 0:
        ocr_quality = "empty"
        ocr_quality_info = "Kein Text extrahiert — Dokument leer oder OCR komplett fehlgeschlagen."
        logger.warning("OCR-Qualität LEER: %s — kein Text extrahiert.", file_path.name)
    elif text_len < 50:
        ocr_quality = "low"
        ocr_quality_info = f"Sehr wenig Text extrahiert ({text_len} Zeichen) — OCR möglicherweise fehlerhaft."
        logger.warning("OCR-Qualität NIEDRIG: %s — nur %d Zeichen.", file_path.name, text_len)
    else:
        # Prüfe Anteil unleserlicher Zeichen (Ersetzungszeichen, Kästchen etc.)
        garbage_chars = sum(1 for c in text_stripped if c in "\ufffd\x00\x01\x02\x03\x04\x05")
        if text_len > 0 and (garbage_chars / text_len) > 0.1:
            ocr_quality = "low"
            ocr_quality_info = f"Hoher Anteil unleserlicher Zeichen ({garbage_chars}/{text_len}) — OCR-Qualität fraglich."
            logger.warning("OCR-Qualität NIEDRIG: %s — %d/%d Garbage-Zeichen.", file_path.name, garbage_chars, text_len)

    return ExtractedDoc(
        text=text,
        metadata=metadata,
        source_path=file_path,
        num_pages=num_pages,
        ocr_quality=ocr_quality,
        ocr_quality_info=ocr_quality_info,
    )
