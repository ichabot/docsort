"""Text-Extraktion aus Dokumenten — schneller Pfad via PyMuPDF, OCR-Fallback via Docling."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docsort.config import Config

logger = logging.getLogger(__name__)

# Chunk-Größe für Datei-Hashing (64 KB reicht für Eindeutigkeit + schnell)
_HASH_CHUNK_SIZE = 65536

# Globaler Cache für den DocumentConverter (Modelle nur einmal laden)
_converter_cache: dict[str, Any] = {}

# Minimale Textmenge ab der ein PDF als "digital" gilt (hat brauchbaren Text-Layer)
_MIN_DIGITAL_TEXT_LENGTH = 50

# OpenDocument-Formate die LibreOffice-Konvertierung brauchen
_LIBREOFFICE_FORMATS: dict[str, str] = {
    ".odt": "docx",   # ODT → DOCX
    ".ods": "xlsx",    # ODS → XLSX
    ".odp": "pptx",    # ODP → PPTX
}

# Cache für LibreOffice-Pfad (None = nicht geprüft, False = nicht gefunden, str = Pfad)
_libreoffice_path: str | None | bool = None


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


def _find_libreoffice() -> str | None:
    """Findet den LibreOffice-Pfad (Windows + Linux/macOS).

    Returns:
        Pfad zum LibreOffice-Binary oder None wenn nicht installiert.
    """
    global _libreoffice_path

    if _libreoffice_path is not None:
        return _libreoffice_path if _libreoffice_path else None

    # Linux/macOS: libreoffice oder soffice im PATH
    for cmd in ("libreoffice", "soffice"):
        path = shutil.which(cmd)
        if path:
            _libreoffice_path = path
            return path

    # Windows: Typische Installationspfade
    if os.name == "nt":
        for base in [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ]:
            if not base:
                continue
            program_dir = Path(base) / "LibreOffice" / "program"
            soffice = program_dir / "soffice.exe"
            if soffice.exists():
                _libreoffice_path = str(soffice)
                return str(soffice)

    _libreoffice_path = False
    return None


def _convert_with_libreoffice(file_path: Path, target_format: str) -> Path:
    """Konvertiert eine Datei via LibreOffice headless.

    Args:
        file_path: Quell-Datei (z.B. .odt).
        target_format: Zielformat (z.B. "docx").

    Returns:
        Pfad zur konvertierten Datei in einem temp-Verzeichnis.

    Raises:
        RuntimeError: LibreOffice nicht installiert oder Konvertierung fehlgeschlagen.
    """
    lo_path = _find_libreoffice()
    if not lo_path:
        raise RuntimeError(
            f"LibreOffice wird für {file_path.suffix.upper()}-Dateien benötigt, "
            f"ist aber nicht installiert.\n"
            f"  Windows: https://www.libreoffice.org/download/\n"
            f"  Linux:   sudo apt install libreoffice-core (oder libreoffice)\n"
            f"  macOS:   brew install --cask libreoffice"
        )

    # Temp-Verzeichnis für Output (LibreOffice bestimmt den Dateinamen selbst)
    tmp_dir = tempfile.mkdtemp(prefix="docsort_lo_")

    try:
        cmd = [
            lo_path,
            "--headless",
            "--norestore",
            "--convert-to", target_format,
            "--outdir", tmp_dir,
            str(file_path),
        ]

        logger.info(
            "LibreOffice-Konvertierung: %s → %s",
            file_path.name, target_format.upper(),
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice-Konvertierung fehlgeschlagen (Exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

        # Konvertierte Datei finden
        expected_name = file_path.stem + "." + target_format
        converted = Path(tmp_dir) / expected_name

        if not converted.exists():
            # Fallback: erste Datei im tmp_dir mit dem Zielformat
            candidates = list(Path(tmp_dir).glob(f"*.{target_format}"))
            if candidates:
                converted = candidates[0]
            else:
                raise RuntimeError(
                    f"LibreOffice hat keine {target_format.upper()}-Datei erzeugt."
                )

        logger.info("Konvertiert: %s → %s", file_path.name, converted.name)
        return converted

    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice-Konvertierung: Timeout nach 60 Sekunden.")


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


def _trim_pdf_to_pages(file_path: Path, max_pages: int) -> tuple[Path | None, int]:
    """Trimmt ein PDF auf die ersten N Seiten in eine temporäre Datei.

    Kompatibel mit Windows (schließt alle Handles vor Rückgabe).

    Args:
        file_path: Quell-PDF.
        max_pages: Maximale Seitenzahl.

    Returns:
        Tuple (temp_path, total_pages). temp_path ist None wenn kein Trim nötig.
    """
    import fitz

    src = fitz.open(str(file_path))
    total_pages = len(src)

    if total_pages <= max_pages:
        src.close()
        return None, total_pages

    # Temp-Datei erzeugen — Handle sofort schließen (Windows-Kompatibilität)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    try:
        dst = fitz.open()
        dst.insert_pdf(src, from_page=0, to_page=max_pages - 1)
        dst.save(tmp_path)
        dst.close()
        src.close()

        logger.info(
            "OCR-Seitenlimit: verarbeite %d von %d Seiten.",
            max_pages, total_pages,
        )
        return Path(tmp_path), total_pages

    except Exception:
        src.close()
        # Temp-Datei aufräumen bei Fehler
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_converter(config: Config) -> Any:
    """Erstellt oder gibt den gecachten Docling DocumentConverter zurück.

    Konfiguriert für PDF und Bild-Formate mit externen Plugins (OnnxTR etc.).
    Images nutzen intern dieselbe StandardPdfPipeline wie PDFs —
    daher teilen sich beide dasselbe PdfPipelineOptions-Objekt.
    """
    cache_key = f"gpu={config.gpu}_ocr={config.ocr_batch_size}"

    if cache_key in _converter_cache:
        return _converter_cache[cache_key]

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Gemeinsame Pipeline-Options für PDF und Bilder
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

    # Format-Optionen: PDF + Bilder teilen sich dieselbe Pipeline-Config
    format_options: dict[InputFormat, Any] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }

    # ImageFormatOption für Bilder — nutzt dasselbe pipeline_options
    try:
        from docling.document_converter import ImageFormatOption

        format_options[InputFormat.IMAGE] = ImageFormatOption(
            pipeline_options=pipeline_options,
        )
    except ImportError:
        logger.debug("ImageFormatOption nicht verfügbar — Bilder nutzen Defaults.")

    converter = DocumentConverter(format_options=format_options)
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
    actual_file = file_path
    total_pages = 0
    tmp_path: Path | None = None

    # PDF auf max_pages begrenzen um OCR-Zeit zu sparen
    if config.max_pages > 0 and file_path.suffix.lower() == ".pdf":
        try:
            tmp_path, total_pages = _trim_pdf_to_pages(file_path, config.max_pages)
            if tmp_path is not None:
                actual_file = tmp_path
        except Exception as exc:
            logger.warning("Seitenlimit-Trim fehlgeschlagen (%s) — verarbeite komplettes PDF.", exc)

    converter = _get_converter(config)
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
    if tmp_path is not None:
        try:
            tmp_path.unlink()
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


# ============================================================
# Disk-Cache für extrahierten Text
# ============================================================

def _file_hash(file_path: Path) -> str:
    """Berechnet SHA-256 über die ersten 64 KB einer Datei (schnell + eindeutig genug).

    Args:
        file_path: Pfad zur Datei.

    Returns:
        Hex-String des SHA-256-Hash.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read(_HASH_CHUNK_SIZE))
    return h.hexdigest()


def _cache_path_for(file_path: Path, cache_dir: str) -> Path | None:
    """Gibt den Cache-Pfad für eine Datei zurück, oder None wenn Cache deaktiviert.

    Cache-Verzeichnis wird relativ zum cwd aufgelöst.
    Struktur: <cache_dir>/<hash_prefix>/<hash>.json
    (Unterordner mit den ersten 2 Hex-Zeichen zur Vermeidung riesiger Flat-Dirs.)
    """
    if not cache_dir:
        return None

    fhash = _file_hash(file_path)
    base = Path(cache_dir)
    return base / fhash[:2] / f"{fhash}.json"


def _read_cache(file_path: Path, cache_dir: str) -> ExtractedDoc | None:
    """Versucht ein ExtractedDoc aus dem Disk-Cache zu lesen.

    Returns:
        ExtractedDoc bei Cache-Hit, None bei Miss oder Fehler.
    """
    cp = _cache_path_for(file_path, cache_dir)
    if cp is None or not cp.exists():
        return None

    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        doc = ExtractedDoc(
            text=data["text"],
            metadata=data.get("metadata", {}),
            source_path=file_path,
            num_pages=data.get("num_pages", 0),
            ocr_quality=data.get("ocr_quality", "ok"),
            ocr_quality_info=data.get("ocr_quality_info", ""),
            extraction_method=data.get("extraction_method", ""),
        )
        logger.info("Cache-Hit für %s (Methode: %s).", file_path.name, doc.extraction_method)
        return doc
    except Exception as exc:
        logger.debug("Cache-Lesefehler für %s: %s — wird neu extrahiert.", file_path.name, exc)
        return None


def _write_cache(file_path: Path, doc: ExtractedDoc, cache_dir: str) -> None:
    """Schreibt ein ExtractedDoc in den Disk-Cache.

    Fehler beim Schreiben werden geloggt aber nicht weitergeworfen.
    """
    cp = _cache_path_for(file_path, cache_dir)
    if cp is None:
        return

    data = {
        "text": doc.text,
        "num_pages": doc.num_pages,
        "metadata": doc.metadata,
        "ocr_quality": doc.ocr_quality,
        "ocr_quality_info": doc.ocr_quality_info,
        "extraction_method": doc.extraction_method,
    }

    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        # Atomar schreiben: temp-Datei im selben Verzeichnis, dann umbenennen
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(cp.parent), suffix=".tmp", prefix=".cache_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            # os.replace ist atomar auf allen Plattformen
            os.replace(tmp_path, str(cp))
        except BaseException:
            # Temp-Datei aufräumen bei Fehler
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.debug("Cache geschrieben für %s.", file_path.name)
    except Exception as exc:
        logger.debug("Cache-Schreibfehler für %s: %s", file_path.name, exc)


def extract_text(file_path: Path, config: Config) -> ExtractedDoc:
    """Extrahiert Text aus einer Datei — schneller Pfad für digitale PDFs, OCR für Scans.

    Strategie:
        0. Disk-Cache prüfen (SHA-256 der ersten 64 KB) → sofort zurückgeben
        1. PDF mit Text-Layer → PyMuPDF (schnell, <1s)
        2. Gescanntes PDF / Bilder → Docling OCR (langsamer, GPU-beschleunigt)
        3. Andere Formate (DOCX, XLSX etc.) → Docling
        4. Ergebnis in Disk-Cache schreiben

    Args:
        file_path: Pfad zur Quelldatei.
        config: DocSort-Konfiguration.

    Returns:
        ExtractedDoc mit extrahiertem Text und Metadaten.
    """
    # --- Cache-Hit? ---
    cached = _read_cache(file_path, config.cache_dir)
    if cached is not None:
        return cached

    suffix = file_path.suffix.lower()
    extraction_method = ""
    text = ""
    num_pages = 0
    metadata: dict[str, Any] = {}
    lo_tmp_file: Path | None = None  # Temp-Datei von LibreOffice-Konvertierung

    # OpenDocument-Formate (ODT/ODS/ODP) → via LibreOffice nach DOCX/XLSX/PPTX konvertieren
    actual_file = file_path
    if suffix in _LIBREOFFICE_FORMATS:
        target_fmt = _LIBREOFFICE_FORMATS[suffix]
        lo_tmp_file = _convert_with_libreoffice(file_path, target_fmt)
        actual_file = lo_tmp_file
        suffix = actual_file.suffix.lower()
        extraction_method = "libreoffice+docling"

    try:
        # Schneller Pfad: PDF mit Text-Layer via PyMuPDF
        if suffix == ".pdf":
            try:
                text, num_pages = _extract_pdf_fast(actual_file, max_pages=config.max_pages)

                if len(text.strip()) >= _MIN_DIGITAL_TEXT_LENGTH:
                    extraction_method = extraction_method or "pymupdf"
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
                    text, num_pages, metadata = _extract_with_docling(actual_file, config)
                    extraction_method = extraction_method or "docling-ocr"

            except Exception as exc:
                logger.warning("PyMuPDF fehlgeschlagen (%s) — Fallback auf Docling.", exc)
                text, num_pages, metadata = _extract_with_docling(actual_file, config)
                extraction_method = extraction_method or "docling-ocr"

        # Bilder → immer OCR
        elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
            logger.info("Bild-Datei erkannt — starte OCR: %s", file_path.name)
            text, num_pages, metadata = _extract_with_docling(actual_file, config)
            extraction_method = extraction_method or "docling-ocr"

        # Andere Formate (DOCX, XLSX, etc.) → Docling
        else:
            logger.info("Nicht-PDF/Bild-Format — nutze Docling: %s", file_path.name)
            text, num_pages, metadata = _extract_with_docling(actual_file, config)
            extraction_method = extraction_method or "docling"

    finally:
        # LibreOffice temp-Datei + Verzeichnis aufräumen
        if lo_tmp_file is not None:
            try:
                lo_dir = lo_tmp_file.parent
                lo_tmp_file.unlink(missing_ok=True)
                lo_dir.rmdir()
            except OSError:
                pass

    # Qualitäts-Check
    ocr_quality, ocr_quality_info = _assess_quality(text, file_path)

    doc = ExtractedDoc(
        text=text,
        metadata=metadata,
        source_path=file_path,
        num_pages=num_pages,
        ocr_quality=ocr_quality,
        ocr_quality_info=ocr_quality_info,
        extraction_method=extraction_method,
    )

    # --- Cache schreiben ---
    _write_cache(file_path, doc, config.cache_dir)

    return doc
