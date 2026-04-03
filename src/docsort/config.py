"""Konfiguration für DocSort."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DOC_TYPES: list[str] = [
    "Rechnung",
    "Quittung",
    "Vertrag",
    "Kuendigung",
    "Brief",
    "Bescheid",
    "Steuerbescheid",
    "Kontoauszug",
    "Lohnabrechnung",
    "Versicherung",
    "Mahnung",
    "Angebot",
    "Lieferschein",
    "Gutschrift",
    "Mietvertrag",
    "Arbeitsvertrag",
    "Zeugnis",
    "Urkunde",
    "Formular",
    "Sonstiges",
]

DEFAULT_EXTENSIONS: list[str] = [
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".html",
    ".htm",
]


@dataclass
class Config:
    """Zentrale Konfiguration für die DocSort-Pipeline."""

    input_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("./sorted"))

    # LLM-Einstellungen (LM Studio / OpenAI-kompatibel)
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = ""
    llm_api_key: str = "lm-studio"

    # Modus
    mode: str = "copy"  # "copy" oder "move"
    dry_run: bool = False

    # GPU / Beschleunigung
    gpu: bool = True
    ocr_batch_size: int = 32
    layout_batch_size: int = 32

    # Unterstützte Dateitypen
    supported_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))

    # Erlaubte Dokumenttypen
    doc_types: list[str] = field(default_factory=lambda: list(DEFAULT_DOC_TYPES))
