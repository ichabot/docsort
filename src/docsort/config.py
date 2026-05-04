"""Konfiguration für DocSort — YAML-basiert mit LLM-Profilen."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Standard-Werte
# ============================================================

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
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".odt", ".ods", ".odp",
    ".jpg", ".jpeg", ".png", ".tiff", ".tif",
    ".bmp", ".webp", ".html", ".htm",
]

DEFAULT_SYSTEM_PROMPT = """\
Du bist ein Dokumenten-Klassifizierer. Analysiere den folgenden Text eines eingescannten Dokuments \
und bestimme den Dokumenttyp, eine kurze Beschreibung und das Dokumentdatum.

Erlaubte Dokumenttypen:
{doc_types}

Antworte ausschließlich mit einem JSON-Objekt in diesem Format:
{{
  "doc_type": "Rechnung",
  "absender": "Stadtwerke-Muenchen",
  "short_info": "Strom-Abrechnung-Januar",
  "doc_date": "2026-11-21",
  "confidence": 0.95
}}

Regeln für absender:
- WICHTIG: Der Absender ist die Organisation/Firma die das Dokument ERSTELLT hat:
  * Lohnabrechnung/Gehaltsabrechnung → Name des ARBEITGEBERS (die Firma bei der man arbeitet), NICHT die Krankenkasse, Rentenversicherung oder Sozialversicherung
  * Rechnung → Firma die die Rechnung ausstellt
  * Versicherung → Versicherungsgesellschaft
  * Behördenschreiben → Name der Behörde
  * Brief/Kündigung → Absender des Briefes
- Bindestriche statt Leerzeichen
- Umlaute ersetzen: ä→ae, ö→oe, ü→ue, ß→ss
- Keine Sonderzeichen außer Bindestrichen
- Maximal 40 Zeichen
- Falls nicht erkennbar: "Unbekannt"

Regeln für short_info:
- Bindestriche statt Leerzeichen
- Umlaute ersetzen: ä→ae, ö→oe, ü→ue, ß→ss
- Keine Sonderzeichen außer Bindestrichen
- Maximal 60 Zeichen
- Kein Datum in der Kurzinfo
- Kein Absender in der Kurzinfo (steht separat)
- Kurz und aussagekräftig

Regeln für doc_date:
- Format: JJJJ-MM-TT
- Falls kein Datum erkennbar: null

Regeln für doc_type:
- Muss einer der erlaubten Dokumenttypen sein
- Falls unklar: "Sonstiges"
"""

DEFAULT_FOLDER_TEMPLATE = "{doc_type}/{year}/{absender}/{filename}"
DEFAULT_FILENAME_TEMPLATE = "{doc_date}_{short_info}"

# ============================================================
# LLM-Profile
# ============================================================

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "lm-studio": {
        "provider": "openai",
        "base_url": "http://localhost:1234/v1",
        "model": "",
        "api_key": "lm-studio",
        "description": "LM Studio (lokal)",
    },
    "ollama": {
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3",
        "api_key": "ollama",
        "description": "Ollama (lokal)",
    },
    "openai": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key": "",
        "description": "OpenAI API",
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-20250514",
        "api_key": "",
        "description": "Anthropic Claude API",
    },
    "gemini": {
        "provider": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "api_key": "",
        "description": "Google Gemini API",
    },
}


@dataclass
class LLMProfile:
    """Ein LLM-Provider-Profil."""

    name: str = "lm-studio"
    provider: str = "openai"  # "openai" oder "anthropic"
    base_url: str = "http://localhost:1234/v1"
    model: str = ""
    api_key: str = "lm-studio"
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> LLMProfile:
        return cls(
            name=name,
            provider=data.get("provider", "openai"),
            base_url=data.get("base_url", ""),
            model=data.get("model", ""),
            api_key=data.get("api_key", ""),
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "description": self.description,
        }


# ============================================================
# Haupt-Config
# ============================================================

@dataclass
class Config:
    """Zentrale Konfiguration für die DocSort-Pipeline."""

    # Verzeichnisse
    input_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("./sorted"))

    # Aktives LLM-Profil
    active_profile: str = "lm-studio"
    profiles: dict[str, LLMProfile] = field(default_factory=dict)

    # Legacy-Felder (werden von active_profile überschrieben)
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = ""
    llm_api_key: str = "lm-studio"
    llm_provider: str = "openai"

    # Modus
    mode: str = "copy"  # "copy" oder "move"
    dry_run: bool = False

    # GPU / Beschleunigung
    gpu: bool = True
    ocr_batch_size: int = 32

    # Seitenlimit — nur die ersten N Seiten extrahieren (0 = alle)
    max_pages: int = 5

    # Dateitypen & Dokumenttypen
    supported_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    doc_types: list[str] = field(default_factory=lambda: list(DEFAULT_DOC_TYPES))

    # Ordnerstruktur-Template
    folder_template: str = DEFAULT_FOLDER_TEMPLATE
    filename_template: str = DEFAULT_FILENAME_TEMPLATE

    # System-Prompt (anpassbar)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Parallelisierung
    max_workers: int = 4  # Anzahl paralleler LLM-Anfragen (1 = sequentiell)

    # Confidence & Retry
    confidence_threshold: float = 0.7
    max_retries: int = 2

    # Logging
    log_file: str = ""
    undo_log: str = ""

    # Extraktions-Cache (Leerstring = deaktiviert)
    cache_dir: str = ".docsort_cache"

    def __post_init__(self) -> None:
        """Initialisiert eingebaute Profile falls leer."""
        if not self.profiles:
            for name, data in BUILTIN_PROFILES.items():
                self.profiles[name] = LLMProfile.from_dict(name, data)

    def get_active_profile(self) -> LLMProfile:
        """Gibt das aktive LLM-Profil zurück."""
        if self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        # Fallback: aus Legacy-Feldern konstruieren
        return LLMProfile(
            name="custom",
            provider=self.llm_provider,
            base_url=self.llm_base_url,
            model=self.llm_model,
            api_key=self.llm_api_key,
        )

    def apply_profile(self, profile_name: str) -> None:
        """Aktiviert ein Profil und setzt die Legacy-Felder."""
        if profile_name in self.profiles:
            self.active_profile = profile_name
            p = self.profiles[profile_name]
            self.llm_provider = p.provider
            self.llm_base_url = p.base_url
            self.llm_model = p.model
            self.llm_api_key = p.api_key

    def to_dict(self) -> dict[str, Any]:
        """Exportiert Config als Dict (für YAML-Speicherung)."""
        profiles_dict = {}
        for name, profile in self.profiles.items():
            # Eingebaute Profile nur speichern wenn geändert
            if name in BUILTIN_PROFILES:
                builtin = BUILTIN_PROFILES[name]
                pdict = profile.to_dict()
                if pdict != builtin:
                    profiles_dict[name] = pdict
            else:
                profiles_dict[name] = profile.to_dict()

        return {
            "active_profile": self.active_profile,
            "profiles": profiles_dict if profiles_dict else None,
            "output_dir": str(self.output_dir),
            "mode": self.mode,
            "gpu": self.gpu,
            "ocr_batch_size": self.ocr_batch_size,
            "max_pages": self.max_pages,
            "doc_types": self.doc_types if self.doc_types != DEFAULT_DOC_TYPES else None,
            "folder_template": self.folder_template if self.folder_template != DEFAULT_FOLDER_TEMPLATE else None,
            "filename_template": self.filename_template if self.filename_template != DEFAULT_FILENAME_TEMPLATE else None,
            "system_prompt": self.system_prompt if self.system_prompt != DEFAULT_SYSTEM_PROMPT else None,
            "max_workers": self.max_workers,
            "confidence_threshold": self.confidence_threshold,
            "max_retries": self.max_retries,
            "log_file": self.log_file or None,
            "undo_log": self.undo_log or None,
            "cache_dir": self.cache_dir if self.cache_dir else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Erstellt Config aus einem Dict (YAML-Daten)."""
        config = cls()

        if "output_dir" in data:
            config.output_dir = Path(data["output_dir"])
        if "mode" in data:
            config.mode = data["mode"]
        if "gpu" in data:
            config.gpu = data["gpu"]
        if "ocr_batch_size" in data:
            config.ocr_batch_size = data["ocr_batch_size"]
        if "max_pages" in data:
            config.max_pages = data["max_pages"]
        if "doc_types" in data and data["doc_types"]:
            config.doc_types = data["doc_types"]
        if "folder_template" in data and data["folder_template"]:
            config.folder_template = data["folder_template"]
        if "filename_template" in data and data["filename_template"]:
            config.filename_template = data["filename_template"]
        if "system_prompt" in data and data["system_prompt"]:
            config.system_prompt = data["system_prompt"]
        if "confidence_threshold" in data:
            config.confidence_threshold = data["confidence_threshold"]
        if "max_retries" in data:
            config.max_retries = data["max_retries"]
        if "max_workers" in data:
            config.max_workers = max(1, int(data["max_workers"]))
        if "log_file" in data and data["log_file"]:
            config.log_file = data["log_file"]
        if "undo_log" in data and data["undo_log"]:
            config.undo_log = data["undo_log"]
        if "cache_dir" in data:
            config.cache_dir = data["cache_dir"] if data["cache_dir"] else ""

        # Profile laden
        if "profiles" in data and data["profiles"]:
            for name, pdata in data["profiles"].items():
                config.profiles[name] = LLMProfile.from_dict(name, pdata)

        if "active_profile" in data:
            config.active_profile = data["active_profile"]
            config.apply_profile(config.active_profile)

        return config


def load_config(path: Path | str | None = None) -> Config:
    """Lädt Config aus YAML-Datei.

    Suchpfade (wenn path=None):
        1. ./docsort.yaml
        2. ~/.config/docsort/docsort.yaml

    Args:
        path: Expliziter Pfad zur Config-Datei.

    Returns:
        Config-Objekt (Standard-Config wenn keine Datei gefunden).
    """
    import yaml

    search_paths: list[Path] = []

    if path:
        search_paths = [Path(path)]
    else:
        search_paths = [
            Path("docsort.yaml"),
            Path("docsort.yml"),
            Path.home() / ".config" / "docsort" / "docsort.yaml",
        ]

    for p in search_paths:
        if p.exists():
            logger.info("Config geladen: %s", p)
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = Config.from_dict(data)
            return config

    logger.debug("Keine Config-Datei gefunden — verwende Standardwerte.")
    return Config()


def save_config(config: Config, path: Path | str | None = None) -> Path:
    """Speichert Config als YAML-Datei.

    Args:
        config: Config-Objekt.
        path: Zielpfad (Standard: ./docsort.yaml).

    Returns:
        Pfad der gespeicherten Datei.
    """
    import yaml

    if path is None:
        path = Path("docsort.yaml")
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()
    # None-Werte entfernen für saubere YAML
    data = {k: v for k, v in data.items() if v is not None}

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("Config gespeichert: %s", path)
    return path
