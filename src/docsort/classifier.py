"""Dokumenten-Klassifizierung per LLM (LM Studio / OpenAI-kompatibel)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from docsort.config import Config
from docsort.extractor import ExtractedDoc

logger = logging.getLogger(__name__)

# Maximale Textlänge die ans LLM gesendet wird
MAX_TEXT_LENGTH = 3000

SYSTEM_PROMPT = """\
Du bist ein Dokumenten-Klassifizierer. Analysiere den folgenden Text eines eingescannten Dokuments \
und bestimme den Dokumenttyp, eine kurze Beschreibung und das Dokumentdatum.

Erlaubte Dokumenttypen:
{doc_types}

Antworte ausschließlich mit einem JSON-Objekt in diesem Format:
{{
  "doc_type": "Rechnung",
  "short_info": "Sanitaerarbeiten-Firma-Krause",
  "doc_date": "2026-11-21",
  "confidence": 0.95
}}

Regeln für short_info:
- Bindestriche statt Leerzeichen
- Umlaute ersetzen: ä→ae, ö→oe, ü→ue, ß→ss
- Keine Sonderzeichen außer Bindestrichen
- Maximal 50 Zeichen
- Kein Datum in der Kurzinfo
- Kurz und aussagekräftig

Regeln für doc_date:
- Format: JJJJ-MM-TT
- Falls kein Datum erkennbar: null

Regeln für doc_type:
- Muss einer der erlaubten Dokumenttypen sein
- Falls unklar: "Sonstiges"
"""


@dataclass
class Classification:
    """Ergebnis der LLM-Klassifizierung."""

    doc_type: str
    short_info: str
    doc_date: str  # JJJJ-MM-TT
    confidence: float


def sanitize_short_info(text: str) -> str:
    """Bereinigt die Kurzinfo: Umlaute ersetzen, Sonderzeichen entfernen.

    Args:
        text: Rohe Kurzinfo vom LLM.

    Returns:
        Bereinigte Kurzinfo mit max. 50 Zeichen.
    """
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Leerzeichen und Unterstriche durch Bindestriche
    text = text.replace(" ", "-").replace("_", "-")

    # Nur Buchstaben, Ziffern und Bindestriche behalten
    text = re.sub(r"[^a-zA-Z0-9\-]", "", text)

    # Mehrfache Bindestriche zusammenfassen
    text = re.sub(r"-{2,}", "-", text)

    # Führende/nachfolgende Bindestriche entfernen
    text = text.strip("-")

    # Max 50 Zeichen
    if len(text) > 50:
        text = text[:50].rstrip("-")

    return text


def _extract_json(text: str) -> dict[str, Any]:
    """Extrahiert JSON aus LLM-Antwort (direkt oder aus Markdown-Codeblock).

    Args:
        text: Rohe LLM-Antwort.

    Returns:
        Geparster JSON-Dict.

    Raises:
        ValueError: Wenn kein valides JSON gefunden.
    """
    # Versuch 1: Direktes JSON-Parsing
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Versuch 2: JSON aus Markdown-Codeblock extrahieren
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Versuch 3: Erstes { bis letztes } extrahieren
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Kein valides JSON in LLM-Antwort gefunden: {text[:200]}")


def _resolve_date(
    doc_date: str | None,
    source_path: Path | None = None,
) -> str:
    """Bestimmt das Dokumentdatum mit Fallback-Logik.

    Priorität:
        1. Vom LLM erkanntes Datum
        2. Änderungsdatum der Quelldatei
        3. Heutiges Datum

    Args:
        doc_date: Vom LLM erkanntes Datum (kann None/"null"/leer sein).
        source_path: Pfad zur Quelldatei für Fallback.

    Returns:
        Datum im Format JJJJ-MM-TT.
    """
    # Priorität 1: LLM-Datum
    if doc_date and doc_date.lower() not in ("null", "none", ""):
        # Validierung
        try:
            datetime.strptime(doc_date, "%Y-%m-%d")
            return doc_date
        except ValueError:
            logger.warning("Ungültiges Datumsformat vom LLM: %s", doc_date)

    # Priorität 2: Datei-Änderungsdatum
    if source_path and source_path.exists():
        try:
            mtime = source_path.stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except OSError:
            pass

    # Priorität 3: Heutiges Datum
    return date.today().strftime("%Y-%m-%d")


def classify(doc: ExtractedDoc, config: Config) -> Classification:
    """Klassifiziert ein Dokument per LLM.

    Args:
        doc: Extrahiertes Dokument mit Text.
        config: DocSort-Konfiguration.

    Returns:
        Classification mit Typ, Kurzinfo, Datum und Konfidenz.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )

    system = SYSTEM_PROMPT.format(doc_types=", ".join(config.doc_types))
    user_text = doc.text[:MAX_TEXT_LENGTH] if doc.text else "(Kein Text extrahiert)"

    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.1,
    }
    if config.llm_model:
        kwargs["model"] = config.llm_model
    else:
        kwargs["model"] = "local-model"

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""

    data = _extract_json(raw)

    doc_type = data.get("doc_type", "Sonstiges")
    if doc_type not in config.doc_types:
        logger.warning("Unbekannter Dokumenttyp '%s' — verwende 'Sonstiges'.", doc_type)
        doc_type = "Sonstiges"

    short_info = sanitize_short_info(data.get("short_info", "Dokument"))
    if not short_info:
        short_info = "Dokument"

    raw_date = data.get("doc_date")
    doc_date = _resolve_date(raw_date, doc.source_path)

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return Classification(
        doc_type=doc_type,
        short_info=short_info,
        doc_date=doc_date,
        confidence=confidence,
    )
