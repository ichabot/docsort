"""Dokumenten-Klassifizierung per LLM — unterstützt OpenAI-kompatible APIs und Anthropic Claude."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from docsort.config import Config
from docsort.extractor import ExtractedDoc

logger = logging.getLogger(__name__)

# Maximale Textlänge die ans LLM gesendet wird
MAX_TEXT_LENGTH = 3000


@dataclass
class Classification:
    """Ergebnis der LLM-Klassifizierung."""

    doc_type: str
    short_info: str
    doc_date: str  # JJJJ-MM-TT
    confidence: float
    absender: str = ""


def sanitize_short_info(text: str) -> str:
    """Bereinigt die Kurzinfo: Umlaute ersetzen, Sonderzeichen entfernen.

    Args:
        text: Rohe Kurzinfo vom LLM.

    Returns:
        Bereinigte Kurzinfo mit max. 60 Zeichen.
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

    # Max 60 Zeichen
    if len(text) > 60:
        text = text[:60].rstrip("-")

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


# ============================================================
# LLM-Aufrufe: OpenAI-kompatibel und Anthropic nativ
# ============================================================

def _call_openai(system: str, user_text: str, config: Config) -> str:
    """Ruft ein OpenAI-kompatibles LLM auf (OpenAI, LM Studio, Ollama, Gemini etc.).

    Args:
        system: System-Prompt.
        user_text: Dokumenttext.
        config: DocSort-Konfiguration.

    Returns:
        Rohe LLM-Antwort als String.
    """
    from openai import OpenAI

    profile = config.get_active_profile()

    client = OpenAI(
        base_url=profile.base_url,
        api_key=profile.api_key,
    )

    model = profile.model or "local-model"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(system: str, user_text: str, config: Config) -> str:
    """Ruft die Anthropic Claude API nativ auf.

    Args:
        system: System-Prompt.
        user_text: Dokumenttext.
        config: DocSort-Konfiguration.

    Returns:
        Rohe LLM-Antwort als String.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Das 'anthropic' Paket ist nicht installiert. "
            "Installieren mit: uv pip install anthropic"
        )

    profile = config.get_active_profile()

    client = Anthropic(api_key=profile.api_key)

    response = client.messages.create(
        model=profile.model or "claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,
    )

    # Anthropic gibt eine Liste von Content-Blöcken zurück
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts)


def _call_llm(system: str, user_text: str, config: Config) -> str:
    """Ruft das LLM auf — wählt automatisch den richtigen Provider.

    Args:
        system: System-Prompt.
        user_text: Dokumenttext.
        config: DocSort-Konfiguration.

    Returns:
        Rohe LLM-Antwort als String.
    """
    profile = config.get_active_profile()

    if profile.provider == "anthropic":
        return _call_anthropic(system, user_text, config)
    else:
        return _call_openai(system, user_text, config)


# ============================================================
# Klassifizierung
# ============================================================

def classify(doc: ExtractedDoc, config: Config) -> Classification:
    """Klassifiziert ein Dokument per LLM mit Retry-Logik.

    Args:
        doc: Extrahiertes Dokument mit Text.
        config: DocSort-Konfiguration.

    Returns:
        Classification mit Typ, Kurzinfo, Datum und Konfidenz.

    Raises:
        RuntimeError: Wenn nach allen Retries keine valide Klassifizierung möglich.
    """
    system = config.system_prompt.format(doc_types=", ".join(config.doc_types))
    user_text = doc.text[:MAX_TEXT_LENGTH] if doc.text else "(Kein Text extrahiert)"

    last_error: Exception | None = None
    max_attempts = max(1, config.max_retries + 1)

    for attempt in range(1, max_attempts + 1):
        try:
            raw = _call_llm(system, user_text, config)
            data = _extract_json(raw)

            doc_type = data.get("doc_type", "Sonstiges")
            if doc_type not in config.doc_types:
                logger.warning("Unbekannter Dokumenttyp '%s' — verwende 'Sonstiges'.", doc_type)
                doc_type = "Sonstiges"

            short_info = sanitize_short_info(data.get("short_info", "Dokument"))
            if not short_info:
                short_info = "Dokument"

            absender = sanitize_short_info(data.get("absender", "Unbekannt"))
            if not absender:
                absender = "Unbekannt"

            raw_date = data.get("doc_date")
            doc_date = _resolve_date(raw_date, doc.source_path)

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            # Confidence-Warnung
            if confidence < config.confidence_threshold:
                logger.warning(
                    "Niedrige Konfidenz (%.0f%%) für %s — unter Schwelle von %.0f%%.",
                    confidence * 100,
                    doc.source_path.name if doc.source_path else "?",
                    config.confidence_threshold * 100,
                )

            return Classification(
                doc_type=doc_type,
                short_info=short_info,
                doc_date=doc_date,
                confidence=confidence,
                absender=absender,
            )

        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "LLM-Fehler (Versuch %d/%d): %s — Retry in %ds...",
                    attempt, max_attempts, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM-Fehler nach %d Versuchen: %s", max_attempts, exc,
                )

    raise RuntimeError(
        f"Klassifizierung fehlgeschlagen nach {max_attempts} Versuchen: {last_error}"
    )
