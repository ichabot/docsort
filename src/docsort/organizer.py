"""Dateien umbenennen und in Ordnerstruktur einsortieren — mit Undo-Log."""

from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from docsort.classifier import Classification
from docsort.config import Config

logger = logging.getLogger(__name__)


@dataclass
class OrganizeResult:
    """Ergebnis einer Datei-Organisation."""

    source: Path
    target: Path
    action: str  # "copy", "move", "dry-run"
    success: bool
    error: str | None = None


def build_target_path(
    source: Path,
    classification: Classification,
    config: Config,
) -> Path:
    """Baut den Zielpfad nach konfigurierbarem Template.

    Template-Variablen:
        {doc_type}  — Dokumenttyp (z.B. Rechnung)
        {absender}  — Absender/Aussteller (z.B. Stadtwerke-Muenchen)
        {year}      — Jahr aus doc_date (z.B. 2026)
        {month}     — Monat aus doc_date (z.B. 01)
        {filename}  — Generierter Dateiname (JJJJ-MM-TT_Kurzinfo.ext)

    Beispiele:
        "{doc_type}/{year}/{absender}/{filename}" → Rechnung/2026/Stadtwerke/2026-01-15_Strom.pdf
        "{doc_type}/{year}/{filename}"            → Rechnung/2026/2026-01-15_Strom.pdf
        "{year}/{doc_type}/{filename}"            → 2026/Rechnung/2026-01-15_Strom.pdf
        "{filename}"                              → 2026-01-15_Strom.pdf (flach)

    Args:
        source: Quelldatei (für Extension).
        classification: Klassifizierungsergebnis.
        config: DocSort-Konfiguration.

    Returns:
        Vollständiger Zielpfad.
    """
    ext = source.suffix.lower()
    doc_date = classification.doc_date
    year = doc_date[:4]
    month = doc_date[5:7] if len(doc_date) >= 7 else "01"

    filename = f"{doc_date}_{classification.short_info}{ext}"

    absender = classification.absender or "Unbekannt"

    relative = config.folder_template.format(
        doc_type=classification.doc_type,
        absender=absender,
        year=year,
        month=month,
        filename=filename,
    )

    return config.output_dir / relative


def resolve_duplicate(target: Path) -> Path:
    """Löst Namenskollisionen durch Suffix _2, _3 etc.

    Args:
        target: Gewünschter Zielpfad.

    Returns:
        Freier Zielpfad (original oder mit Suffix).
    """
    if not target.exists():
        return target

    stem = target.stem
    ext = target.suffix
    parent = target.parent
    counter = 2

    while True:
        candidate = parent / f"{stem}_{counter}{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def _write_undo_log(source: Path, target: Path, action: str, config: Config) -> None:
    """Schreibt einen Eintrag ins Undo-Log.

    Format: CSV mit Spalten timestamp, action, source, target
    """
    if not config.undo_log:
        return

    log_path = Path(config.undo_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()

    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "action", "source", "target"])
            writer.writerow([
                datetime.now().isoformat(),
                action,
                str(source),
                str(target),
            ])
    except OSError as exc:
        logger.warning("Undo-Log konnte nicht geschrieben werden: %s", exc)


def organize(
    source: Path,
    classification: Classification,
    config: Config,
) -> OrganizeResult:
    """Organisiert eine Datei: umbenennen und kopieren/verschieben.

    Args:
        source: Pfad zur Quelldatei.
        classification: Klassifizierungsergebnis.
        config: DocSort-Konfiguration.

    Returns:
        OrganizeResult mit Ergebnis der Operation.
    """
    target = build_target_path(source, classification, config)
    target = resolve_duplicate(target)

    action = "dry-run" if config.dry_run else config.mode

    if config.dry_run:
        logger.info("[DRY-RUN] %s → %s", source.name, target)
        return OrganizeResult(
            source=source,
            target=target,
            action=action,
            success=True,
        )

    try:
        # Zielverzeichnis erstellen
        target.parent.mkdir(parents=True, exist_ok=True)

        if config.mode == "move":
            shutil.move(str(source), str(target))
            logger.info("[MOVE] %s → %s", source.name, target)
        else:
            shutil.copy2(str(source), str(target))
            logger.info("[COPY] %s → %s", source.name, target)

        # Undo-Log schreiben
        _write_undo_log(source, target, config.mode, config)

        return OrganizeResult(
            source=source,
            target=target,
            action=action,
            success=True,
        )

    except Exception as exc:
        logger.error("Fehler bei %s: %s", source.name, exc)
        return OrganizeResult(
            source=source,
            target=target,
            action=action,
            success=False,
            error=str(exc),
        )


def undo_last(config: Config, count: int = 0) -> list[str]:
    """Macht die letzten Operationen rückgängig basierend auf dem Undo-Log.

    Args:
        config: DocSort-Konfiguration mit undo_log Pfad.
        count: Anzahl der Operationen (0 = alle).

    Returns:
        Liste mit Statusmeldungen.
    """
    if not config.undo_log:
        return ["Kein Undo-Log konfiguriert."]

    log_path = Path(config.undo_log)
    if not log_path.exists():
        return ["Undo-Log nicht gefunden."]

    messages: list[str] = []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return ["Undo-Log ist leer."]

    # Von hinten nach vorne (neueste zuerst)
    to_undo = rows[::-1]
    if count > 0:
        to_undo = to_undo[:count]

    remaining = [r for r in rows if r not in to_undo]

    for row in to_undo:
        action = row.get("action", "")
        source = Path(row.get("source", ""))
        target = Path(row.get("target", ""))

        try:
            if action == "copy" and target.exists():
                target.unlink()
                messages.append(f"✓ Gelöscht: {target}")
            elif action == "move" and target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
                messages.append(f"✓ Zurück verschoben: {target} → {source}")
            else:
                messages.append(f"⚠ Übersprungen (Datei nicht gefunden): {target}")
        except Exception as exc:
            messages.append(f"✗ Fehler bei {target}: {exc}")

    # Log aktualisieren
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "action", "source", "target"])
        for r in remaining:
            writer.writerow([r["timestamp"], r["action"], r["source"], r["target"]])

    return messages
