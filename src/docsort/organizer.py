"""Dateien umbenennen und in Ordnerstruktur einsortieren."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
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
    """Baut den Zielpfad nach Schema: output_dir / Dokumenttyp / Jahr / Dateiname.

    Dateiname: JJJJ-MM-TT_Typ-Kurzinfo.ext

    Args:
        source: Quelldatei (für Extension).
        classification: Klassifizierungsergebnis.
        config: DocSort-Konfiguration.

    Returns:
        Vollständiger Zielpfad.
    """
    ext = source.suffix.lower()
    year = classification.doc_date[:4]

    filename = f"{classification.doc_date}_{classification.doc_type}-{classification.short_info}{ext}"

    return config.output_dir / classification.doc_type / year / filename


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
