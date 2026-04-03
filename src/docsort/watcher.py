"""Watchfolder — überwacht ein Verzeichnis und verarbeitet neue Dateien automatisch."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from docsort.config import Config
from docsort.pipeline import process_file, ProcessResult

logger = logging.getLogger(__name__)


def watch_directory(
    watch_dir: Path,
    config: Config,
    on_result: Callable[[ProcessResult], None] | None = None,
    poll_interval: float = 5.0,
    run_once: bool = False,
) -> None:
    """Überwacht ein Verzeichnis auf neue Dateien und verarbeitet sie.

    Nutzt Polling statt OS-Events (einfacher, plattformunabhängig, keine extra Dependency).
    Verarbeitete Dateien werden in einem Set gespeichert und nicht erneut verarbeitet.

    Args:
        watch_dir: Zu überwachendes Verzeichnis.
        config: DocSort-Konfiguration.
        on_result: Callback für jedes verarbeitete Ergebnis.
        poll_interval: Sekunden zwischen Prüfungen (Standard: 5).
        run_once: Nur einmal prüfen (für Tests).
    """
    if not watch_dir.is_dir():
        logger.error("Watchfolder existiert nicht: %s", watch_dir)
        return

    seen: set[Path] = set()
    extensions = set(config.supported_extensions)

    logger.info("Watchfolder gestartet: %s (Intervall: %.1fs)", watch_dir, poll_interval)
    logger.info("Ausgabe: %s | Modus: %s", config.output_dir, config.mode)

    try:
        while True:
            # Alle unterstützten Dateien im Verzeichnis finden
            current_files = set()
            for item in watch_dir.iterdir():
                if item.is_file() and item.suffix.lower() in extensions:
                    current_files.add(item)

            # Neue Dateien verarbeiten
            new_files = current_files - seen
            for file_path in sorted(new_files):
                logger.info("Neue Datei erkannt: %s", file_path.name)

                # Kurz warten falls die Datei noch geschrieben wird
                _wait_for_stable(file_path)

                result = process_file(file_path, config)
                seen.add(file_path)

                if result.success:
                    logger.info(
                        "✓ %s → %s (%s)",
                        file_path.name,
                        result.target,
                        "verschoben" if config.mode == "move" else "kopiert",
                    )
                else:
                    logger.error("✗ %s: %s", file_path.name, result.error)

                if on_result:
                    on_result(result)

            if run_once:
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Watchfolder gestoppt.")


def _wait_for_stable(file_path: Path, checks: int = 3, interval: float = 0.5) -> None:
    """Wartet bis eine Datei nicht mehr wächst (fertig geschrieben).

    Args:
        file_path: Pfad zur Datei.
        checks: Anzahl aufeinanderfolgender Prüfungen mit gleicher Größe.
        interval: Sekunden zwischen Prüfungen.
    """
    last_size = -1
    stable_count = 0

    for _ in range(checks * 3):  # max Versuche
        try:
            current_size = file_path.stat().st_size
        except OSError:
            time.sleep(interval)
            continue

        if current_size == last_size:
            stable_count += 1
            if stable_count >= checks:
                return
        else:
            stable_count = 0

        last_size = current_size
        time.sleep(interval)
