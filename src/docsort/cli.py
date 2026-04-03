"""CLI-Interface für DocSort (Click-basiert)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from docsort.config import Config


def _setup_logging(verbose: bool = False) -> None:
    """Konfiguriert Logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


@click.group()
@click.version_option(package_name="docsort")
def main() -> None:
    """DocSort — Automatische Dokumenten-Sortierung per OCR und LLM."""


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default="./sorted", help="Ausgabeverzeichnis (Standard: ./sorted)")
@click.option("--copy/--move", "do_copy", default=True, help="Dateien kopieren (Standard) oder verschieben")
@click.option("--dry-run", is_flag=True, help="Nur Vorschau anzeigen, nichts ausführen")
@click.option("--llm-url", default="http://localhost:1234/v1", help="LLM API URL")
@click.option("--model", default="", help="LLM Modellname")
@click.option("--no-gpu", is_flag=True, help="GPU deaktivieren")
@click.option("--batch-size", type=int, default=32, help="OCR/Layout Batch-Größe")
@click.option("-v", "--verbose", is_flag=True, help="Ausführliche Ausgabe")
def process(
    input_dir: Path,
    output_dir: Path,
    do_copy: bool,
    dry_run: bool,
    llm_url: str,
    model: str,
    no_gpu: bool,
    batch_size: int,
    verbose: bool,
) -> None:
    """Dokumente aus INPUT_DIR verarbeiten und sortieren."""
    _setup_logging(verbose)

    config = Config(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_base_url=llm_url,
        llm_model=model,
        mode="copy" if do_copy else "move",
        dry_run=dry_run,
        gpu=not no_gpu,
        ocr_batch_size=batch_size,
        layout_batch_size=batch_size,
    )

    if dry_run:
        click.secho("=== DRY-RUN Modus (keine Änderungen) ===", fg="yellow", bold=True)

    click.secho(f"Eingabe:  {input_dir.resolve()}", fg="blue")
    click.secho(f"Ausgabe:  {output_dir.resolve()}", fg="blue")
    click.secho(f"Modus:    {'Kopieren' if do_copy else 'Verschieben'}", fg="blue")
    click.echo()

    from docsort.pipeline import collect_files, process_file

    files = collect_files(input_dir, config)
    if not files:
        click.secho("Keine unterstützten Dateien gefunden.", fg="red")
        sys.exit(1)

    click.secho(f"{len(files)} Datei(en) gefunden.\n", fg="green")

    ok_count = 0
    fail_count = 0

    for i, file_path in enumerate(files, 1):
        result = process_file(file_path, config)

        if result.success:
            ok_count += 1
            target_rel = result.target.relative_to(output_dir) if result.target else "?"
            click.secho(f"  [{i}/{len(files)}] ✓ ", fg="green", nl=False)
            click.echo(f"{file_path.name} → {target_rel}")
        else:
            fail_count += 1
            click.secho(f"  [{i}/{len(files)}] ✗ ", fg="red", nl=False)
            click.echo(f"{file_path.name} — {result.error}")

    click.echo()
    click.secho(f"Fertig: {ok_count} erfolgreich, {fail_count} Fehler.", fg="green" if fail_count == 0 else "yellow", bold=True)


@main.command()
@click.option("--port", type=int, default=7860, help="Port für Web-UI (Standard: 7860)")
@click.option("--share", is_flag=True, help="Öffentlichen Gradio-Link erstellen")
@click.option("--llm-url", default="http://localhost:1234/v1", help="LLM API URL")
@click.option("--model", default="", help="LLM Modellname")
@click.option("--no-gpu", is_flag=True, help="GPU deaktivieren")
def web(
    port: int,
    share: bool,
    llm_url: str,
    model: str,
    no_gpu: bool,
) -> None:
    """Gradio Web-UI starten."""
    _setup_logging()

    click.secho("Starte DocSort Web-UI...", fg="green", bold=True)

    from docsort.web import create_ui

    app = create_ui(
        llm_url=llm_url,
        llm_model=model,
        gpu=not no_gpu,
    )
    app.launch(server_port=port, share=share)
