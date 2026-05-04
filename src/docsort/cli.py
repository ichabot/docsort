"""CLI-Interface für DocSort (Click-basiert)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from docsort.config import Config, load_config, save_config


def _setup_logging(verbose: bool = False, log_file: str = "") -> None:
    """Konfiguriert Logging (Konsole + optional Datei)."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s" if log_file else "%(levelname)s: %(message)s",
        handlers=handlers,
    )


@click.group()
@click.version_option(
    package_name="docsort",
    message="%(prog)s v%(version)s — KI-gestützt (Vibe Coding), MIT-Lizenz, keine Gewährleistung.",
)
def main() -> None:
    """DocSort — Automatische Dokumenten-Sortierung per OCR und LLM.

    ⚠️  Dieses Tool wurde mit KI-Unterstützung entwickelt und nutzt ungeprüfte
    Drittanbieter-Bibliotheken. Erstelle Backups und prüfe Ergebnisse.
    """


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default=None, help="Ausgabeverzeichnis (Standard: ./sorted)")
@click.option("--copy/--move", "do_copy", default=None, help="Dateien kopieren (Standard) oder verschieben")
@click.option("--dry-run", is_flag=True, default=False, help="Nur Vorschau anzeigen, nichts ausführen")
@click.option("--profile", default=None, help="LLM-Profil (lm-studio, ollama, openai, anthropic, gemini)")
@click.option("--llm-url", default=None, help="LLM API URL (überschreibt Profil)")
@click.option("--model", default=None, help="LLM Modellname (überschreibt Profil)")
@click.option("--api-key", default=None, help="API-Key (überschreibt Profil)")
@click.option("--no-gpu", is_flag=True, default=False, help="GPU deaktivieren")
@click.option("--batch-size", type=int, default=None, help="OCR Batch-Größe")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Pfad zur Config-Datei")
@click.option("-v", "--verbose", is_flag=True, help="Ausführliche Ausgabe")
def process(
    input_dir: Path,
    output_dir: Path | None,
    do_copy: bool | None,
    dry_run: bool,
    profile: str | None,
    llm_url: str | None,
    model: str | None,
    api_key: str | None,
    no_gpu: bool,
    batch_size: int | None,
    config_path: Path | None,
    verbose: bool,
) -> None:
    """Dokumente aus INPUT_DIR verarbeiten und sortieren."""
    # Config laden
    config = load_config(config_path)

    # CLI-Optionen überschreiben Config
    config.input_dir = input_dir
    if output_dir:
        config.output_dir = output_dir
    if do_copy is not None:
        config.mode = "copy" if do_copy else "move"
    config.dry_run = dry_run
    if no_gpu:
        config.gpu = False
    if batch_size:
        config.ocr_batch_size = batch_size

    # Profil anwenden
    if profile:
        config.apply_profile(profile)
    # Einzelne LLM-Optionen überschreiben Profil
    if llm_url:
        config.llm_base_url = llm_url
    if model:
        config.llm_model = model
    if api_key:
        config.llm_api_key = api_key

    _setup_logging(verbose, config.log_file)

    if dry_run:
        click.secho("=== DRY-RUN Modus (keine Änderungen) ===", fg="yellow", bold=True)

    p = config.get_active_profile()
    click.secho(f"Eingabe:  {input_dir.resolve()}", fg="blue")
    click.secho(f"Ausgabe:  {config.output_dir.resolve()}", fg="blue")
    click.secho(f"Modus:    {'Kopieren' if config.mode == 'copy' else 'Verschieben'}", fg="blue")
    click.secho(f"LLM:      {p.description or p.name} ({p.model or 'Standard'})", fg="blue")
    click.echo()

    from docsort.pipeline import collect_files, process_file

    files = collect_files(input_dir, config)
    if not files:
        click.secho("Keine unterstützten Dateien gefunden.", fg="red")
        sys.exit(1)

    click.secho(f"{len(files)} Datei(en) gefunden.\n", fg="green")

    ok_count = 0
    fail_count = 0
    low_conf_count = 0

    for i, file_path in enumerate(files, 1):
        result = process_file(file_path, config)

        if result.success:
            ok_count += 1
            target_rel = result.target.relative_to(config.output_dir) if result.target else "?"
            marker = "⚠" if result.low_confidence else "✓"
            color = "yellow" if result.low_confidence else "green"
            click.secho(f"  [{i}/{len(files)}] {marker} ", fg=color, nl=False)
            click.echo(f"{file_path.name} → {target_rel}")
            if result.low_confidence:
                low_conf_count += 1
                click.secho(f"           Konfidenz: {result.classification.confidence:.0%}", fg="yellow")
        else:
            fail_count += 1
            click.secho(f"  [{i}/{len(files)}] ✗ ", fg="red", nl=False)
            click.echo(f"{file_path.name} — {result.error}")

    click.echo()
    summary = f"Fertig: {ok_count} erfolgreich, {fail_count} Fehler"
    if low_conf_count:
        summary += f", {low_conf_count} unsicher"
    click.secho(summary, fg="green" if fail_count == 0 else "yellow", bold=True)


@main.command()
@click.option("--port", type=int, default=7860, help="Port für Web-UI (Standard: 7860)")
@click.option("--share", is_flag=True, help="Öffentlichen Gradio-Link erstellen")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Pfad zur Config-Datei")
def web(port: int, share: bool, config_path: Path | None) -> None:
    """Gradio Web-UI starten."""
    _setup_logging()

    config = load_config(config_path)

    click.secho("Starte DocSort Web-UI...", fg="green", bold=True)

    from docsort.web import create_ui

    app = create_ui(config=config)
    import gradio as gr
    app.launch(server_port=port, share=share, theme=gr.themes.Soft())


@main.command()
@click.option("--count", "-n", type=int, default=0, help="Anzahl Operationen rückgängig machen (0 = alle)")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Pfad zur Config-Datei")
def undo(count: int, config_path: Path | None) -> None:
    """Letzte Operationen rückgängig machen (basierend auf Undo-Log)."""
    config = load_config(config_path)

    from docsort.organizer import undo_last

    messages = undo_last(config, count)
    for msg in messages:
        if msg.startswith("✓"):
            click.secho(msg, fg="green")
        elif msg.startswith("⚠"):
            click.secho(msg, fg="yellow")
        else:
            click.secho(msg, fg="red")


@main.command()
@click.argument("watch_dir", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default=None, help="Ausgabeverzeichnis")
@click.option("--copy/--move", "do_copy", default=None, help="Kopieren oder verschieben")
@click.option("--profile", default=None, help="LLM-Profil")
@click.option("--interval", type=float, default=5.0, help="Prüf-Intervall in Sekunden (Standard: 5)")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Config-Datei")
@click.option("-v", "--verbose", is_flag=True, help="Ausführliche Ausgabe")
def watch(
    watch_dir: Path,
    output_dir: Path | None,
    do_copy: bool | None,
    profile: str | None,
    interval: float,
    config_path: Path | None,
    verbose: bool,
) -> None:
    """Verzeichnis überwachen und neue Dateien automatisch verarbeiten.

    Beispiel: docsort watch ./scans -o ./sorted
    """
    config = load_config(config_path)

    if output_dir:
        config.output_dir = output_dir
    if do_copy is not None:
        config.mode = "copy" if do_copy else "move"
    if profile:
        config.apply_profile(profile)

    _setup_logging(verbose, config.log_file)

    p = config.get_active_profile()
    click.secho("=== DocSort Watchfolder ===", fg="green", bold=True)
    click.secho(f"Überwache: {watch_dir.resolve()}", fg="blue")
    click.secho(f"Ausgabe:   {config.output_dir.resolve()}", fg="blue")
    click.secho(f"Modus:     {'Kopieren' if config.mode == 'copy' else 'Verschieben'}", fg="blue")
    click.secho(f"LLM:       {p.description or p.name}", fg="blue")
    click.secho(f"Intervall: {interval}s", fg="blue")
    click.secho("Drücke Ctrl+C zum Beenden.\n", fg="yellow")

    from docsort.watcher import watch_directory

    def on_result(result):
        if result.success:
            click.secho(f"  ✓ {result.source.name} → {result.target.name}", fg="green")
        else:
            click.secho(f"  ✗ {result.source.name}: {result.error}", fg="red")

    watch_directory(
        watch_dir=watch_dir,
        config=config,
        on_result=on_result,
        poll_interval=interval,
    )


@main.command("init")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Wo die Config gespeichert werden soll")
def init_config(output: Path | None) -> None:
    """Erstellt eine Beispiel-Config-Datei (docsort.yaml)."""
    config = Config()
    path = save_config(config, output)
    click.secho(f"Config erstellt: {path}", fg="green")
    click.echo("Bearbeite die Datei um Profile, Dokumenttypen und Prompt anzupassen.")


@main.command("profiles")
def list_profiles() -> None:
    """Zeigt alle verfügbaren LLM-Profile."""
    config = load_config()
    click.secho("Verfügbare LLM-Profile:\n", bold=True)
    for name, profile in config.profiles.items():
        active = " ← aktiv" if name == config.active_profile else ""
        click.secho(f"  {name}", fg="green" if active else "white", nl=False, bold=bool(active))
        click.echo(f"  {profile.description or ''}{active}")
        click.echo(f"    Provider: {profile.provider}")
        click.echo(f"    URL:      {profile.base_url}")
        click.echo(f"    Model:    {profile.model or '(Standard)'}")
        key_display = "***" if profile.api_key and profile.api_key not in ("lm-studio", "ollama") else profile.api_key
        click.echo(f"    API-Key:  {key_display}")
        click.echo()
