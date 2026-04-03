"""Gradio Web-UI für DocSort."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr

from docsort.config import Config

logger = logging.getLogger(__name__)


def create_ui(
    llm_url: str = "http://localhost:1234/v1",
    llm_model: str = "",
    gpu: bool = True,
) -> gr.Blocks:
    """Erstellt die Gradio Web-UI.

    Args:
        llm_url: LLM API URL.
        llm_model: LLM Modellname.
        gpu: GPU-Beschleunigung nutzen.

    Returns:
        Gradio Blocks App.
    """

    def analyze(
        files: list[str] | None,
        output_dir: str,
        llm_url_input: str,
        model_input: str,
        mode: str,
    ) -> tuple[list[list[str]], str]:
        """Dry-Run: Analysiert Dateien und zeigt Vorschau."""
        if not files:
            return [], "⚠️ Keine Dateien hochgeladen."

        from docsort.pipeline import process_file

        config = Config(
            output_dir=Path(output_dir),
            llm_base_url=llm_url_input or llm_url,
            llm_model=model_input or llm_model,
            mode=mode.lower(),
            dry_run=True,
            gpu=gpu,
        )

        rows: list[list[str]] = []
        log_lines: list[str] = []

        for i, file_path_str in enumerate(files, 1):
            file_path = Path(file_path_str)
            result = process_file(file_path, config)

            if result.success and result.classification:
                c = result.classification
                target = str(result.target) if result.target else "—"
                rows.append([
                    file_path.name,
                    c.doc_type,
                    c.short_info,
                    c.doc_date,
                    f"{c.confidence:.0%}",
                    target,
                ])
                log_lines.append(f"[{i}/{len(files)}] ✓ {file_path.name} → {c.doc_type}")
            else:
                rows.append([
                    file_path.name,
                    "FEHLER",
                    result.error or "Unbekannt",
                    "—",
                    "—",
                    "—",
                ])
                log_lines.append(f"[{i}/{len(files)}] ✗ {file_path.name}: {result.error}")

        log_text = "\n".join(log_lines)
        return rows, log_text

    def execute(
        files: list[str] | None,
        output_dir: str,
        llm_url_input: str,
        model_input: str,
        mode: str,
    ) -> str:
        """Führt die tatsächliche Organisation durch."""
        if not files:
            return "⚠️ Keine Dateien hochgeladen."

        from docsort.pipeline import process_file

        config = Config(
            output_dir=Path(output_dir),
            llm_base_url=llm_url_input or llm_url,
            llm_model=model_input or llm_model,
            mode=mode.lower(),
            dry_run=False,
            gpu=gpu,
        )

        log_lines: list[str] = []

        for i, file_path_str in enumerate(files, 1):
            file_path = Path(file_path_str)
            result = process_file(file_path, config)

            if result.success:
                action = "verschoben" if mode.lower() == "move" else "kopiert"
                log_lines.append(f"[{i}/{len(files)}] ✓ {file_path.name} → {result.target} ({action})")
            else:
                log_lines.append(f"[{i}/{len(files)}] ✗ {file_path.name}: {result.error}")

        ok = sum(1 for l in log_lines if "✓" in l)
        fail = len(log_lines) - ok
        log_lines.append(f"\n{'='*50}")
        log_lines.append(f"Fertig: {ok}/{len(files)} erfolgreich, {fail} Fehler.")

        return "\n".join(log_lines)

    # --- UI Layout ---
    with gr.Blocks(title="DocSort", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 DocSort\nAutomatische Dokumenten-Klassifizierung und -Sortierung")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Einstellungen")
                output_dir_input = gr.Textbox(
                    label="Ausgabeverzeichnis",
                    value="./sorted",
                    placeholder="/pfad/zum/zielordner",
                )
                llm_url_input = gr.Textbox(
                    label="LLM API URL",
                    value=llm_url,
                )
                model_input = gr.Textbox(
                    label="Modellname",
                    value=llm_model,
                    placeholder="(leer = Standardmodell in LM Studio)",
                )
                mode_input = gr.Radio(
                    choices=["copy", "move"],
                    value="copy",
                    label="Modus",
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📁 Dateien")
                file_input = gr.File(
                    label="Dokumente hochladen",
                    file_count="multiple",
                    type="filepath",
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analysieren (Vorschau)", variant="secondary", scale=1)
                    execute_btn = gr.Button("▶️ Ausführen", variant="primary", scale=1)

        gr.Markdown("### 📊 Ergebnis")
        result_table = gr.Dataframe(
            headers=["Datei", "Typ", "Kurzinfo", "Datum", "Konfidenz", "Zielpfad"],
            datatype=["str", "str", "str", "str", "str", "str"],
            label="Vorschau",
        )
        log_output = gr.Textbox(
            label="Log",
            lines=10,
            interactive=False,
        )

        # --- Event Bindings ---
        analyze_btn.click(
            fn=analyze,
            inputs=[file_input, output_dir_input, llm_url_input, model_input, mode_input],
            outputs=[result_table, log_output],
        )
        execute_btn.click(
            fn=execute,
            inputs=[file_input, output_dir_input, llm_url_input, model_input, mode_input],
            outputs=[log_output],
        )

    return app
