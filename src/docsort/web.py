"""Gradio Web-UI für DocSort — mit LLM-Profilen, Prompt-Editor und Settings."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Gradio Telemetrie deaktivieren
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

import gradio as gr

from docsort.config import Config, load_config, save_config, BUILTIN_PROFILES, LLMProfile

logger = logging.getLogger(__name__)


def create_ui(config: Config | None = None) -> gr.Blocks:
    """Erstellt die Gradio Web-UI.

    Args:
        config: Optionale Config (sonst wird docsort.yaml geladen).

    Returns:
        Gradio Blocks App.
    """
    if config is None:
        config = load_config()

    # Profil-Namen für Dropdown
    profile_names = list(config.profiles.keys())

    def _build_config(
        output_dir: str,
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
        dry_run: bool = False,
    ) -> Config:
        """Baut Config-Objekt aus Web-UI Feldern."""
        cfg = Config(
            output_dir=Path(output_dir),
            mode=mode.lower(),
            dry_run=dry_run,
            gpu=config.gpu,
            ocr_batch_size=config.ocr_batch_size,
            layout_batch_size=config.layout_batch_size,
            folder_template=folder_template,
            system_prompt=system_prompt,
            confidence_threshold=confidence_threshold,
            max_retries=max_retries,
            undo_log=config.undo_log,
            log_file=config.log_file,
        )

        # Dokumenttypen aus Textarea parsen
        if doc_types_text.strip():
            cfg.doc_types = [t.strip() for t in doc_types_text.split("\n") if t.strip()]

        # LLM-Profil
        if profile_name == "(Eigene Einstellungen)":
            cfg.llm_provider = custom_provider
            cfg.llm_base_url = custom_url
            cfg.llm_model = custom_model
            cfg.llm_api_key = custom_key
            cfg.active_profile = "custom"
            cfg.profiles["custom"] = LLMProfile(
                name="custom",
                provider=custom_provider,
                base_url=custom_url,
                model=custom_model,
                api_key=custom_key,
            )
        elif profile_name in cfg.profiles:
            cfg.apply_profile(profile_name)
            # Überschreibe mit custom-Feldern wenn ausgefüllt
            if custom_url:
                cfg.llm_base_url = custom_url
            if custom_model:
                cfg.llm_model = custom_model
            if custom_key:
                cfg.llm_api_key = custom_key

        return cfg

    def on_profile_change(profile_name: str) -> tuple[str, str, str, str]:
        """Aktualisiert die LLM-Felder wenn ein Profil ausgewählt wird."""
        if profile_name == "(Eigene Einstellungen)":
            return "", "", "", "openai"

        if profile_name in config.profiles:
            p = config.profiles[profile_name]
            return p.base_url, p.model, p.api_key, p.provider

        return "", "", "", "openai"

    def analyze(
        files: list[str] | None,
        output_dir: str,
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
    ) -> tuple[list[list[str]], str]:
        """Dry-Run: Analysiert Dateien und zeigt Vorschau."""
        if not files:
            return [], "⚠️ Keine Dateien hochgeladen."

        from docsort.pipeline import process_file

        cfg = _build_config(
            output_dir, profile_name, custom_url, custom_model, custom_key,
            custom_provider, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=True,
        )

        rows: list[list[str]] = []
        log_lines: list[str] = []

        for i, file_path_str in enumerate(files, 1):
            file_path = Path(file_path_str)
            result = process_file(file_path, cfg)

            if result.success and result.classification:
                c = result.classification
                target = str(result.target) if result.target else "—"
                conf_str = f"{c.confidence:.0%}"
                if result.low_confidence:
                    conf_str += " ⚠️"
                rows.append([
                    file_path.name,
                    c.doc_type,
                    c.short_info,
                    c.doc_date,
                    conf_str,
                    target,
                ])
                status = "⚠️" if result.low_confidence else "✓"
                log_lines.append(f"[{i}/{len(files)}] {status} {file_path.name} → {c.doc_type} ({c.confidence:.0%})")
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
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
    ) -> str:
        """Führt die tatsächliche Organisation durch."""
        if not files:
            return "⚠️ Keine Dateien hochgeladen."

        from docsort.pipeline import process_file

        cfg = _build_config(
            output_dir, profile_name, custom_url, custom_model, custom_key,
            custom_provider, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=False,
        )

        log_lines: list[str] = []

        for i, file_path_str in enumerate(files, 1):
            file_path = Path(file_path_str)
            result = process_file(file_path, cfg)

            if result.success:
                action = "verschoben" if mode.lower() == "move" else "kopiert"
                conf_info = ""
                if result.low_confidence:
                    conf_info = f" ⚠️ Konfidenz: {result.classification.confidence:.0%}"
                log_lines.append(f"[{i}/{len(files)}] ✓ {file_path.name} → {result.target} ({action}){conf_info}")
            else:
                log_lines.append(f"[{i}/{len(files)}] ✗ {file_path.name}: {result.error}")

        ok = sum(1 for l in log_lines if "✓" in l)
        fail = len(log_lines) - ok
        log_lines.append(f"\n{'='*50}")
        log_lines.append(f"Fertig: {ok}/{len(files)} erfolgreich, {fail} Fehler.")

        return "\n".join(log_lines)

    def do_save_config(
        output_dir: str,
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
    ) -> str:
        """Speichert aktuelle Einstellungen als docsort.yaml."""
        cfg = _build_config(
            output_dir, profile_name, custom_url, custom_model, custom_key,
            custom_provider, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries,
        )
        path = save_config(cfg)
        return f"✅ Config gespeichert: {path}"

    # === UI Layout ===
    with gr.Blocks(title="DocSort", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 DocSort\nAutomatische Dokumenten-Klassifizierung und -Sortierung")

        with gr.Tabs():
            # === Tab 1: Verarbeitung ===
            with gr.Tab("📁 Verarbeitung"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Basis-Einstellungen")
                        output_dir_input = gr.Textbox(
                            label="Ausgabeverzeichnis",
                            value=str(config.output_dir),
                        )
                        mode_input = gr.Radio(
                            choices=["copy", "move"],
                            value=config.mode,
                            label="Modus",
                        )

                        gr.Markdown("### 🤖 LLM-Profil")
                        profile_dropdown = gr.Dropdown(
                            choices=profile_names + ["(Eigene Einstellungen)"],
                            value=config.active_profile,
                            label="Profil",
                        )
                        with gr.Accordion("LLM Details", open=False):
                            active = config.get_active_profile()
                            custom_provider_input = gr.Radio(
                                choices=["openai", "anthropic"],
                                value=active.provider,
                                label="Provider",
                            )
                            custom_url_input = gr.Textbox(
                                label="API URL",
                                value=active.base_url,
                            )
                            custom_model_input = gr.Textbox(
                                label="Modellname",
                                value=active.model,
                            )
                            custom_key_input = gr.Textbox(
                                label="API-Key",
                                value=active.api_key,
                                type="password",
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
                log_output = gr.Textbox(label="Log", lines=10, interactive=False)

            # === Tab 2: Einstellungen ===
            with gr.Tab("⚙️ Einstellungen"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📂 Ordnerstruktur")
                        folder_template_input = gr.Textbox(
                            label="Ordner-Template",
                            value=config.folder_template,
                            info="Variablen: {doc_type}, {year}, {month}, {filename}",
                        )
                        gr.Markdown(
                            "**Beispiele:**\n"
                            "- `{doc_type}/{year}/{filename}` → Rechnung/2026/...\n"
                            "- `{year}/{doc_type}/{filename}` → 2026/Rechnung/...\n"
                            "- `{doc_type}/{year}/{month}/{filename}` → Rechnung/2026/03/...\n"
                            "- `{filename}` → Flache Struktur"
                        )

                        gr.Markdown("### 📋 Dokumenttypen")
                        doc_types_input = gr.Textbox(
                            label="Dokumenttypen (einer pro Zeile)",
                            value="\n".join(config.doc_types),
                            lines=10,
                        )

                    with gr.Column():
                        gr.Markdown("### 🎯 Qualität")
                        confidence_input = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=config.confidence_threshold,
                            label="Confidence-Schwelle",
                            info="Unter diesem Wert wird eine Warnung angezeigt",
                        )
                        retry_input = gr.Slider(
                            minimum=0, maximum=5, step=1,
                            value=config.max_retries,
                            label="Max. Retries bei LLM-Fehler",
                        )

                        gr.Markdown("### 💾 Config")
                        save_btn = gr.Button("💾 Einstellungen speichern (docsort.yaml)", variant="secondary")
                        save_status = gr.Textbox(label="Status", interactive=False)

            # === Tab 3: System-Prompt ===
            with gr.Tab("📝 System-Prompt"):
                gr.Markdown(
                    "### System-Prompt anpassen\n"
                    "Hier kannst du den Prompt bearbeiten, der an das LLM gesendet wird. "
                    "Verwende `{doc_types}` als Platzhalter für die Dokumenttypen-Liste."
                )
                system_prompt_input = gr.Textbox(
                    label="System-Prompt",
                    value=config.system_prompt,
                    lines=25,
                )

        # Gemeinsame Inputs für analyze/execute
        common_inputs = [
            file_input, output_dir_input, profile_dropdown,
            custom_url_input, custom_model_input, custom_key_input,
            custom_provider_input, mode_input, folder_template_input,
            system_prompt_input, doc_types_input, confidence_input, retry_input,
        ]

        # === Event Bindings ===
        profile_dropdown.change(
            fn=on_profile_change,
            inputs=[profile_dropdown],
            outputs=[custom_url_input, custom_model_input, custom_key_input, custom_provider_input],
        )

        analyze_btn.click(
            fn=analyze,
            inputs=common_inputs,
            outputs=[result_table, log_output],
        )
        execute_btn.click(
            fn=execute,
            inputs=common_inputs,
            outputs=[log_output],
        )

        save_btn.click(
            fn=do_save_config,
            inputs=[
                output_dir_input, profile_dropdown,
                custom_url_input, custom_model_input, custom_key_input,
                custom_provider_input, mode_input, folder_template_input,
                system_prompt_input, doc_types_input, confidence_input, retry_input,
            ],
            outputs=[save_status],
        )

    return app
