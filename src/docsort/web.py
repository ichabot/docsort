"""Gradio Web-UI für DocSort — mit LLM-Profilen, Prompt-Editor, PDF-Preview und Settings."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

# Gradio Telemetrie deaktivieren
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

import gradio as gr

from docsort.config import Config, load_config, save_config, LLMProfile
from docsort.classifier import Classification, sanitize_short_info

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

    profile_names = list(config.profiles.keys())

    def _build_config(
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        output_dir: str,
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

        if doc_types_text.strip():
            cfg.doc_types = [t.strip() for t in doc_types_text.split("\n") if t.strip()]

        if profile_name == "(Eigene Einstellungen)":
            cfg.llm_provider = custom_provider
            cfg.llm_base_url = custom_url
            cfg.llm_model = custom_model
            cfg.llm_api_key = custom_key
            cfg.active_profile = "custom"
            cfg.profiles["custom"] = LLMProfile(
                name="custom", provider=custom_provider,
                base_url=custom_url, model=custom_model, api_key=custom_key,
            )
        elif profile_name in cfg.profiles:
            cfg.apply_profile(profile_name)
            if custom_url:
                cfg.llm_base_url = custom_url
            if custom_model:
                cfg.llm_model = custom_model
            if custom_key:
                cfg.llm_api_key = custom_key

        return cfg

    def on_profile_change(profile_name: str) -> tuple[str, str, str, str]:
        if profile_name == "(Eigene Einstellungen)":
            return "", "", "", "openai"
        if profile_name in config.profiles:
            p = config.profiles[profile_name]
            return p.base_url, p.model, p.api_key, p.provider
        return "", "", "", "openai"

    def analyze(
        files: list[str] | None,
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        output_dir: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
    ) -> tuple[list[list[str]], str, list[dict], dict]:
        """Analysiert Dateien und cached Ergebnisse."""
        if not files:
            return [], "⚠️ Keine Dateien hochgeladen.", [], {}

        from docsort.pipeline import process_file

        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=True,
        )

        rows: list[list[str]] = []
        log_lines: list[str] = []
        cache: list[dict] = []
        file_map: dict[str, str] = {}  # filename → full path (für Preview)

        for i, file_path_str in enumerate(files, 1):
            file_path = Path(file_path_str)
            file_map[file_path.name] = file_path_str
            result = process_file(file_path, cfg)

            if result.success and result.classification:
                c = result.classification
                conf_str = f"{c.confidence:.0%}"
                if result.low_confidence:
                    conf_str += " ⚠️"

                # Zielpfad aufteilen in Ordner + Dateiname
                if result.target:
                    target_dir = str(result.target.parent)
                    target_file = result.target.name
                else:
                    target_dir = "—"
                    target_file = "—"

                rows.append([
                    file_path.name,
                    c.doc_type,
                    c.short_info,
                    c.doc_date,
                    conf_str,
                    target_dir,
                    target_file,
                ])
                cache.append({
                    "file_path": file_path_str,
                    "doc_type": c.doc_type,
                    "short_info": c.short_info,
                    "doc_date": c.doc_date,
                    "confidence": c.confidence,
                })
                status = "⚠️" if result.low_confidence else "✓"
                log_lines.append(f"[{i}/{len(files)}] {status} {file_path.name} → {c.doc_type} ({c.confidence:.0%})")
            else:
                rows.append([
                    file_path.name, "FEHLER",
                    result.error or "Unbekannt", "—", "—", "—", "—",
                ])
                log_lines.append(f"[{i}/{len(files)}] ✗ {file_path.name}: {result.error}")

        log_text = "\n".join(log_lines)
        log_text += f"\n\n💡 Du kannst Typ, Kurzinfo und Datum in der Tabelle bearbeiten bevor du ausführst."
        return rows, log_text, cache, file_map

    def execute(
        files: list[str] | None,
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        output_dir: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
        cached_results: list[dict],
        edited_table: Any,
    ) -> str:
        """Führt Organisation durch — nutzt Cache + editierte Tabelle."""
        if not files:
            return "⚠️ Keine Dateien hochgeladen."

        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=False,
        )

        log_lines: list[str] = []
        use_cache = cached_results and len(cached_results) > 0

        # Editierte Tabelle auslesen
        edits: dict[str, dict] = {}
        if edited_table is not None and use_cache:
            try:
                table_rows: list[list] = []
                if hasattr(edited_table, "values"):
                    # Pandas DataFrame
                    table_rows = edited_table.values.tolist()
                elif isinstance(edited_table, dict) and "data" in edited_table:
                    table_rows = edited_table["data"]
                elif isinstance(edited_table, list):
                    table_rows = edited_table

                logger.info("Editierte Tabelle: %d Zeilen gelesen", len(table_rows))

                for row in table_rows:
                    if len(row) >= 5 and str(row[1]) != "FEHLER":
                        edits[str(row[0])] = {
                            "doc_type": str(row[1]).strip(),
                            "short_info": str(row[2]).strip(),
                            "doc_date": str(row[3]).strip(),
                        }
            except Exception as exc:
                logger.warning("Konnte editierte Tabelle nicht lesen: %s", exc)
                log_lines.append(f"⚠️ Tabellen-Edits konnten nicht gelesen werden: {exc}\n")

        if use_cache:
            log_lines.append("⚡ Nutze gecachte Analyse-Ergebnisse (kein erneutes OCR/LLM).\n")

            from docsort.organizer import organize

            for i, entry in enumerate(cached_results, 1):
                file_path = Path(entry["file_path"])
                filename = file_path.name

                # Editierte Werte übernehmen
                orig_type = entry["doc_type"]
                orig_info = entry["short_info"]
                orig_date = entry["doc_date"]

                if filename in edits:
                    e = edits[filename]
                    doc_type = e["doc_type"] or orig_type
                    short_info = sanitize_short_info(e["short_info"]) if e["short_info"] else orig_info
                    doc_date = e["doc_date"] or orig_date
                    was_edited = (doc_type != orig_type or short_info != orig_info or doc_date != orig_date)
                else:
                    doc_type = orig_type
                    short_info = orig_info
                    doc_date = orig_date
                    was_edited = False

                classification = Classification(
                    doc_type=doc_type,
                    short_info=short_info,
                    doc_date=doc_date,
                    confidence=entry["confidence"],
                )

                try:
                    result = organize(file_path, classification, cfg)
                    if result.success:
                        action = "verschoben" if cfg.mode == "move" else "kopiert"
                        edit_marker = " ✏️" if was_edited else ""
                        log_lines.append(
                            f"[{i}/{len(cached_results)}] ✓ {filename} → "
                            f"{result.target.parent.name}/{result.target.name} ({action}){edit_marker}"
                        )
                    else:
                        log_lines.append(f"[{i}/{len(cached_results)}] ✗ {filename}: {result.error}")
                except Exception as exc:
                    log_lines.append(f"[{i}/{len(cached_results)}] ✗ {filename}: {exc}")

            ok = sum(1 for l in log_lines if "✓" in l)
            total = len(cached_results)
        else:
            log_lines.append("ℹ️ Keine Vorschau vorhanden — führe vollständige Analyse durch.\n")

            from docsort.pipeline import process_file

            for i, file_path_str in enumerate(files, 1):
                file_path = Path(file_path_str)
                result = process_file(file_path, cfg)

                if result.success:
                    action = "verschoben" if cfg.mode == "move" else "kopiert"
                    conf_info = ""
                    if result.low_confidence:
                        conf_info = f" ⚠️ Konfidenz: {result.classification.confidence:.0%}"
                    log_lines.append(
                        f"[{i}/{len(files)}] ✓ {file_path.name} → {result.target} ({action}){conf_info}"
                    )
                else:
                    log_lines.append(f"[{i}/{len(files)}] ✗ {file_path.name}: {result.error}")

            ok = sum(1 for l in log_lines if "✓" in l)
            total = len(files)

        fail = total - ok
        log_lines.append(f"\n{'='*50}")
        log_lines.append(f"Fertig: {ok}/{total} erfolgreich, {fail} Fehler.")

        return "\n".join(log_lines)

    def show_preview(
        file_selection: str,
        file_map: dict[str, str],
    ) -> str:
        """Zeigt PDF/Bild-Preview für die ausgewählte Datei."""
        if not file_selection or not file_map:
            return "<p style='color: gray; text-align: center; padding: 40px;'>Datei oben auswählen um Vorschau anzuzeigen.</p>"

        file_path_str = file_map.get(file_selection, "")
        if not file_path_str:
            return "<p style='color: red;'>Datei nicht gefunden.</p>"

        file_path = Path(file_path_str)
        if not file_path.exists():
            return f"<p style='color: red;'>Datei nicht mehr vorhanden: {file_path}</p>"

        suffix = file_path.suffix.lower()

        # Bilder direkt anzeigen
        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"):
            try:
                with open(file_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                mime = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".bmp": "image/bmp",
                    ".webp": "image/webp", ".tiff": "image/tiff", ".tif": "image/tiff",
                }.get(suffix, "image/png")
                return (
                    f'<div style="text-align:center; max-height:600px; overflow:auto;">'
                    f'<img src="data:{mime};base64,{data}" style="max-width:100%; max-height:580px;" />'
                    f'</div>'
                )
            except Exception as exc:
                return f"<p style='color: red;'>Fehler beim Laden: {exc}</p>"

        # PDFs per iframe anzeigen
        if suffix == ".pdf":
            try:
                with open(file_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                return (
                    f'<iframe src="data:application/pdf;base64,{data}" '
                    f'width="100%" height="600px" style="border: 1px solid #ccc; border-radius: 8px;"></iframe>'
                )
            except Exception as exc:
                return f"<p style='color: red;'>PDF konnte nicht geladen werden: {exc}</p>"

        # Andere Formate: nur Info
        size_kb = file_path.stat().st_size / 1024
        return (
            f'<div style="text-align:center; padding:40px; color:gray;">'
            f'<p style="font-size:48px;">📄</p>'
            f'<p><strong>{file_path.name}</strong></p>'
            f'<p>{suffix.upper()[1:]} Datei — {size_kb:.1f} KB</p>'
            f'<p>Vorschau nicht verfügbar für dieses Format.</p>'
            f'</div>'
        )

    def on_table_select(
        evt: gr.SelectData,
        edited_table: Any,
        file_map: dict[str, str],
    ) -> tuple[str, str]:
        """Wenn eine Zeile in der Tabelle angeklickt wird → Preview + Dropdown updaten."""
        if edited_table is None or not file_map:
            return gr.update(), ""

        try:
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if hasattr(edited_table, "iloc"):
                row = edited_table.iloc[row_idx]
                filename = str(row.iloc[0])
            elif isinstance(edited_table, list) and len(edited_table) > row_idx:
                filename = str(edited_table[row_idx][0])
            else:
                return gr.update(), ""

            return gr.update(value=filename), show_preview(filename, file_map)
        except Exception:
            return gr.update(), ""

    def update_file_dropdown(
        cached_results: list[dict],
        file_map: dict[str, str],
    ) -> Any:
        """Aktualisiert das Datei-Dropdown nach der Analyse."""
        if not file_map:
            return gr.update(choices=[], value=None)
        names = list(file_map.keys())
        return gr.update(choices=names, value=names[0] if names else None)

    def do_save_config(
        profile_name: str,
        custom_url: str,
        custom_model: str,
        custom_key: str,
        custom_provider: str,
        output_dir: str,
        mode: str,
        folder_template: str,
        system_prompt: str,
        doc_types_text: str,
        confidence_threshold: float,
        max_retries: int,
    ) -> str:
        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries,
        )
        path = save_config(cfg)
        return f"✅ Config gespeichert: {path}"

    # ==========================================================
    # UI Layout
    # ==========================================================
    with gr.Blocks(title="DocSort", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 DocSort\nAutomatische Dokumenten-Klassifizierung und -Sortierung")

        # States
        cached_state = gr.State(value=[])
        file_map_state = gr.State(value={})

        with gr.Tabs():
            # ======== Tab 1: Verarbeitung ========
            with gr.Tab("📁 Verarbeitung"):
                gr.Markdown("### 📁 Dateien hochladen")
                file_input = gr.File(
                    label="Dokumente hochladen (Drag & Drop oder klicken)",
                    file_count="multiple",
                    type="filepath",
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analysieren (Vorschau)", variant="secondary", scale=1)
                    execute_btn = gr.Button("▶️ Ausführen", variant="primary", scale=1)

                gr.Markdown(
                    "### 📊 Ergebnis\n"
                    "Du kannst **Typ**, **Kurzinfo** und **Datum** direkt in der Tabelle bearbeiten. "
                    "Klicke eine Zeile an um die Datei-Vorschau zu sehen."
                )
                result_table = gr.Dataframe(
                    headers=["Datei", "Typ", "Kurzinfo", "Datum", "Konfidenz", "Zielpfad", "Zieldatei"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    col_count=(7, "fixed"),
                    interactive=True,
                    label="Vorschau (editierbar)",
                )
                log_output = gr.Textbox(label="Log", lines=8, interactive=False)

                # --- Preview-Bereich ---
                gr.Markdown("### 👁️ Dokument-Vorschau")
                with gr.Row():
                    with gr.Column(scale=1):
                        preview_dropdown = gr.Dropdown(
                            label="Datei auswählen",
                            choices=[],
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        preview_html = gr.HTML(
                            value="<p style='color: gray; text-align: center; padding: 40px;'>Nach der Analyse eine Datei auswählen oder Zeile anklicken.</p>",
                            label="Vorschau",
                        )

            # ======== Tab 2: Einstellungen ========
            with gr.Tab("⚙️ Einstellungen"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🤖 LLM-Profil")
                        profile_dropdown = gr.Dropdown(
                            choices=profile_names + ["(Eigene Einstellungen)"],
                            value=config.active_profile,
                            label="Profil",
                        )
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

                        gr.Markdown("---")
                        gr.Markdown("### 📂 Ausgabe")
                        output_dir_input = gr.Textbox(
                            label="Ausgabeverzeichnis",
                            value=str(config.output_dir),
                        )
                        mode_input = gr.Radio(
                            choices=["copy", "move"],
                            value=config.mode,
                            label="Modus",
                        )

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
                            lines=8,
                        )

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

                        gr.Markdown("### 💾 Config speichern")
                        save_btn = gr.Button("💾 Einstellungen speichern (docsort.yaml)", variant="secondary")
                        save_status = gr.Textbox(label="Status", interactive=False)

            # ======== Tab 3: System-Prompt ========
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

        # ==========================================================
        # Inputs (Reihenfolge muss zu _build_config passen)
        # ==========================================================
        settings_inputs = [
            profile_dropdown,
            custom_url_input, custom_model_input, custom_key_input,
            custom_provider_input,
            output_dir_input, mode_input, folder_template_input,
            system_prompt_input, doc_types_input,
            confidence_input, retry_input,
        ]

        # ==========================================================
        # Event Bindings
        # ==========================================================
        profile_dropdown.change(
            fn=on_profile_change,
            inputs=[profile_dropdown],
            outputs=[custom_url_input, custom_model_input, custom_key_input, custom_provider_input],
        )

        # Analysieren → Tabelle + Log + Cache + FileMap
        analyze_btn.click(
            fn=analyze,
            inputs=[file_input] + settings_inputs,
            outputs=[result_table, log_output, cached_state, file_map_state],
        ).then(
            fn=update_file_dropdown,
            inputs=[cached_state, file_map_state],
            outputs=[preview_dropdown],
        )

        # Ausführen → nutzt Cache + editierte Tabelle
        execute_btn.click(
            fn=execute,
            inputs=[file_input] + settings_inputs + [cached_state, result_table],
            outputs=[log_output],
        )

        # Tabelle anklicken → Preview
        result_table.select(
            fn=on_table_select,
            inputs=[result_table, file_map_state],
            outputs=[preview_dropdown, preview_html],
        )

        # Preview-Dropdown → Preview
        preview_dropdown.change(
            fn=show_preview,
            inputs=[preview_dropdown, file_map_state],
            outputs=[preview_html],
        )

        # Config speichern
        save_btn.click(
            fn=do_save_config,
            inputs=settings_inputs,
            outputs=[save_status],
        )

    return app
