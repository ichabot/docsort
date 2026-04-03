"""Gradio Web-UI für DocSort — mit Seitenpanel, PDF-Preview und Edit-Feldern."""

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
    """Erstellt die Gradio Web-UI."""
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

    # ----------------------------------------------------------
    # Analyze
    # ----------------------------------------------------------
    def analyze(
        files: list[str] | None,
        profile_name: str, custom_url: str, custom_model: str,
        custom_key: str, custom_provider: str,
        output_dir: str, mode: str, folder_template: str,
        system_prompt: str, doc_types_text: str,
        confidence_threshold: float, max_retries: int,
    ) -> tuple[list[list[str]], str, list[dict], dict]:
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
        file_map: dict[str, str] = {}

        for i, fp_str in enumerate(files, 1):
            fp = Path(fp_str)
            file_map[fp.name] = fp_str
            result = process_file(fp, cfg)

            if result.success and result.classification:
                c = result.classification
                conf_str = f"{c.confidence:.0%}"
                # Status-Symbole
                status_parts: list[str] = []
                if result.low_confidence:
                    status_parts.append("⚠️ Konfidenz")
                if result.ocr_quality != "ok":
                    status_parts.append(f"🔍 OCR: {result.ocr_quality_info[:40]}")
                status = " | ".join(status_parts) if status_parts else "✅"

                target_dir = str(result.target.parent) if result.target else "—"
                target_file = result.target.name if result.target else "—"

                rows.append([
                    fp.name, c.doc_type, c.short_info, c.doc_date,
                    conf_str, status, target_dir, target_file,
                ])
                cache.append({
                    "file_path": fp_str,
                    "filename": fp.name,
                    "doc_type": c.doc_type,
                    "short_info": c.short_info,
                    "doc_date": c.doc_date,
                    "confidence": c.confidence,
                    "ocr_quality": result.ocr_quality,
                    "ocr_quality_info": result.ocr_quality_info,
                })
                log_icon = "⚠️" if result.low_confidence else "✓"
                ocr_warn = f" 🔍 OCR-Qualität: {result.ocr_quality}" if result.ocr_quality != "ok" else ""
                log_lines.append(f"[{i}/{len(files)}] {log_icon} {fp.name} → {c.doc_type} ({c.confidence:.0%}){ocr_warn}")
            else:
                rows.append([fp.name, "—", "—", "—", "—", f"❌ {result.error or 'Fehler'}", "—", "—"])
                log_lines.append(f"[{i}/{len(files)}] ✗ {fp.name}: {result.error}")

        return rows, "\n".join(log_lines), cache, file_map

    # ----------------------------------------------------------
    # Execute (nutzt Cache)
    # ----------------------------------------------------------
    def execute(
        files: list[str] | None,
        profile_name: str, custom_url: str, custom_model: str,
        custom_key: str, custom_provider: str,
        output_dir: str, mode: str, folder_template: str,
        system_prompt: str, doc_types_text: str,
        confidence_threshold: float, max_retries: int,
        cached_results: list[dict],
    ) -> str:
        if not files:
            return "⚠️ Keine Dateien hochgeladen."

        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=False,
        )

        log_lines: list[str] = []
        use_cache = cached_results and len(cached_results) > 0

        if use_cache:
            log_lines.append("⚡ Nutze gecachte Analyse-Ergebnisse (kein erneutes OCR/LLM).\n")
            from docsort.organizer import organize

            for i, entry in enumerate(cached_results, 1):
                file_path = Path(entry["file_path"])
                filename = file_path.name

                classification = Classification(
                    doc_type=entry["doc_type"],
                    short_info=entry["short_info"],
                    doc_date=entry["doc_date"],
                    confidence=entry["confidence"],
                )

                try:
                    result = organize(file_path, classification, cfg)
                    if result.success:
                        action = "verschoben" if cfg.mode == "move" else "kopiert"
                        log_lines.append(f"[{i}/{len(cached_results)}] ✓ {filename} → {result.target.name} ({action})")
                    else:
                        log_lines.append(f"[{i}/{len(cached_results)}] ✗ {filename}: {result.error}")
                except Exception as exc:
                    log_lines.append(f"[{i}/{len(cached_results)}] ✗ {filename}: {exc}")

            ok = sum(1 for l in log_lines if "✓" in l)
            total = len(cached_results)
        else:
            log_lines.append("ℹ️ Keine Vorschau vorhanden — vollständige Analyse.\n")
            from docsort.pipeline import process_file

            for i, fp_str in enumerate(files, 1):
                fp = Path(fp_str)
                result = process_file(fp, cfg)
                if result.success:
                    action = "verschoben" if cfg.mode == "move" else "kopiert"
                    log_lines.append(f"[{i}/{len(files)}] ✓ {fp.name} → {result.target} ({action})")
                else:
                    log_lines.append(f"[{i}/{len(files)}] ✗ {fp.name}: {result.error}")

            ok = sum(1 for l in log_lines if "✓" in l)
            total = len(files)

        fail = total - ok
        log_lines.append(f"\n{'='*50}")
        log_lines.append(f"Fertig: {ok}/{total} erfolgreich, {fail} Fehler.")
        return "\n".join(log_lines)

    # ----------------------------------------------------------
    # Seitenpanel: Preview + Edit
    # ----------------------------------------------------------
    def render_preview(file_path_str: str) -> str:
        """Erzeugt HTML-Preview für PDF/Bild."""
        if not file_path_str:
            return _placeholder("Datei in der Tabelle anklicken um Vorschau zu laden.")

        fp = Path(file_path_str)
        if not fp.exists():
            return _placeholder(f"Datei nicht gefunden: {fp.name}", error=True)

        suffix = fp.suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"):
            try:
                data = base64.b64encode(fp.read_bytes()).decode()
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "bmp": "image/bmp", "webp": "image/webp", "tiff": "image/tiff", "tif": "image/tiff",
                        }.get(suffix.lstrip("."), "image/png")
                return f'<div style="text-align:center;max-height:500px;overflow:auto;"><img src="data:{mime};base64,{data}" style="max-width:100%;"/></div>'
            except Exception as exc:
                return _placeholder(f"Bild-Fehler: {exc}", error=True)

        if suffix == ".pdf":
            try:
                data = base64.b64encode(fp.read_bytes()).decode()
                return f'<iframe src="data:application/pdf;base64,{data}" width="100%" height="500px" style="border:1px solid #ccc;border-radius:8px;"></iframe>'
            except Exception as exc:
                return _placeholder(f"PDF-Fehler: {exc}", error=True)

        size_kb = fp.stat().st_size / 1024
        return _placeholder(f"📄 {fp.name}\n{suffix.upper()[1:]} — {size_kb:.1f} KB\nKeine Vorschau verfügbar.")

    def _placeholder(text: str, error: bool = False) -> str:
        color = "#e74c3c" if error else "#888"
        return f'<div style="text-align:center;padding:60px 20px;color:{color};"><pre>{text}</pre></div>'

    def on_table_select(
        evt: gr.SelectData,
        table_data: Any,
        cached_results: list[dict],
        file_map: dict[str, str],
    ) -> tuple[str, str, str, str, str, int, str]:
        """Zeile angeklickt → Seitenpanel füllen."""
        try:
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index

            if hasattr(table_data, "iloc"):
                row = table_data.iloc[row_idx].tolist()
            elif isinstance(table_data, list):
                row = table_data[row_idx]
            else:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""

            filename = str(row[0])
            file_path_str = file_map.get(filename, "")

            # Felder aus Cache laden
            edit_type = str(row[1]) if len(row) > 1 else ""
            edit_info = str(row[2]) if len(row) > 2 else ""
            edit_date = str(row[3]) if len(row) > 3 else ""

            # OCR-Info aus Cache
            ocr_info = ""
            for entry in cached_results:
                if entry.get("filename") == filename:
                    if entry.get("ocr_quality") != "ok":
                        ocr_info = f"🔍 {entry.get('ocr_quality_info', '')}"
                    break

            preview = render_preview(file_path_str)

            return filename, edit_type, edit_info, edit_date, preview, row_idx, ocr_info

        except Exception as exc:
            logger.warning("Tabellen-Auswahl Fehler: %s", exc)
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""

    def apply_edit(
        selected_idx: int,
        edit_type: str,
        edit_info: str,
        edit_date: str,
        cached_results: list[dict],
        table_data: Any,
        folder_template: str,
        output_dir: str,
    ) -> tuple[list[list[str]], list[dict], str]:
        """Übernimmt editierte Werte in Cache und aktualisiert Tabelle."""
        if selected_idx < 0 or not cached_results or selected_idx >= len(cached_results):
            return gr.update(), cached_results, "⚠️ Keine Zeile ausgewählt."

        entry = cached_results[selected_idx]
        entry["doc_type"] = edit_type.strip()
        entry["short_info"] = sanitize_short_info(edit_info.strip()) if edit_info.strip() else entry["short_info"]
        entry["doc_date"] = edit_date.strip() if edit_date.strip() else entry["doc_date"]

        # Tabelle neu aufbauen
        from docsort.organizer import build_target_path
        cfg = Config(output_dir=Path(output_dir), folder_template=folder_template)

        rows: list[list[str]] = []
        for e in cached_results:
            cl = Classification(
                doc_type=e["doc_type"], short_info=e["short_info"],
                doc_date=e["doc_date"], confidence=e["confidence"],
            )
            fp = Path(e["file_path"])
            target = build_target_path(fp, cl, cfg)

            conf_str = f"{e['confidence']:.0%}"
            status_parts: list[str] = []
            if e["confidence"] < 0.7:
                status_parts.append("⚠️ Konfidenz")
            if e.get("ocr_quality", "ok") != "ok":
                status_parts.append(f"🔍 OCR: {e.get('ocr_quality_info', '')[:40]}")
            status = " | ".join(status_parts) if status_parts else "✅"

            rows.append([
                fp.name, e["doc_type"], e["short_info"], e["doc_date"],
                conf_str, status, str(target.parent), target.name,
            ])

        return rows, cached_results, f"✅ Änderung übernommen: {entry['filename']}"

    def do_save_config(
        profile_name: str, custom_url: str, custom_model: str,
        custom_key: str, custom_provider: str,
        output_dir: str, mode: str, folder_template: str,
        system_prompt: str, doc_types_text: str,
        confidence_threshold: float, max_retries: int,
    ) -> str:
        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries,
        )
        path = save_config(cfg)
        return f"✅ Config gespeichert: {path}"

    # ==========================================================
    # UI LAYOUT
    # ==========================================================
    with gr.Blocks(title="DocSort") as app:
        gr.Markdown("# 📄 DocSort\nAutomatische Dokumenten-Klassifizierung und -Sortierung")

        # States
        cached_state = gr.State(value=[])
        file_map_state = gr.State(value={})
        selected_idx_state = gr.State(value=-1)

        with gr.Tabs():
            # ======== Tab 1: Verarbeitung ========
            with gr.Tab("📁 Verarbeitung"):
                file_input = gr.File(
                    label="Dokumente hochladen (Drag & Drop)",
                    file_count="multiple",
                    type="filepath",
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analysieren (Vorschau)", variant="secondary", scale=1)
                    execute_btn = gr.Button("▶️ Ausführen", variant="primary", scale=1)

                log_output = gr.Textbox(label="Log", lines=6, interactive=False)

                gr.Markdown("### 📊 Ergebnis — Klicke eine Zeile an um Details zu sehen und zu bearbeiten")

                with gr.Row():
                    # --- Linke Seite: Tabelle ---
                    with gr.Column(scale=3):
                        result_table = gr.Dataframe(
                            headers=["Datei", "Typ", "Kurzinfo", "Datum", "Konfidenz", "Status", "Zielpfad", "Zieldatei"],
                            datatype=["str"] * 8,
                            col_count=(8, "fixed"),
                            interactive=False,
                            label="Ergebnis (read-only — zum Bearbeiten Zeile anklicken)",
                        )

                    # --- Rechte Seite: Seitenpanel ---
                    with gr.Column(scale=2):
                        gr.Markdown("### 👁️ Dokument & Bearbeitung")

                        # Preview
                        preview_html = gr.HTML(
                            value=_placeholder("Zeile in der Tabelle anklicken um Dokument anzuzeigen."),
                        )

                        # OCR-Qualitäts-Warnung
                        ocr_warning = gr.Textbox(label="OCR-Qualität", visible=True, interactive=False, lines=1)

                        # Edit-Felder
                        gr.Markdown("#### ✏️ Klassifizierung anpassen")
                        edit_filename = gr.Textbox(label="Datei", interactive=False)
                        edit_type = gr.Textbox(label="Dokumenttyp")
                        edit_info = gr.Textbox(label="Kurzinfo")
                        edit_date = gr.Textbox(label="Datum (JJJJ-MM-TT)")
                        apply_btn = gr.Button("✅ Änderung übernehmen", variant="primary")
                        apply_status = gr.Textbox(label="", interactive=False, lines=1)

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
                            value=active.provider, label="Provider",
                        )
                        custom_url_input = gr.Textbox(label="API URL", value=active.base_url)
                        custom_model_input = gr.Textbox(label="Modellname", value=active.model)
                        custom_key_input = gr.Textbox(label="API-Key", value=active.api_key, type="password")

                        gr.Markdown("---")
                        gr.Markdown("### 📂 Ausgabe")
                        output_dir_input = gr.Textbox(label="Ausgabeverzeichnis", value=str(config.output_dir))
                        mode_input = gr.Radio(choices=["copy", "move"], value=config.mode, label="Modus")

                    with gr.Column():
                        gr.Markdown("### 📂 Ordnerstruktur")
                        folder_template_input = gr.Textbox(
                            label="Ordner-Template", value=config.folder_template,
                            info="Variablen: {doc_type}, {year}, {month}, {filename}",
                        )
                        gr.Markdown(
                            "**Beispiele:**\n"
                            "- `{doc_type}/{year}/{filename}` → Rechnung/2026/...\n"
                            "- `{year}/{doc_type}/{filename}` → 2026/Rechnung/...\n"
                            "- `{doc_type}/{year}/{month}/{filename}` → Rechnung/2026/03/...\n"
                        )
                        gr.Markdown("### 📋 Dokumenttypen")
                        doc_types_input = gr.Textbox(
                            label="Dokumenttypen (einer pro Zeile)",
                            value="\n".join(config.doc_types), lines=8,
                        )
                        gr.Markdown("### 🎯 Qualität")
                        confidence_input = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=config.confidence_threshold,
                            label="Confidence-Schwelle",
                        )
                        retry_input = gr.Slider(
                            minimum=0, maximum=5, step=1,
                            value=config.max_retries, label="Max. Retries",
                        )
                        gr.Markdown("### 💾 Config")
                        save_btn = gr.Button("💾 Einstellungen speichern", variant="secondary")
                        save_status = gr.Textbox(label="Status", interactive=False)

            # ======== Tab 3: System-Prompt ========
            with gr.Tab("📝 System-Prompt"):
                gr.Markdown("### System-Prompt anpassen\n`{doc_types}` = Platzhalter für Dokumenttypen.")
                system_prompt_input = gr.Textbox(
                    label="System-Prompt", value=config.system_prompt, lines=25,
                )

        # ==========================================================
        # Inputs
        # ==========================================================
        settings_inputs = [
            profile_dropdown, custom_url_input, custom_model_input,
            custom_key_input, custom_provider_input,
            output_dir_input, mode_input, folder_template_input,
            system_prompt_input, doc_types_input,
            confidence_input, retry_input,
        ]

        # ==========================================================
        # Events
        # ==========================================================
        profile_dropdown.change(
            fn=on_profile_change, inputs=[profile_dropdown],
            outputs=[custom_url_input, custom_model_input, custom_key_input, custom_provider_input],
        )

        analyze_btn.click(
            fn=analyze,
            inputs=[file_input] + settings_inputs,
            outputs=[result_table, log_output, cached_state, file_map_state],
        )

        execute_btn.click(
            fn=execute,
            inputs=[file_input] + settings_inputs + [cached_state],
            outputs=[log_output],
        )

        result_table.select(
            fn=on_table_select,
            inputs=[result_table, cached_state, file_map_state],
            outputs=[edit_filename, edit_type, edit_info, edit_date, preview_html, selected_idx_state, ocr_warning],
        )

        apply_btn.click(
            fn=apply_edit,
            inputs=[
                selected_idx_state, edit_type, edit_info, edit_date,
                cached_state, result_table, folder_template_input, output_dir_input,
            ],
            outputs=[result_table, cached_state, apply_status],
        )

        save_btn.click(
            fn=do_save_config, inputs=settings_inputs, outputs=[save_status],
        )

    return app
