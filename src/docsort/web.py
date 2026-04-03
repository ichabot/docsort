"""Gradio Web-UI für DocSort — Widescreen-optimiert mit Seitenpanel."""

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
        profile_name: str, custom_url: str, custom_model: str,
        custom_key: str, custom_provider: str,
        output_dir: str, mode: str, folder_template: str,
        system_prompt: str, doc_types_text: str,
        confidence_threshold: float, max_retries: int,
        dry_run: bool = False,
    ) -> Config:
        cfg = Config(
            output_dir=Path(output_dir), mode=mode.lower(), dry_run=dry_run,
            gpu=config.gpu, ocr_batch_size=config.ocr_batch_size,
            layout_batch_size=config.layout_batch_size,
            folder_template=folder_template, system_prompt=system_prompt,
            confidence_threshold=confidence_threshold, max_retries=max_retries,
            undo_log=config.undo_log, log_file=config.log_file,
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
    def analyze(
        files, profile_name, custom_url, custom_model,
        custom_key, custom_provider, output_dir, mode,
        folder_template, system_prompt, doc_types_text,
        confidence_threshold, max_retries,
    ):
        if not files:
            return [], "⚠️ Keine Dateien hochgeladen.", [], {}

        from docsort.pipeline import process_file
        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=True,
        )

        rows, log_lines, cache, file_map = [], [], [], {}

        for i, fp_str in enumerate(files, 1):
            fp = Path(fp_str)
            file_map[fp.name] = fp_str
            result = process_file(fp, cfg)

            if result.success and result.classification:
                c = result.classification
                conf_str = f"{c.confidence:.0%}"
                warnings = []
                if result.low_confidence:
                    warnings.append("⚠️ Konfidenz")
                if result.ocr_quality != "ok":
                    warnings.append(f"🔍 OCR")
                status = " ".join(warnings) if warnings else "✅"
                target_dir = str(result.target.parent) if result.target else "—"
                target_file = result.target.name if result.target else "—"

                rows.append([fp.name, c.doc_type, c.short_info, c.doc_date, conf_str, status, target_dir, target_file])
                cache.append({
                    "file_path": fp_str, "filename": fp.name,
                    "doc_type": c.doc_type, "short_info": c.short_info,
                    "doc_date": c.doc_date, "confidence": c.confidence,
                    "ocr_quality": result.ocr_quality, "ocr_quality_info": result.ocr_quality_info,
                })
                icon = "⚠️" if result.low_confidence else "✓"
                ocr_warn = f" 🔍OCR:{result.ocr_quality}" if result.ocr_quality != "ok" else ""
                log_lines.append(f"[{i}/{len(files)}] {icon} {fp.name} → {c.doc_type} ({c.confidence:.0%}){ocr_warn}")
            else:
                rows.append([fp.name, "—", "—", "—", "—", f"❌ Fehler", "—", "—"])
                log_lines.append(f"[{i}/{len(files)}] ✗ {fp.name}: {result.error}")

        return rows, "\n".join(log_lines), cache, file_map

    # ----------------------------------------------------------
    def execute(
        files, profile_name, custom_url, custom_model,
        custom_key, custom_provider, output_dir, mode,
        folder_template, system_prompt, doc_types_text,
        confidence_threshold, max_retries, cached_results,
    ):
        if not files:
            return "⚠️ Keine Dateien hochgeladen."

        cfg = _build_config(
            profile_name, custom_url, custom_model, custom_key,
            custom_provider, output_dir, mode, folder_template, system_prompt,
            doc_types_text, confidence_threshold, max_retries, dry_run=False,
        )

        log_lines = []
        use_cache = cached_results and len(cached_results) > 0

        if use_cache:
            log_lines.append("⚡ Nutze Cache (kein erneutes OCR/LLM).\n")
            from docsort.organizer import organize

            for i, entry in enumerate(cached_results, 1):
                fp = Path(entry["file_path"])
                cl = Classification(
                    doc_type=entry["doc_type"], short_info=entry["short_info"],
                    doc_date=entry["doc_date"], confidence=entry["confidence"],
                )
                try:
                    result = organize(fp, cl, cfg)
                    if result.success:
                        action = "verschoben" if cfg.mode == "move" else "kopiert"
                        log_lines.append(f"[{i}/{len(cached_results)}] ✓ {fp.name} → {result.target.name} ({action})")
                    else:
                        log_lines.append(f"[{i}/{len(cached_results)}] ✗ {fp.name}: {result.error}")
                except Exception as exc:
                    log_lines.append(f"[{i}/{len(cached_results)}] ✗ {fp.name}: {exc}")
            ok = sum(1 for l in log_lines if "✓" in l)
            total = len(cached_results)
        else:
            log_lines.append("ℹ️ Kein Cache — vollständige Analyse.\n")
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
        log_lines.append(f"\n{'='*40}\nFertig: {ok}/{total} erfolgreich, {fail} Fehler.")
        return "\n".join(log_lines)

    # ----------------------------------------------------------
    def render_preview(file_path_str: str) -> str:
        if not file_path_str:
            return _ph("Datei in der Tabelle anklicken")
        fp = Path(file_path_str)
        if not fp.exists():
            return _ph(f"Nicht gefunden: {fp.name}", err=True)
        suffix = fp.suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"):
            try:
                data = base64.b64encode(fp.read_bytes()).decode()
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "bmp": "image/bmp", "webp": "image/webp", "tiff": "image/tiff", "tif": "image/tiff",
                        }.get(suffix.lstrip("."), "image/png")
                return f'<div style="text-align:center;overflow:auto;"><img src="data:{mime};base64,{data}" style="max-width:100%;max-height:70vh;"/></div>'
            except Exception as exc:
                return _ph(f"Fehler: {exc}", err=True)

        if suffix == ".pdf":
            try:
                data = base64.b64encode(fp.read_bytes()).decode()
                return f'<iframe src="data:application/pdf;base64,{data}" width="100%" height="70vh" style="border:1px solid #ddd;border-radius:6px;min-height:400px;"></iframe>'
            except Exception as exc:
                return _ph(f"PDF-Fehler: {exc}", err=True)

        size_kb = fp.stat().st_size / 1024
        return _ph(f"📄 {fp.name}\n{suffix.upper()[1:]} — {size_kb:.1f} KB\nKeine Vorschau verfügbar.")

    def _ph(text: str, err: bool = False) -> str:
        c = "#e74c3c" if err else "#999"
        return f'<div style="display:flex;align-items:center;justify-content:center;height:200px;color:{c};"><pre style="text-align:center;">{text}</pre></div>'

    def on_table_select(evt: gr.SelectData, table_data, cached_results, file_map):
        try:
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if hasattr(table_data, "iloc"):
                row = table_data.iloc[row_idx].tolist()
            elif isinstance(table_data, list):
                row = table_data[row_idx]
            else:
                return (gr.update(),) * 7

            filename = str(row[0])
            fp_str = file_map.get(filename, "")
            edit_type = str(row[1]) if len(row) > 1 else ""
            edit_info = str(row[2]) if len(row) > 2 else ""
            edit_date = str(row[3]) if len(row) > 3 else ""

            ocr_info = ""
            for entry in cached_results:
                if entry.get("filename") == filename and entry.get("ocr_quality") != "ok":
                    ocr_info = f"🔍 {entry.get('ocr_quality_info', '')}"
                    break

            return filename, edit_type, edit_info, edit_date, render_preview(fp_str), row_idx, ocr_info
        except Exception:
            return (gr.update(),) * 7

    def apply_edit(selected_idx, edit_type, edit_info, edit_date, cached_results, table_data, folder_template, output_dir):
        if selected_idx < 0 or not cached_results or selected_idx >= len(cached_results):
            return gr.update(), cached_results, "⚠️ Keine Zeile ausgewählt."

        entry = cached_results[selected_idx]
        entry["doc_type"] = edit_type.strip()
        entry["short_info"] = sanitize_short_info(edit_info.strip()) if edit_info.strip() else entry["short_info"]
        entry["doc_date"] = edit_date.strip() if edit_date.strip() else entry["doc_date"]

        from docsort.organizer import build_target_path
        cfg = Config(output_dir=Path(output_dir), folder_template=folder_template)

        rows = []
        for e in cached_results:
            cl = Classification(doc_type=e["doc_type"], short_info=e["short_info"], doc_date=e["doc_date"], confidence=e["confidence"])
            fp = Path(e["file_path"])
            target = build_target_path(fp, cl, cfg)
            conf_str = f"{e['confidence']:.0%}"
            warnings = []
            if e["confidence"] < 0.7:
                warnings.append("⚠️")
            if e.get("ocr_quality", "ok") != "ok":
                warnings.append("🔍")
            status = " ".join(warnings) if warnings else "✅"
            rows.append([fp.name, e["doc_type"], e["short_info"], e["doc_date"], conf_str, status, str(target.parent), target.name])

        return rows, cached_results, f"✅ {entry['filename']} aktualisiert"

    def do_save_config(profile_name, custom_url, custom_model, custom_key, custom_provider, output_dir, mode, folder_template, system_prompt, doc_types_text, confidence_threshold, max_retries):
        cfg = _build_config(profile_name, custom_url, custom_model, custom_key, custom_provider, output_dir, mode, folder_template, system_prompt, doc_types_text, confidence_threshold, max_retries)
        path = save_config(cfg)
        return f"✅ Config gespeichert: {path}"

    # ==========================================================
    # LAYOUT
    # ==========================================================
    custom_css = """
    .compact-upload { max-height: 80px !important; min-height: 60px !important; }
    .compact-upload .file-preview { display: none !important; }
    """

    with gr.Blocks(title="DocSort", theme=gr.themes.Soft(), css=custom_css) as app:
        # States
        cached_state = gr.State(value=[])
        file_map_state = gr.State(value={})
        selected_idx_state = gr.State(value=-1)

        gr.Markdown("## 📄 DocSort")

        with gr.Tabs():
            # ======== Tab 1: Verarbeitung ========
            with gr.Tab("📁 Verarbeitung"):

                # --- Obere Zeile: Upload + Buttons kompakt ---
                with gr.Row():
                    file_input = gr.File(
                        label="Dateien auswählen",
                        file_count="multiple",
                        type="filepath",
                        scale=3,
                        height=80,
                    )
                    with gr.Column(scale=1, min_width=200):
                        analyze_btn = gr.Button("🔍 Analysieren", variant="secondary", size="lg")
                        execute_btn = gr.Button("▶️ Ausführen", variant="primary", size="lg")

                # --- Hauptbereich: Tabelle links | Panel rechts ---
                with gr.Row():
                    # Tabelle
                    with gr.Column(scale=5):
                        result_table = gr.Dataframe(
                            headers=["Datei", "Typ", "Kurzinfo", "Datum", "Konf.", "Status", "Zielpfad", "Zieldatei"],
                            datatype=["str"] * 8,
                            col_count=(8, "fixed"),
                            interactive=False,
                            label="Ergebnis — Zeile anklicken für Details",
                            height=400,
                        )
                        with gr.Accordion("📋 Log", open=False):
                            log_output = gr.Textbox(lines=8, interactive=False, show_label=False)

                    # Seitenpanel
                    with gr.Column(scale=3, min_width=350):
                        preview_html = gr.HTML(value=_ph("Zeile anklicken für Vorschau"))
                        ocr_warning = gr.Textbox(label="OCR-Qualität", interactive=False, lines=1, visible=True)

                        with gr.Group():
                            gr.Markdown("#### ✏️ Bearbeiten")
                            edit_filename = gr.Textbox(label="Datei", interactive=False, max_lines=1)
                            with gr.Row():
                                edit_type = gr.Textbox(label="Typ", scale=2)
                                edit_date = gr.Textbox(label="Datum", scale=1, placeholder="JJJJ-MM-TT")
                            edit_info = gr.Textbox(label="Kurzinfo")
                            with gr.Row():
                                apply_btn = gr.Button("✅ Übernehmen", variant="primary", scale=2)
                                apply_status = gr.Textbox(show_label=False, interactive=False, scale=3, max_lines=1)

            # ======== Tab 2: Einstellungen ========
            with gr.Tab("⚙️ Einstellungen"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🤖 LLM")
                        profile_dropdown = gr.Dropdown(
                            choices=profile_names + ["(Eigene Einstellungen)"],
                            value=config.active_profile, label="Profil",
                        )
                        active = config.get_active_profile()
                        custom_provider_input = gr.Radio(choices=["openai", "anthropic"], value=active.provider, label="Provider")
                        custom_url_input = gr.Textbox(label="API URL", value=active.base_url)
                        custom_model_input = gr.Textbox(label="Modell", value=active.model)
                        custom_key_input = gr.Textbox(label="API-Key", value=active.api_key, type="password")

                        gr.Markdown("### 📂 Ausgabe")
                        output_dir_input = gr.Textbox(label="Ausgabeverzeichnis", value=str(config.output_dir))
                        mode_input = gr.Radio(choices=["copy", "move"], value=config.mode, label="Modus")

                    with gr.Column():
                        gr.Markdown("### 📂 Ordnerstruktur")
                        folder_template_input = gr.Textbox(
                            label="Template", value=config.folder_template,
                            info="{doc_type}, {year}, {month}, {filename}",
                        )
                        gr.Markdown("Beispiele: `{doc_type}/{year}/{filename}` · `{year}/{doc_type}/{filename}` · `{filename}`")

                        gr.Markdown("### 📋 Dokumenttypen")
                        doc_types_input = gr.Textbox(label="Einer pro Zeile", value="\n".join(config.doc_types), lines=8)

                        gr.Markdown("### 🎯 Qualität")
                        confidence_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=config.confidence_threshold, label="Confidence-Schwelle")
                        retry_input = gr.Slider(minimum=0, maximum=5, step=1, value=config.max_retries, label="Max. Retries")

                        save_btn = gr.Button("💾 Speichern", variant="secondary")
                        save_status = gr.Textbox(label="Status", interactive=False)

            # ======== Tab 3: System-Prompt ========
            with gr.Tab("📝 Prompt"):
                gr.Markdown("System-Prompt anpassen — `{doc_types}` wird durch die Dokumenttypen-Liste ersetzt.")
                system_prompt_input = gr.Textbox(label="System-Prompt", value=config.system_prompt, lines=25)

        # ==========================================================
        # Events
        # ==========================================================
        settings_inputs = [
            profile_dropdown, custom_url_input, custom_model_input,
            custom_key_input, custom_provider_input,
            output_dir_input, mode_input, folder_template_input,
            system_prompt_input, doc_types_input,
            confidence_input, retry_input,
        ]

        profile_dropdown.change(fn=on_profile_change, inputs=[profile_dropdown],
            outputs=[custom_url_input, custom_model_input, custom_key_input, custom_provider_input])

        analyze_btn.click(fn=analyze, inputs=[file_input] + settings_inputs,
            outputs=[result_table, log_output, cached_state, file_map_state])

        execute_btn.click(fn=execute, inputs=[file_input] + settings_inputs + [cached_state],
            outputs=[log_output])

        result_table.select(fn=on_table_select, inputs=[result_table, cached_state, file_map_state],
            outputs=[edit_filename, edit_type, edit_info, edit_date, preview_html, selected_idx_state, ocr_warning])

        apply_btn.click(fn=apply_edit,
            inputs=[selected_idx_state, edit_type, edit_info, edit_date, cached_state, result_table, folder_template_input, output_dir_input],
            outputs=[result_table, cached_state, apply_status])

        save_btn.click(fn=do_save_config, inputs=settings_inputs, outputs=[save_status])

    return app
