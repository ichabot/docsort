# 📄 DocSort

> 🇩🇪 [Deutsche Version / German Version](README_DE.md)

**Automatic classification and sorting of scanned documents using OCR and local LLM.**

DocSort reads scanned documents (PDF, DOCX, XLSX, images, etc.), extracts text via OCR, classifies the document type using an LLM, and automatically sorts files into a consistent folder structure with descriptive filenames.

---

## ✨ Features

- **Universal document formats** — PDF, DOCX, XLSX, PPTX, ODT, JPG, PNG, TIFF and more
- **GPU-accelerated OCR** — Docling with OnnxTR OCR engine on NVIDIA GPUs
- **Multi-LLM support** — LM Studio, Ollama, OpenAI, Anthropic Claude, Google Gemini
- **Sender detection** — Automatically identifies the document sender/issuer
- **Consistent naming** — `YYYY-MM-DD_Description.ext`
- **Flexible folder structure** — Configurable template (`{doc_type}/{year}/{absender}/{filename}` etc.)
- **YAML configuration** — All settings persisted in `docsort.yaml`
- **LLM profiles** — Quickly switch between providers
- **Customizable system prompt** — Tailor classification to your needs
- **Confidence threshold** — Uncertain classifications are flagged
- **Retry logic** — Automatic retry on LLM errors
- **Undo function** — Revert operations via undo log
- **Copy & Move mode** — Originals stay intact (default) or get moved
- **Dry-run** — Preview without changes
- **CLI & Web UI** — Command line (Click) or graphical interface (Gradio)
- **OCR quality check** — Warning on empty or unreadable text
- **Watchfolder** — Monitor a directory and auto-process new files
- **Document preview** — PDF/image preview directly in the Web UI
- **Fault tolerance** — One failed file doesn't stop the rest

---

## ⚠️ Disclaimer

This project was developed with AI assistance ("vibe coding") and uses third-party open-source dependencies that have **not been independently audited**. The software is provided "as is" under the MIT License, without warranty of any kind.

**Please note:**
- **Back up your documents** before using DocSort, especially in `--move` mode (which relocates originals)
- The `--copy` mode (default) keeps your originals untouched — use this until you trust the results
- The `--dry-run` flag lets you preview all changes before anything happens
- OCR and LLM classification can produce errors — always verify results for important documents
- External dependencies (Docling, Gradio, OpenAI SDK, etc.) are maintained by their respective projects — vulnerabilities in those packages are outside our control
- This tool is a personal/hobby project, not a certified document management system

> **Short version:** Test with copies first, check the results, keep backups. Don't blindly trust AI classification for legally or financially critical documents.

---

## 📋 Requirements

| Component | Version | Note |
|---|---|---|
| **Python** | 3.11+ | Recommended: 3.12 |
| **uv** | latest | Package manager ([Install](https://docs.astral.sh/uv/getting-started/installation/)) |
| **LLM server** | — | LM Studio, Ollama (local) or cloud API (OpenAI, Claude, Gemini) |
| **NVIDIA GPU** | optional | Recommended: RTX 4070 Ti SUPER (12 GB VRAM) |
| **CUDA** | 12.8 | For GPU acceleration |

### Setting up an LLM

DocSort supports 5 LLM providers out of the box:

| Provider | Type | Cost | Setup |
|---|---|---|---|
| **LM Studio** | Local | Free | [lmstudio.ai](https://lmstudio.ai/) → Load model → Start server |
| **Ollama** | Local | Free | [ollama.com](https://ollama.com/) → `ollama pull llama3` |
| **OpenAI** | Cloud | ~$0.003/doc | API key from [platform.openai.com](https://platform.openai.com/) |
| **Anthropic Claude** | Cloud | ~$0.003/doc | API key from [console.anthropic.com](https://console.anthropic.com/) |
| **Google Gemini** | Cloud | ~$0.001/doc | API key from [aistudio.google.com](https://aistudio.google.com/) |

> **Recommendation:** LM Studio or Ollama — free, local, no data leaves your machine.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ichabot/docsort.git
cd docsort
```

### 2. Create Python environment (using uv)

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install PyTorch with CUDA (GPU)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **No GPU?** Skip this step — DocSort automatically falls back to CPU.

### 4. Install DocSort

```bash
# Standard installation (OpenAI-compatible LLMs)
uv pip install -e .

# With Anthropic Claude support
uv pip install -e ".[anthropic]"

# With everything (Anthropic + dev tools)
uv pip install -e ".[all-llm,dev]"
```

### 5. OnnxTR OCR engine (optional, recommended)

```bash
uv pip install "docling-ocr-onnxtr[gpu]"
```

### 6. Create config

```bash
docsort init
```

Creates a `docsort.yaml` with all settings. Alternatively, use `docsort.example.yaml` as a template.

---

## 🔧 Configuration

DocSort is configured via a YAML file. Search paths:
1. `./docsort.yaml` (current directory)
2. `~/.config/docsort/docsort.yaml`

### Example config

```yaml
# Active LLM profile
active_profile: lm-studio

# Add/customize profiles
profiles:
  openai:
    api_key: sk-your-key-here
  anthropic:
    api_key: sk-ant-your-key-here
  my-server:
    provider: openai
    base_url: http://192.168.1.100:8080/v1
    model: my-model
    api_key: optional
    description: My own LLM server

# Output
output_dir: ./sorted
mode: copy

# Folder structure template
folder_template: "{doc_type}/{year}/{absender}/{filename}"

# Quality
confidence_threshold: 0.7
max_retries: 2

# Enable undo log
undo_log: ./docsort_undo.csv
```

### LLM Profiles

Built-in profiles:

| Profile | Provider | URL | Default model |
|---|---|---|---|
| `lm-studio` | openai | `localhost:1234/v1` | (LM Studio default) |
| `ollama` | openai | `localhost:11434/v1` | `llama3` |
| `openai` | openai | `api.openai.com/v1` | `gpt-4o-mini` |
| `anthropic` | anthropic | `api.anthropic.com` | `claude-sonnet-4-20250514` |
| `gemini` | openai | `generativelanguage.googleapis.com/...` | `gemini-2.0-flash` |

Show profiles:

```bash
docsort profiles
```

### Folder Structure Template

The template determines the folder structure. Available variables:

| Variable | Description | Example |
|---|---|---|
| `{doc_type}` | Document type | `Rechnung` |
| `{absender}` | Sender/issuer | `Stadtwerke-Muenchen` |
| `{year}` | Year from document date | `2026` |
| `{month}` | Month from document date | `03` |
| `{filename}` | Generated filename | `2026-03-15_Strom-Abrechnung.pdf` |

Examples:

```yaml
# Default: Type → Year → Sender (recommended)
folder_template: "{doc_type}/{year}/{absender}/{filename}"
# → sorted/Rechnung/2026/Stadtwerke-Muenchen/2026-01-15_Strom-Abrechnung.pdf

# Without sender folder
folder_template: "{doc_type}/{year}/{filename}"
# → sorted/Rechnung/2026/2026-01-15_Strom-Abrechnung.pdf

# Year first
folder_template: "{year}/{doc_type}/{absender}/{filename}"
# → sorted/2026/Rechnung/Stadtwerke-Muenchen/2026-01-15_Strom-Abrechnung.pdf

# Flat (no subfolders)
folder_template: "{filename}"
# → sorted/2026-01-15_Strom-Abrechnung.pdf
```

### Customizing the System Prompt

The system prompt controls how the LLM classifies documents. Edit it in the config or Web UI:

```yaml
system_prompt: |
  You are a document classifier for a medical practice.
  Analyze the following text and classify the document.
  
  Allowed document types:
  {doc_types}
  
  Respond with JSON:
  {"doc_type": "...", "absender": "...", "short_info": "...", "doc_date": "YYYY-MM-DD", "confidence": 0.95}
```

> **Note:** `{doc_types}` is automatically replaced with the configured document types list.

### Customizing Document Types

```yaml
doc_types:
  - Invoice
  - Contract
  - Letter
  - Tax Notice
  - Insurance
  - Other
```

---

## 💻 Usage

### CLI — Command Line

#### Process documents

```bash
# Default (uses docsort.yaml)
docsort process ./scans

# Specify output directory
docsort process ./scans -o ./archive

# Dry-run: preview only
docsort process ./scans --dry-run

# With specific LLM profile
docsort process ./scans --profile openai

# Move files instead of copying
docsort process ./scans --move

# All options
docsort process ./scans \
    -o ./archive \
    --move \
    --profile anthropic \
    --batch-size 16 \
    --verbose
```

#### CLI Options

| Option | Description | Default |
|---|---|---|
| `-o, --output` | Output directory | `./sorted` |
| `--copy / --move` | Copy or move files | `--copy` |
| `--dry-run` | Preview only | off |
| `--profile` | LLM profile | from config |
| `--llm-url` | LLM API URL (overrides profile) | — |
| `--model` | Model name (overrides profile) | — |
| `--api-key` | API key (overrides profile) | — |
| `--no-gpu` | Disable GPU | GPU on |
| `--batch-size` | OCR/layout batch size | `32` |
| `--config` | Path to config file | auto |
| `-v, --verbose` | Verbose output | off |

#### Watchfolder — Automatic Processing

```bash
# Monitor directory (checks every 5 seconds)
docsort watch ./scans -o ./sorted

# With shorter interval
docsort watch ./scans --interval 2

# With specific profile and move mode
docsort watch ./scans -o ./archive --profile openai --move
```

New files in the monitored directory are automatically detected, OCR-processed, classified, and sorted. Already processed files are skipped. Stop with `Ctrl+C`.

#### Additional Commands

```bash
# Create config
docsort init

# Show LLM profiles
docsort profiles

# Undo last operations
docsort undo           # all
docsort undo -n 5      # last 5
```

### Web UI — Graphical Interface

```bash
# Default on port 7860
docsort web

# Different port
docsort web --port 8080

# Public link
docsort web --share
```

The Web UI has four tabs:

1. **📁 Processing** — Upload files, analyze & execute
   - Results table (read-only) with status and OCR quality
   - **Side panel**: Click a row → PDF/image preview + edit fields
   - Correct document type, sender, description and date
   - Apply changes → table updates
2. **⚙️ Settings** — LLM profile, output, folder structure, document types, confidence threshold
3. **📝 System Prompt** — Customize the classification prompt
4. **ℹ️ Info** — Version info, pipeline overview, and help

#### Screenshots

| Processing | Settings |
|:---:|:---:|
| ![Processing Tab](docs/screenshots/01_verarbeitung.png) | ![Settings Tab](docs/screenshots/02_einstellungen.png) |

| System Prompt | Info |
|:---:|:---:|
| ![Prompt Tab](docs/screenshots/04_prompt.png) | ![Info Tab](docs/screenshots/05_info.png) |

---

## 📁 Output Schema

### Filenames

```
YYYY-MM-DD_Description.ext
```

Examples:
- `2026-11-21_Strom-Abrechnung-Januar.pdf`
- `2025-06-01_Haftpflicht-Jahresbeitrag.pdf`
- `2026-03-10_Einkommensteuer-2025.pdf`

> Document type and sender are in the folder path and not repeated in the filename.

### Folder Structure (default)

```
sorted/
├── Rechnung/
│   └── 2026/
│       ├── Stadtwerke-Muenchen/
│       │   └── 2026-01-15_Strom-Abrechnung.pdf
│       └── Telekom/
│           └── 2026-02-01_Mobilfunk-Februar.pdf
├── Vertrag/
│   └── 2025/
│       └── Allianz-Versicherung/
│           └── 2025-06-01_Haftpflicht.pdf
└── Bescheid/
    └── 2026/
        └── Finanzamt-Muenchen/
            └── 2026-03-10_Einkommensteuer-2025.pdf
```

### Description Rules

- Hyphens instead of spaces
- Umlauts are replaced: `ä→ae`, `ö→oe`, `ü→ue`, `ß→ss`
- No special characters except hyphens
- Maximum 60 characters
- No date in the description

### Default Document Types

Rechnung · Quittung · Vertrag · Kuendigung · Brief · Bescheid · Steuerbescheid · Kontoauszug · Lohnabrechnung · Versicherung · Mahnung · Angebot · Lieferschein · Gutschrift · Mietvertrag · Arbeitsvertrag · Zeugnis · Urkunde · Formular · Sonstiges

> Custom document types can be defined via config or Web UI.

### Duplicates

On name collisions: `_2`, `_3` etc.

---

## 🏗️ Project Structure

```
docsort/
├── pyproject.toml              # Project config & dependencies
├── docsort.yaml                # Your config (after docsort init)
├── docsort.example.yaml        # Example config template
├── README.md                   # This file (English)
├── README_DE.md                # German documentation
├── LICENSE
├── .gitignore
├── src/
│   └── docsort/
│       ├── __init__.py         # Package init + version
│       ├── config.py           # YAML config, LLM profiles, load/save
│       ├── extractor.py        # Docling text extraction + OCR + quality check
│       ├── classifier.py       # Multi-LLM classification (OpenAI + Anthropic)
│       ├── organizer.py        # Rename + sort files + undo log
│       ├── pipeline.py         # Orchestrates: extract → classify → organize
│       ├── cli.py              # Click CLI (process, web, watch, undo, init, profiles)
│       ├── web.py              # Gradio Web UI with side panel + PDF preview
│       └── watcher.py          # Watchfolder — automatic processing
└── tests/
    ├── __init__.py
    └── test_pipeline.py        # Unit tests (55 tests)
```

### Pipeline Flow

```
Input File
    │
    ▼
[1. Extractor]  ─── Docling + OCR → Text + Metadata
    │
    ▼
[2. Classifier] ─── LLM (selectable) → Type, Sender, Info, Date, Confidence
    │                 ↻ Retry on error
    │                 ⚠ Warning on low confidence
    ▼
[3. Organizer]  ─── Template → Target path → Copy/Move
    │                 📝 Write undo log
    ▼
Sorted file in target folder
```

---

## 🧪 Tests

```bash
# Run tests
pytest

# Verbose output
pytest -v

# With coverage report
pytest --cov=docsort --cov-report=term-missing
```

55 tests covering:
- Description sanitization (umlauts, special characters, length limits)
- JSON extraction (direct, markdown code blocks, embedded)
- Date fallback logic (LLM → file date → today)
- Target path construction with various templates
- Duplicate resolution
- File organization (dry-run, copy, move, undo log)
- File collection (recursive, single file, empty, invalid)
- Config system (profiles, YAML load/save, defaults)
- OCR quality detection (empty, short text, garbage characters)
- Watchfolder (file stability, run-once)

---

## ⚡ GPU Configuration

### Recommended Settings (RTX 4070 Ti SUPER, 12 GB VRAM)

| Parameter | Value |
|---|---|
| `ocr_batch_size` | `16` to `32` |
| `layout_batch_size` | `16` to `32` |
| CUDA Version | 12.8 |

Disable GPU:

```bash
docsort process ./scans --no-gpu
```

DocSort automatically detects whether CUDA is available and falls back to CPU if needed.

---

## 📝 License

MIT License — see [LICENSE](LICENSE)
