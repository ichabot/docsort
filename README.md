# 📄 DocSort

**Automatische Klassifizierung und Sortierung eingescannter Dokumente per OCR und lokalem LLM.**

DocSort liest eingescannte Dokumente (PDF, DOCX, XLSX, Bilder etc.) ein, extrahiert den Text per OCR, klassifiziert den Dokumenttyp per LLM und sortiert die Dateien automatisch in eine einheitliche Ordnerstruktur mit sprechendem Dateinamen.

---

## ✨ Features

- **Universelle Dokumentenformate** — PDF, DOCX, XLSX, PPTX, ODT, JPG, PNG, TIFF und mehr
- **GPU-beschleunigte OCR** — Docling mit OnnxTR OCR-Engine auf NVIDIA GPUs
- **Lokales LLM** — Klassifizierung über LM Studio (OpenAI-kompatible API), keine Cloud nötig
- **Einheitliche Benennung** — `JJJJ-MM-TT_Dokumenttyp-Kurzinfo.ext`
- **Automatische Ordnerstruktur** — `Dokumenttyp/Jahr/Dateiname`
- **Copy & Move Modus** — Originale bleiben erhalten (Standard) oder werden verschoben
- **Dry-Run** — Vorschau ohne Änderungen
- **CLI & Web-UI** — Kommandozeile (Click) oder grafische Oberfläche (Gradio)
- **Fehlertoleranz** — Eine fehlerhafte Datei stoppt nicht den Rest

---

## 📋 Voraussetzungen

| Komponente | Version | Hinweis |
|---|---|---|
| **Python** | 3.11+ | Empfohlen: 3.12 |
| **uv** | aktuell | Paketmanager ([Installieren](https://docs.astral.sh/uv/getting-started/installation/)) |
| **LM Studio** | aktuell | Lokal laufend auf `http://localhost:1234` |
| **NVIDIA GPU** | optional | Empfohlen: RTX 4070 Ti SUPER (12 GB VRAM) |
| **CUDA** | 12.8 | Für GPU-Beschleunigung |

### LM Studio einrichten

1. [LM Studio herunterladen](https://lmstudio.ai/) und installieren
2. Ein deutschsprachig-fähiges Modell laden (z.B. Llama 3, Mistral, Qwen)
3. Server starten → läuft auf `http://localhost:1234/v1`
4. DocSort nutzt die OpenAI-kompatible API automatisch

---

## 🚀 Installation

### 1. Repository klonen

```bash
git clone https://github.com/ichabot/docsort.git
cd docsort
```

### 2. Python-Umgebung erstellen (mit uv)

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. PyTorch mit CUDA installieren (GPU)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **Ohne GPU?** Diesen Schritt überspringen — DocSort fällt automatisch auf CPU zurück.

### 4. DocSort installieren

```bash
# Standard-Installation
uv pip install -e .

# Mit Entwicklungs-Tools (pytest)
uv pip install -e ".[dev]"
```

### 5. OnnxTR OCR-Engine (optional, empfohlen)

```bash
uv pip install "docling-ocr-onnxtr[gpu]"
```

Die OnnxTR-Engine liefert bessere OCR-Ergebnisse bei gescannten Dokumenten. Ohne Installation wird Doclings Standard-OCR genutzt.

---

## 🔧 Nutzung

### CLI — Kommandozeile

#### Dokumente verarbeiten

```bash
# Standard: Dateien analysieren und in ./sorted/ kopieren
docsort process ./scans

# Ausgabeverzeichnis angeben
docsort process ./scans -o ./archiv

# Dry-Run: nur Vorschau, nichts ändern
docsort process ./scans --dry-run

# Dateien verschieben statt kopieren
docsort process ./scans --move

# Alle Optionen
docsort process ./scans \
    -o ./archiv \
    --move \
    --llm-url http://localhost:1234/v1 \
    --model "qwen2.5-7b" \
    --batch-size 16 \
    --verbose
```

#### Optionen

| Option | Beschreibung | Standard |
|---|---|---|
| `-o, --output` | Ausgabeverzeichnis | `./sorted` |
| `--copy / --move` | Dateien kopieren oder verschieben | `--copy` |
| `--dry-run` | Nur Vorschau anzeigen | aus |
| `--llm-url` | LLM API URL | `http://localhost:1234/v1` |
| `--model` | LLM Modellname | (LM Studio Standard) |
| `--no-gpu` | GPU deaktivieren | GPU an |
| `--batch-size` | OCR/Layout Batch-Größe | `32` |
| `-v, --verbose` | Ausführliche Ausgabe | aus |

#### Web-UI starten

```bash
# Standard auf Port 7860
docsort web

# Anderer Port
docsort web --port 8080

# Öffentlicher Link (z.B. für Zugriff von anderem Gerät)
docsort web --share
```

### Web-UI — Grafische Oberfläche

1. `docsort web` ausführen
2. Browser öffnet `http://localhost:7860`
3. **Einstellungen** konfigurieren (Ausgabeverzeichnis, LLM URL, Modus)
4. **Dateien hochladen** (Drag & Drop oder Datei-Dialog)
5. **"Analysieren"** klicken → Vorschau-Tabelle mit Klassifizierung
6. **"Ausführen"** klicken → Dateien werden kopiert/verschoben

---

## 📁 Ausgabe-Schema

### Dateinamen

```
JJJJ-MM-TT_Dokumenttyp-Kurzinfo.ext
```

Beispiele:
- `2026-11-21_Rechnung-Sanitaerarbeiten-Firma-Krause.pdf`
- `2025-06-01_Vertrag-Mietvertrag-Hauptstr-5.pdf`
- `2026-03-10_Bescheid-Steuerbescheid-2025.pdf`

### Ordnerstruktur

```
sorted/
├── Rechnung/
│   └── 2026/
│       ├── 2026-01-15_Rechnung-Strom-Januar.pdf
│       └── 2026-11-21_Rechnung-Sanitaerarbeiten-Firma-Krause.pdf
├── Vertrag/
│   └── 2025/
│       └── 2025-06-01_Vertrag-Mietvertrag-Hauptstr-5.pdf
└── Bescheid/
    └── 2026/
        └── 2026-03-10_Bescheid-Steuerbescheid-2025.pdf
```

### Kurzinfo-Regeln

- Bindestriche statt Leerzeichen
- Umlaute werden ersetzt: `ä→ae`, `ö→oe`, `ü→ue`, `ß→ss`
- Keine Sonderzeichen außer Bindestrichen
- Maximal 50 Zeichen
- Kein Datum in der Kurzinfo (steht im Prefix)

### Dokumenttypen

Rechnung · Quittung · Vertrag · Kuendigung · Brief · Bescheid · Steuerbescheid · Kontoauszug · Lohnabrechnung · Versicherung · Mahnung · Angebot · Lieferschein · Gutschrift · Mietvertrag · Arbeitsvertrag · Zeugnis · Urkunde · Formular · Sonstiges

### Duplikate

Bei Namenskollisionen werden Suffixe angehängt:
- `2026-01-15_Rechnung-Strom-Januar.pdf`
- `2026-01-15_Rechnung-Strom-Januar_2.pdf`
- `2026-01-15_Rechnung-Strom-Januar_3.pdf`

---

## 🏗️ Projektstruktur

```
docsort/
├── pyproject.toml              # Projekt-Konfiguration & Dependencies
├── README.md                   # Diese Datei
├── LICENSE                     # MIT-Lizenz
├── .gitignore
├── src/
│   └── docsort/
│       ├── __init__.py         # Package-Init + Version
│       ├── config.py           # Dataclass Config mit allen Einstellungen
│       ├── extractor.py        # Docling-basierte Text-Extraktion + OCR
│       ├── classifier.py       # LLM-Klassifizierung via OpenAI SDK
│       ├── organizer.py        # Dateien umbenennen + kopieren/verschieben
│       ├── pipeline.py         # Orchestriert: extract → classify → organize
│       ├── cli.py              # Click CLI (process + web Kommandos)
│       └── web.py              # Gradio Web-UI
└── tests/
    ├── __init__.py
    └── test_pipeline.py        # Unit-Tests (23 Tests)
```

### Pipeline-Ablauf

```
Eingabe-Datei
    │
    ▼
[1. Extractor]  ─── Docling + OCR → Text + Metadaten
    │
    ▼
[2. Classifier] ─── LLM (LM Studio) → Typ, Kurzinfo, Datum, Konfidenz
    │
    ▼
[3. Organizer]  ─── Zielpfad berechnen → Kopieren/Verschieben
    │
    ▼
Sortierte Datei in Zielordner
```

---

## 🧪 Tests

```bash
# Tests ausführen
pytest

# Mit ausführlicher Ausgabe
pytest -v

# Mit Coverage-Report
pytest --cov=docsort --cov-report=term-missing
```

Die Testsuite umfasst 23 Tests für:
- Kurzinfo-Bereinigung (Umlaute, Sonderzeichen, Längenbegrenzung)
- JSON-Extraktion (direkt, Markdown-Codeblock, eingebettet)
- Datums-Fallback-Logik (LLM-Datum → Datei-Datum → heute)
- Zielpfad-Aufbau und Extension-Handling
- Duplikat-Auflösung
- Datei-Organisation (Dry-Run, Copy, Move)
- Dateisammlung (rekursiv, einzeln, leer, ungültig)

---

## ⚡ GPU-Konfiguration

### Empfohlene Einstellungen (RTX 4070 Ti SUPER, 12 GB VRAM)

| Parameter | Wert |
|---|---|
| `--batch-size` | `16` bis `32` |
| CUDA Version | 12.8 |

### GPU deaktivieren

```bash
docsort process ./scans --no-gpu
```

DocSort erkennt automatisch, ob CUDA verfügbar ist und fällt bei Bedarf auf CPU zurück.

---

## 🔌 LLM-Konfiguration

DocSort nutzt die **OpenAI-kompatible API** und funktioniert mit jedem Server, der dieses Format unterstützt:

| Server | URL |
|---|---|
| **LM Studio** (Standard) | `http://localhost:1234/v1` |
| **Ollama** | `http://localhost:11434/v1` |
| **vLLM** | `http://localhost:8000/v1` |
| **text-generation-webui** | `http://localhost:5000/v1` |

```bash
# Beispiel mit Ollama
docsort process ./scans --llm-url http://localhost:11434/v1 --model llama3

# Beispiel mit vLLM
docsort process ./scans --llm-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B
```

---

## 📝 Lizenz

MIT License — siehe [LICENSE](LICENSE)
