# 🔬 LLM Agent Failure Mode Analyzer

A research-grade toolkit for systematically analyzing **why LLM agents fail** on complex tasks. Collects agent transcripts, embeds them via Ollama, clusters failure patterns with UMAP + HDBSCAN, and generates visual reports.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-local-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📊 Example Output

```
Failure Mode Distribution
──────────────────────────────────────
  40.2%  Ambiguity / Unclear Instructions
  31.8%  Code Convention Violations
  18.1%  Incomplete Solutions
   9.9%  Hallucinated APIs / Libraries
──────────────────────────────────────
  Clusters: 4 | Noise: 3 transcripts
```

Generates:
- Interactive UMAP scatter plot (HTML)
- Cluster summary report (Markdown + JSON)
- Per-cluster failure cards with representative examples

---

## 🗂 Project Structure

```
llm-failure-analyzer/
├── src/
│   ├── collector.py       # Ingest transcripts (JSON, plain text, JSONL)
│   ├── embedder.py        # Embed text via Ollama (nomic-embed-text)
│   ├── clusterer.py       # UMAP dimensionality reduction + HDBSCAN clustering
│   ├── labeler.py         # Auto-label clusters via Ollama LLM
│   ├── reporter.py        # Generate Markdown + JSON reports
│   ├── visualizer.py      # Plotly/matplotlib UMAP plots
│   └── pipeline.py        # End-to-end orchestration
├── data/
│   └── transcripts/       # Drop your .json / .txt / .jsonl files here
├── reports/               # Generated output lands here
├── tests/
│   ├── test_collector.py
│   ├── test_clusterer.py
│   └── fixtures/
├── analyze.py             # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull models

```bash
ollama serve
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2           # cluster labeling
```

### 3. Add your transcripts

Drop files into `data/transcripts/`. Supported formats:

**JSON** (`transcript.json`):
```json
{
  "id": "run-001",
  "task": "Refactor authentication module",
  "turns": [
    {"role": "user", "content": "Rewrite the auth module using JWT"},
    {"role": "assistant", "content": "..."}
  ],
  "outcome": "failure",
  "notes": "Agent hallucinated a non-existent library"
}
```

**Plain text** (`transcript.txt`):
```
TASK: Write a web scraper for product prices
USER: ...
ASSISTANT: ...
OUTCOME: failure
```

**JSONL** — one transcript object per line.

### 4. Run the analyzer

```bash
# Full pipeline
python analyze.py run

# Step by step
python analyze.py collect
python analyze.py embed
python analyze.py cluster
python analyze.py report

# With options
python analyze.py run --model llama3.2 --embed-model nomic-embed-text --min-cluster-size 3
```

---

## 🧠 How It Works

```
Transcripts → Embed (Ollama) → UMAP (2D) → HDBSCAN Clusters → LLM Labels → Report
```

1. **Collect** — Parses transcripts into a normalized schema
2. **Embed** — Sends each transcript to Ollama's embedding endpoint (`/api/embeddings`)
3. **Reduce** — UMAP projects high-dimensional embeddings to 2D for visualization
4. **Cluster** — HDBSCAN finds dense failure groups without needing a fixed `k`
5. **Label** — Each cluster's transcripts are summarized by an Ollama LLM, which assigns a failure category and description
6. **Report** — Generates Markdown report, JSON data, and an interactive HTML plot

---

## ⚙️ Configuration

Edit `config.yaml` or pass CLI flags:

| Option | Default | Description |
|---|---|---|
| `--model` | `llama3.2` | Ollama model for labeling |
| `--embed-model` | `nomic-embed-text` | Ollama embedding model |
| `--min-cluster-size` | `2` | HDBSCAN min cluster size |
| `--umap-neighbors` | `15` | UMAP n_neighbors |
| `--umap-metric` | `cosine` | UMAP distance metric |
| `--output-dir` | `reports/` | Where to save output |

---

## 📦 Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- See `requirements.txt` for Python deps

---

## 🤝 Contributing

PRs welcome. To add a new transcript format, implement a parser in `src/collector.py` following the `TranscriptParser` protocol.

---

## 📄 License

MIT
