# ai-agentic-graph-reasoning

A graph-based agentic AI system that performs multi-step research using **stateful LangGraph workflows**, a **planner → researcher → writer** agent loop, and **DuckDuckGo tool integration** — all powered by Groq's free-tier LLMs.

---

## How it works

```
User topic
    │
    ▼
┌──────────┐     ┌────────────┐     ┌──────────┐
│  Planner │────▶│ Researcher │────▶│  Writer  │
│  (LLM)   │     │ (LLM+DDG)  │     │  (LLM)   │
└──────────┘     └────────────┘     └──────────┘
                                          │
                                          ▼
                                  research_report.md
```

Each node is a LangGraph graph node that reads from and writes to a shared state dictionary (`topic`, `plan`, `research_data`, `final_report`). The graph is built in `graph/builder.py` and the agents live under `agents/`.

---

## Project structure

```
ai-agentic-graph-reasoning/
├── agents/             # Planner, researcher, and writer agent nodes
├── app/                # Optional UI / API layer (Streamlit or FastAPI)
├── config/             # Model and prompt configuration
├── graph/
│   └── builder.py      # LangGraph graph construction
├── utils/              # Shared helpers (logging, formatting, etc.)
├── tests/              # Pytest test suite
│   ├── test_main.py
│   └── test_graph_state.py
├── main.py             # CLI entry point
├── check_env.py        # Quick dependency + env-var health check
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/MK19-Tech/ai-agentic-graph-reasoning.git
cd ai-agentic-graph-reasoning
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Then open .env and paste your Groq API key
# Get a free key at https://console.groq.com
```

### 5. Verify your environment

```bash
python check_env.py
```

### 6. Run

```bash
python main.py
```

You'll be prompted for a research topic. The final report is saved to `research_report.md`.

**Example session:**

```
--- Starting Agentic Graph Reasoning Engine ---

Enter research topic: quantum computing applications in drug discovery

2025-05-12 10:23:01 - INFO - Running graph for topic: 'quantum computing applications in drug discovery'
2025-05-12 10:23:18 - INFO - Report saved → /home/user/ai-agentic-graph-reasoning/research_report.md

✅ Done! Report saved to 'research_report.md'
```

---

## Configuration

All settings live in `.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Your Groq API key |
| `GROQ_MODEL` | ❌ | `llama3-8b-8192` | Which Groq model to use |
| `LOG_LEVEL` | ❌ | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

To also see coverage:

```bash
pytest --cov=. --cov-report=term-missing
```

---

## Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .          # lint
ruff format .         # auto-format
ruff check --fix .    # lint + auto-fix
```

---

## CI

GitHub Actions runs lint and tests on every push and pull request against `main`. See `.github/workflows/ci.yml`.

To enable the test suite in CI, add your `GROQ_API_KEY` as a GitHub repository secret (Settings → Secrets → Actions).

---

## Security note

**Never commit your `.env` file.** It is listed in `.gitignore`. If you accidentally push an API key, revoke it immediately at [console.groq.com](https://console.groq.com) and use `git filter-repo` to clean the history.

---

## License

MIT
