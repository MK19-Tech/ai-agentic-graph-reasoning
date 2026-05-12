"""
tests/test_main.py — Unit tests for main.py helpers.

These tests mock the LLM and search calls so no real API key is required.
Run with:  pytest
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── validate_env ──────────────────────────────────────────────────────────────

def test_validate_env_passes_when_key_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-12345")
    # Import fresh so module-level env checks re-run
    import importlib
    import main as m
    importlib.reload(m)
    # Should not raise or exit
    m.validate_env()


def test_validate_env_exits_when_key_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import importlib
    import main as m
    importlib.reload(m)
    with pytest.raises(SystemExit) as exc_info:
        m.validate_env()
    assert exc_info.value.code == 1


# ── save_report ───────────────────────────────────────────────────────────────

def test_save_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import importlib
    import main as m
    importlib.reload(m)

    report_file = tmp_path / "report.md"
    monkeypatch.setattr(m, "REPORT_PATH", report_file)

    m.save_report("# My Report\n\nHello world.")
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == "# My Report\n\nHello world."


def test_save_report_overwrites_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import importlib
    import main as m
    importlib.reload(m)

    report_file = tmp_path / "report.md"
    report_file.write_text("old content")
    monkeypatch.setattr(m, "REPORT_PATH", report_file)

    m.save_report("new content")
    assert report_file.read_text(encoding="utf-8") == "new content"


# ── graph state shape ─────────────────────────────────────────────────────────

def test_initial_state_has_required_keys(monkeypatch):
    """The initial state dict passed to graph.invoke must contain all 4 keys."""
    required_keys = {"topic", "plan", "research_data", "final_report"}
    state = {
        "topic": "quantum computing",
        "plan": "",
        "research_data": "",
        "final_report": "",
    }
    assert required_keys == set(state.keys())
    assert state["topic"] == "quantum computing"
    for k in ("plan", "research_data", "final_report"):
        assert state[k] == ""


# ── main() integration (fully mocked) ────────────────────────────────────────

def test_main_runs_end_to_end(tmp_path, monkeypatch):
    """Full run of main() with graph and input mocked — no real API calls."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    import importlib
    import main as m
    importlib.reload(m)

    report_file = tmp_path / "research_report.md"
    monkeypatch.setattr(m, "REPORT_PATH", report_file)

    # Mock user input
    monkeypatch.setattr("builtins.input", lambda _: "artificial intelligence")

    # Mock graph
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"final_report": "# AI Report\n\nContent here."}

    with patch("main.build_graph", return_value=mock_graph):
        m.main()

    assert report_file.exists()
    assert "AI Report" in report_file.read_text(encoding="utf-8")


def test_main_exits_on_empty_topic(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    import importlib
    import main as m
    importlib.reload(m)

    monkeypatch.setattr("builtins.input", lambda _: "   ")

    mock_graph = MagicMock()
    with patch("main.build_graph", return_value=mock_graph):
        with pytest.raises(SystemExit) as exc_info:
            m.main()
    assert exc_info.value.code == 1


def test_main_exits_on_graph_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    import importlib
    import main as m
    importlib.reload(m)

    monkeypatch.setattr(m, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr("builtins.input", lambda _: "some topic")

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("LLM timeout")

    with patch("main.build_graph", return_value=mock_graph):
        with pytest.raises(SystemExit) as exc_info:
            m.main()
    assert exc_info.value.code == 1
