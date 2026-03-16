"""Tests for src/collector.py"""
import json
import tempfile
from pathlib import Path

import pytest

from src.collector import JSONParser, JSONLParser, TextParser, collect


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ──────────────────────────────────────────────
# JSONParser
# ──────────────────────────────────────────────

class TestJSONParser:
    def test_single_transcript(self, tmp_dir):
        p = write(tmp_dir / "t.json", json.dumps({
            "id": "abc",
            "task": "Fix auth",
            "turns": [{"role": "user", "content": "Fix it"}, {"role": "assistant", "content": "Done"}],
            "outcome": "failure",
        }))
        parser = JSONParser()
        assert parser.can_parse(p)
        results = parser.parse(p)
        assert len(results) == 1
        assert results[0].id == "abc"
        assert results[0].task == "Fix auth"
        assert len(results[0].turns) == 2

    def test_list_of_transcripts(self, tmp_dir):
        p = write(tmp_dir / "batch.json", json.dumps([
            {"id": "1", "task": "A"},
            {"id": "2", "task": "B"},
        ]))
        results = JSONParser().parse(p)
        assert len(results) == 2

    def test_missing_fields_use_defaults(self, tmp_dir):
        p = write(tmp_dir / "minimal.json", json.dumps({"task": "something"}))
        result = JSONParser().parse(p)[0]
        assert result.outcome == "failure"
        assert result.turns == []


# ──────────────────────────────────────────────
# JSONLParser
# ──────────────────────────────────────────────

class TestJSONLParser:
    def test_multiple_lines(self, tmp_dir):
        lines = "\n".join([
            json.dumps({"id": "a", "task": "task1"}),
            json.dumps({"id": "b", "task": "task2"}),
            "",  # blank line should be skipped
        ])
        p = write(tmp_dir / "data.jsonl", lines)
        results = JSONLParser().parse(p)
        assert len(results) == 2

    def test_malformed_line_skipped(self, tmp_dir, capsys):
        lines = json.dumps({"id": "ok"}) + "\nnot-json\n" + json.dumps({"id": "ok2"})
        p = write(tmp_dir / "mixed.jsonl", lines)
        results = JSONLParser().parse(p)
        assert len(results) == 2


# ──────────────────────────────────────────────
# TextParser
# ──────────────────────────────────────────────

class TestTextParser:
    def test_basic_parse(self, tmp_dir):
        content = (
            "TASK: Write a web scraper\n"
            "USER: Please scrape prices\n"
            "ASSISTANT: I'll do it\n"
            "OUTCOME: failure\n"
            "NOTES: Used wrong library\n"
        )
        p = write(tmp_dir / "t.txt", content)
        results = TextParser().parse(p)
        assert len(results) == 1
        t = results[0]
        assert t.task == "Write a web scraper"
        assert t.outcome == "failure"
        assert t.notes == "Used wrong library"
        assert any(turn.role == "user" for turn in t.turns)

    def test_multiple_blocks_separated_by_dashes(self, tmp_dir):
        content = (
            "TASK: Task A\nOUTCOME: failure\n"
            "---\n"
            "TASK: Task B\nOUTCOME: partial\n"
        )
        p = write(tmp_dir / "multi.txt", content)
        results = TextParser().parse(p)
        assert len(results) == 2


# ──────────────────────────────────────────────
# collect()
# ──────────────────────────────────────────────

class TestCollect:
    def test_mixed_formats(self, tmp_dir):
        write(tmp_dir / "a.json", json.dumps({"id": "j1", "task": "JSON task"}))
        write(tmp_dir / "b.jsonl", json.dumps({"id": "jl1"}) + "\n" + json.dumps({"id": "jl2"}))
        write(tmp_dir / "c.txt", "TASK: Text task\nOUTCOME: failure\n")

        results = collect(str(tmp_dir))
        assert len(results) == 4

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            collect("/nonexistent/path/does/not/exist")

    def test_deduplicates_ids(self, tmp_dir):
        # Two files with same id → second should get a suffix
        write(tmp_dir / "1.json", json.dumps({"id": "dup", "task": "A"}))
        write(tmp_dir / "2.json", json.dumps({"id": "dup", "task": "B"}))
        results = collect(str(tmp_dir))
        ids = [t.id for t in results]
        assert len(set(ids)) == 2  # both are unique after dedup
