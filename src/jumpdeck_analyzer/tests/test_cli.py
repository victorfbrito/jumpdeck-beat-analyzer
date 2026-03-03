import json
from pathlib import Path

import pytest

from jumpdeck_analyzer import cli


def test_cli_writes_json_and_prints_summary(tmp_path, monkeypatch, capsys):
    out_path = tmp_path / "analysis.json"

    fake_analysis = {
        "version": 8,
        "jobId": "job_test",
        "beatGraph": {"beatCount": 12},
    }

    # Stub analyzer
    monkeypatch.setattr(cli, "analyze_file", lambda audio_path, job_id=None: fake_analysis)

    rc = cli.main(["song.mp3", "--out", str(out_path), "--job-id", "job_test"])
    assert rc == 0

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["version"] == 8
    assert data["jobId"] == "job_test"

    out = capsys.readouterr().out
    assert "Wrote:" in out
    assert "jobId: job_test" in out
    assert "beats: 12" in out


def test_cli_pretty_prints_json(tmp_path, monkeypatch):
    out_path = tmp_path / "analysis_pretty.json"
    fake_analysis = {"version": 8, "jobId": "x", "beatGraph": {"beatCount": 1}}

    monkeypatch.setattr(cli, "analyze_file", lambda audio_path, job_id=None: fake_analysis)

    rc = cli.main(["song.mp3", "--out", str(out_path), "--pretty"])
    assert rc == 0

    txt = out_path.read_text(encoding="utf-8")
    # pretty JSON has newlines + indentation
    assert "\n" in txt
    assert "  " in txt