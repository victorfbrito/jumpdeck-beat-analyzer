import json
import pytest

from jumpdeck_analyzer.validate_analysis import validate_analysis_dict, main


def _valid_analysis(n: int = 8):
    return {
        "version": 8,
        "beatGraph": {
            "beatCount": n,
            "beatStartSec": [0.0 + i * 0.5 for i in range(n)],
            "beatDurSec": [0.5 for _ in range(n)],
            "beatPosInBar": [i % 4 for i in range(n)],
            "edges": [
                {"from": i, "to": (i + 1) % n, "type": "default"}
                for i in range(n)
            ],
        },
        "timing": {"durationSec": 10.0},
    }


def test_validate_analysis_dict_ok():
    validate_analysis_dict(_valid_analysis(8))


def test_validate_rejects_wrong_version():
    a = _valid_analysis(8)
    a["version"] = 7
    with pytest.raises(AssertionError):
        validate_analysis_dict(a)


def test_validate_rejects_beatcount_mismatch():
    a = _valid_analysis(8)
    a["beatGraph"]["beatStartSec"] = a["beatGraph"]["beatStartSec"][:-1]
    with pytest.raises(AssertionError):
        validate_analysis_dict(a)


def test_validate_rejects_invalid_edge():
    a = _valid_analysis(8)
    a["beatGraph"]["edges"].append({"from": 999, "to": 0, "type": "default"})
    with pytest.raises(AssertionError):
        validate_analysis_dict(a)


def test_validate_cli_main_ok(tmp_path, capsys):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(_valid_analysis(8)), encoding="utf-8")

    rc = main([str(path)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == "OK"