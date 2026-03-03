"""
Validate a JumpDeck analysis JSON (v8).

Usage:
  python -m jumpdeck_analyzer.validate_analysis analysis.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def validate_analysis_dict(a: Dict[str, Any]) -> None:
    assert a.get("version") == 8, f"Expected version 8, got {a.get('version')!r}"

    bg = a.get("beatGraph")
    assert isinstance(bg, dict), "Missing or invalid beatGraph"

    n = bg.get("beatCount")
    assert isinstance(n, int) and n > 0, f"Invalid beatCount: {n!r}"

    beat_start = bg.get("beatStartSec")
    beat_dur = bg.get("beatDurSec")
    beat_pos = bg.get("beatPosInBar")

    assert isinstance(beat_start, list) and len(beat_start) == n, "beatStartSec length mismatch"
    assert isinstance(beat_dur, list) and len(beat_dur) == n, "beatDurSec length mismatch"
    assert isinstance(beat_pos, list) and len(beat_pos) == n, "beatPosInBar length mismatch"

    edges = bg.get("edges")
    assert isinstance(edges, list), "Missing beatGraph.edges"

    for e in edges:
        assert isinstance(e, dict), "Edge is not an object"
        fr = e.get("from")
        to = e.get("to")
        tp = e.get("type")
        assert isinstance(fr, int) and 0 <= fr < n, f"Invalid edge.from: {fr!r}"
        assert isinstance(to, int) and 0 <= to < n, f"Invalid edge.to: {to!r}"
        assert tp in ("default", "jump"), f"Invalid edge.type: {tp!r}"

    # quick sanity checks
    timing = a.get("timing", {})
    if isinstance(timing, dict):
        dur = timing.get("durationSec")
        if dur is not None:
            assert isinstance(dur, (int, float)) and dur > 0, f"Invalid timing.durationSec: {dur!r}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jumpdeck-validate", description="Validate JumpDeck analysis JSON (v8)")
    p.add_argument("analysis_json", help="Path to analysis.json produced by jumpdeck-analyzer")
    args = p.parse_args(argv)

    with open(args.analysis_json, "r", encoding="utf-8") as f:
        a = json.load(f)

    validate_analysis_dict(a)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())