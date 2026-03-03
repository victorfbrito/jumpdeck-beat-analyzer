from __future__ import annotations

import argparse
import json
import sys

from .analyzer import analyze_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jumpdeck-analyzer",
        description="JumpDeck beat/audio analyzer - build analysis JSON from an audio file",
    )
    parser.add_argument("audio_path", help="Path to an audio file (mp3/wav/etc)")
    parser.add_argument("--out", "-o", default="analysis.json", help="Output JSON path (default: analysis.json)")
    parser.add_argument("--job-id", default=None, help="Job id to embed in JSON (default: derived from filename)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    analysis = analyze_file(args.audio_path, job_id=args.job_id)

    with open(args.out, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        else:
            json.dump(analysis, f, ensure_ascii=False)

    bg = analysis.get("beatGraph", {})
    print(f"Wrote: {args.out}")
    print(f"jobId: {analysis.get('jobId')}, beats: {bg.get('beatCount')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))