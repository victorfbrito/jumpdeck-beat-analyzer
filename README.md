# jumpdeck-beat-analyzer

Generate a JumpDeck-compatible `analysis.json` from an audio file (mp3,
wav, etc).

This tool analyzes a track's rhythmic structure and builds a beat-level
transition graph with musically compatible "jump" edges.

The output JSON matches the schema used by JumpDeck's frontend
(`version: 8`).

------------------------------------------------------------------------

## Features

-   Beat and bar detection
-   Phrase length estimation
-   Per-beat features (chroma, MFCC, loudness, RMS)
-   Structured beat graph with:
    -   default edges (linear progression)
    -   jump edges (similar structural transitions)
-   Chunking (4 bars per chunk)
-   Deterministic graph generation (seeded by jobId)

------------------------------------------------------------------------

## Requirements

-   Python 3.9+
-   For MP3 support: FFmpeg installed (librosa requires a working audio
    backend)

------------------------------------------------------------------------

## Install (from source)

``` bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

If you encounter audio backend errors (e.g., `NoBackendError`), install
optional audio dependencies:

``` bash
pip install -e ".[audio]"
```

You may also need system libraries:

**macOS (Homebrew)**

``` bash
brew install ffmpeg libsndfile
```

**Ubuntu/Debian**

``` bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

------------------------------------------------------------------------

## Usage

Generate an analysis JSON:

``` bash
jumpdeck-analyzer path/to/song.mp3 -o analysis.json
```

With custom job id:

``` bash
jumpdeck-analyzer path/to/song.mp3 --job-id mytrack -o analysis.json
```

If no `--job-id` is provided, one is derived from the filename.

------------------------------------------------------------------------

## Output Structure

The generated JSON includes:

-   `version`
-   `jobId`
-   `timing`
    -   `durationSec`
    -   `tempoBpm`
    -   `timeSignature`
    -   `beats`
    -   `bars`
-   `chunks`
-   `beatGraph`
    -   `beatCount`
    -   `beatStartSec`
    -   `beatDurSec`
    -   `beatPosInBar`
    -   `features`
    -   `edges`
    -   `jumpCandidates`

This file can be consumed directly by the JumpDeck frontend or any
compatible player.

------------------------------------------------------------------------

## Limitations

-   Assumes 4/4 time signature
-   Maximum track length: 20 minutes
-   Designed for structured, beat-driven music
-   Similarity threshold and jump density are conservative by default

------------------------------------------------------------------------

## License

MIT
