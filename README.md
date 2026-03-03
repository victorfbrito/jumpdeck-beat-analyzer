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
-   FFmpeg (recommended/required for MP3 support depending on your
    platform)
-   libsndfile (recommended; helps audio decoding backends)

### Install system deps

**macOS (Homebrew)**

``` bash
brew install ffmpeg libsndfile
```

**Ubuntu/Debian**

``` bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

**Windows** - Install FFmpeg and ensure `ffmpeg` is on PATH. - If you
have audio decoding issues, try installing a libsndfile package for your
environment.

------------------------------------------------------------------------

## Install

### From source (recommended during development)

``` bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

### With dev tools

``` bash
pip install -e ".[dev]"
```

------------------------------------------------------------------------

## Usage

Basic:

``` bash
jumpdeck-analyzer path/to/song.mp3 -o analysis.json
```

With custom job id:

``` bash
jumpdeck-analyzer path/to/song.mp3 --job-id mytrack -o analysis.json
```

If no `--job-id` is provided, one is derived from the filename.

If you encounter audio decoding errors, install the optional audio backends.

``` bash
pip install -e ".[audio]"
```


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
