from __future__ import annotations

import numpy as np
import pytest

import jumpdeck_analyzer.analyzer as analyzer


def test_build_real_analysis_smoke(monkeypatch, tmp_path):
    # --- Fake audio + SR ---
    y = np.zeros(22050 * 8, dtype=np.float32)  # 8 seconds (more room)

    # librosa.load
    monkeypatch.setattr(analyzer.librosa, "load", lambda path, sr=22050, mono=True: (y, sr))

    # onset strength (frames ~ len(y)/hop)
    frames = 1 + len(y) // 512
    onset_env = np.ones(frames, dtype=np.float32)
    monkeypatch.setattr(analyzer.librosa.onset, "onset_strength", lambda y, sr: onset_env)

    # beat_track: return tempo + MANY beat frames (need >= 17 beat times to get 4 bars)
    beat_frames = np.arange(0, 20 * 22, 22, dtype=int)  # 20 beats, spaced
    monkeypatch.setattr(
        analyzer.librosa.beat,
        "beat_track",
        lambda onset_envelope, sr: (120.0, beat_frames),
    )

    # frames_to_time: simple mapping (hop=512)
    monkeypatch.setattr(
        analyzer.librosa,
        "frames_to_time",
        lambda frames, sr: frames.astype(float) * (512.0 / sr),
    )

    # HPSS
    monkeypatch.setattr(analyzer.librosa.effects, "hpss", lambda y: (y, y))

    # Feature fakers: ensure shapes match expectations
    monkeypatch.setattr(
        analyzer.librosa.feature,
        "chroma_stft",
        lambda y, sr, hop_length: np.ones((12, frames), dtype=np.float32),
    )
    monkeypatch.setattr(
        analyzer.librosa.feature,
        "mfcc",
        lambda y, sr, n_mfcc, hop_length: np.ones((12, frames), dtype=np.float32),
    )
    monkeypatch.setattr(
        analyzer.librosa.feature,
        "rms",
        lambda y, hop_length: np.ones((1, frames), dtype=np.float32) * 0.1,
    )
    monkeypatch.setattr(
        analyzer.librosa.feature,
        "spectral_centroid",
        lambda y, sr, hop_length: np.ones((1, frames), dtype=np.float32) * 1000.0,
    )

    # time_to_frames: consistent conversion
    monkeypatch.setattr(
        analyzer.librosa,
        "time_to_frames",
        lambda t, sr, hop_length: int(t * sr / hop_length),
    )

    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"fake")

    res = analyzer.build_real_analysis(job_id="job_test", local_path=str(audio_path))

    # --- Key invariants ---
    assert res["version"] == 8
    assert res["jobId"] == "job_test"
    assert res["timing"]["timeSignature"] == 4
    assert res["beatGraph"]["beatCount"] > 0

    n = res["beatGraph"]["beatCount"]
    assert len(res["beatGraph"]["beatStartSec"]) == n
    assert len(res["beatGraph"]["beatDurSec"]) == n
    assert len(res["beatGraph"]["beatPosInBar"]) == n

    # should have at least one chunk and defaultPath
    assert res["chunking"]["chunkCount"] >= 1
    assert len(res["defaultPath"]) == res["chunking"]["chunkCount"]


def test_analyze_file_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        analyzer.analyze_file("does-not-exist.mp3")