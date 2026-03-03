from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _stable_u32(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:4], "big")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _pick_phrase_bars(bar_vecs: np.ndarray, candidates: Tuple[int, ...] = (8, 12, 16)) -> int:
    """
    Pick phrase length (in bars) that best matches repetition.
    Score each candidate by avg cosine similarity: bar i vs bar i+L.
    """
    if bar_vecs.shape[0] < 24:
        return 8

    best_L = 8
    best_score = -1e9

    for L in candidates:
        if bar_vecs.shape[0] <= L + 8:
            continue

        sims: List[float] = []
        for i in range(0, bar_vecs.shape[0] - L):
            sims.append(_cosine(bar_vecs[i], bar_vecs[i + L]))

        score = float(np.mean(sims)) if sims else -1e9
        if score > best_score:
            best_score = score
            best_L = L

    return int(best_L)


def _round_list(x, nd: int = 4) -> List[float]:
    return [float(np.round(v, nd)) for v in x]


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True) + 1e-9
    return (x - mu) / sd


def default_job_id_from_path(path: str) -> str:
    """
    Stable-ish but short: filename + hash prefix of absolute path.
    (Good enough for local usage; callers can pass their own job_id.)
    """
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    h = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    return f"{name}-{h}"


# -----------------------------
# Main analyzer
# -----------------------------
def build_real_analysis(job_id: str, local_path: str) -> Dict[str, Any]:
    # Downsample for speed; ok for beat/bar + matching
    y, sr = librosa.load(local_path, sr=22050, mono=True)
    duration = float(len(y) / sr) if sr else 0.0

    if duration > 60 * 20:
        raise ValueError("Track too long for MVP (max 20 minutes)")

    # --- Tempo + Beats ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(float)

    if beat_times.size < 8:
        tempo = float(tempo) if tempo else 120.0
        beat_times = np.arange(0, max(duration, 1.0), 60.0 / tempo, dtype=float)

    time_signature = 4

    # --- Bars from beats (every 4 beats) ---
    bars: List[Dict[str, float]] = []
    for i in range(0, len(beat_times) - time_signature, time_signature):
        start = float(beat_times[i])
        end = float(beat_times[i + time_signature])
        dur = max(0.001, end - start)
        bars.append({"t": start, "dur": dur, "c": 0.75})

    bars_per_chunk = 4
    chunk_count = len(bars) // bars_per_chunk
    chunk_count = max(chunk_count, 1)

    # Trim bars to full chunks
    bars = bars[: chunk_count * bars_per_chunk]

    # -------------------------------
    # HPSS (better pitch vs drums separation)
    # -------------------------------
    y_h, y_p = librosa.effects.hpss(y)

    # -------------------------------
    # Frame features
    # -------------------------------
    hop_length = 512

    chroma_f = librosa.feature.chroma_stft(y=y_h, sr=sr, hop_length=hop_length)  # (12, frames)
    mfcc_f = librosa.feature.mfcc(y=y_p, sr=sr, n_mfcc=12, hop_length=hop_length)  # (12, frames)
    rms_f = librosa.feature.rms(y=y, hop_length=hop_length)[0]  # (frames,)
    centroid_f = librosa.feature.spectral_centroid(y=y_h, sr=sr, hop_length=hop_length)[0]
    onset_f = onset_env  # aligned with hop_length=512 by default

    # -------------------------------
    # Beat windows (last beat ends at true audio duration)
    # -------------------------------
    bt = np.asarray(beat_times, dtype=float)

    if bt.size >= 2:
        beat_starts = bt
        beat_ends = np.concatenate([bt[1:], np.array([duration], dtype=float)])
    else:
        step = 60.0 / max(float(tempo), 60.0)
        beat_starts = np.arange(0, max(duration, 1.0), step, dtype=float)
        beat_ends = np.concatenate(
            [beat_starts[1:], np.array([min(duration, beat_starts[-1] + step)], dtype=float)]
        )

    n_beats = int(min(len(beat_starts), len(beat_ends)))
    if n_beats <= 0:
        n_beats = 1
        beat_starts = np.array([0.0], dtype=float)
        beat_ends = np.array([min(duration, 0.5)], dtype=float)

    # Position of each beat inside the bar: 0..time_signature-1
    beat_pos = (np.arange(n_beats, dtype=int) % max(1, time_signature)).astype(int)

    # Align loop duration to LAST BEAT END
    duration_loopable = float(beat_ends[n_beats - 1]) if n_beats > 0 else duration

    # -------------------------------
    # Beat feature matrices (pre-normalization)
    # -------------------------------
    chroma_mat: List[np.ndarray] = []
    mfcc_mat: List[np.ndarray] = []
    loud_vec: List[List[float]] = []
    cent_vec: List[List[float]] = []

    beat_rms: List[float] = []
    beat_loud_db: List[float] = []

    for i in range(n_beats):
        t0 = float(beat_starts[i])
        t1 = float(beat_ends[i])

        f0 = int(librosa.time_to_frames(t0, sr=sr, hop_length=hop_length))
        f1 = int(librosa.time_to_frames(t1, sr=sr, hop_length=hop_length))
        f0 = max(0, min(f0, rms_f.shape[0] - 1))
        f1 = max(f0 + 1, min(f1, rms_f.shape[0]))

        chroma_b = np.mean(chroma_f[:, f0:f1], axis=1)
        mfcc_b = np.mean(mfcc_f[:, f0:f1], axis=1)
        cent_b = float(np.mean(centroid_f[f0:f1])) if f1 > f0 else 0.0

        rms_b = float(np.mean(rms_f[f0:f1])) if f1 > f0 else 0.0
        loud_b = float(20.0 * np.log10(rms_b + 1e-9))

        chroma_mat.append(chroma_b)
        mfcc_mat.append(mfcc_b)
        loud_vec.append([loud_b])
        cent_vec.append([cent_b])

        beat_rms.append(rms_b)
        beat_loud_db.append(loud_b)

    chroma_mat_np = np.asarray(chroma_mat, dtype=float)
    mfcc_mat_np = np.asarray(mfcc_mat, dtype=float)
    loud_vec_np = np.asarray(loud_vec, dtype=float)
    cent_vec_np = np.asarray(cent_vec, dtype=float)

    beat_rms_np = np.asarray(beat_rms, dtype=float)
    beat_loud_db_np = np.asarray(beat_loud_db, dtype=float)

    # -------------------------------
    # Proper feature normalization
    # -------------------------------
    chroma_mat_np = chroma_mat_np / (np.sum(chroma_mat_np, axis=1, keepdims=True) + 1e-9)

    mfcc_z = _zscore(mfcc_mat_np)
    loud_z = _zscore(loud_vec_np)
    cent_z = _zscore(cent_vec_np)

    CHROMA_W = 2.0
    MFCC_W = 1.0
    LOUD_W = 0.25
    CENT_W = 0.25

    beat_vecs = np.concatenate(
        [
            CHROMA_W * chroma_mat_np,
            MFCC_W * mfcc_z,
            LOUD_W * loud_z,
            CENT_W * cent_z,
        ],
        axis=1,
    )
    beat_vecs = _normalize_rows(beat_vecs)

    # -------------------------------
    # Bar vectors (phrase detection only)
    # -------------------------------
    n_bars = len(bars)
    bar_vecs: List[np.ndarray] = []

    for b in range(n_bars):
        t0 = float(bars[b]["t"])
        t1 = float(bars[b]["t"] + bars[b]["dur"])
        f0 = int(librosa.time_to_frames(t0, sr=sr, hop_length=hop_length))
        f1 = int(librosa.time_to_frames(t1, sr=sr, hop_length=hop_length))
        f0 = max(0, min(f0, rms_f.shape[0] - 1))
        f1 = max(f0 + 1, min(f1, rms_f.shape[0]))

        rms_b = float(np.mean(rms_f[f0:f1])) if f1 > f0 else 0.0
        cent_b = float(np.mean(centroid_f[f0:f1])) if f1 > f0 else 0.0
        onset_b = float(np.mean(onset_f[f0:f1])) if (f1 > f0 and f1 <= onset_f.shape[0]) else 0.0
        chroma_b = np.mean(chroma_f[:, f0:f1], axis=1) if f1 > f0 else np.zeros((12,), dtype=float)

        v = np.concatenate([chroma_b, np.array([cent_b, onset_b, rms_b], dtype=float)], axis=0)
        bar_vecs.append(v)

    bar_vecs_np = (
        _normalize_rows(np.asarray(bar_vecs, dtype=float)) if bar_vecs else np.zeros((0, 15), dtype=float)
    )
    phrase_bars = _pick_phrase_bars(bar_vecs_np, candidates=(8, 12, 16))

    # -------------------------------
    # Chunks (4 bars) + chunk energy (for UI only)
    # -------------------------------
    chunks: List[Dict[str, Any]] = []
    energies: List[float] = []
    max_energy = 0.0

    for cid in range(chunk_count):
        bar_start = cid * bars_per_chunk
        start_sec = float(bars[bar_start]["t"])
        last_bar = bars[bar_start + bars_per_chunk - 1]
        end_sec = float(last_bar["t"] + last_bar["dur"])

        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(y), int(end_sec * sr))
        seg = y[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0

        energies.append(rms)
        max_energy = max(max_energy, rms)

        chunks.append(
            {
                "id": cid,
                "startSec": start_sec,
                "endSec": end_sec,
                "barStart": bar_start,
                "barCount": bars_per_chunk,
                "c": 0.75,
                "features": {},
            }
        )

    for cid, ch in enumerate(chunks):
        ch["features"]["energy"] = float(energies[cid] / max_energy) if max_energy > 0 else 0.0

    chunk_edges = [
        {"from": i, "to": (i + 1) % chunk_count, "type": "default", "weight": 1.0}
        for i in range(chunk_count)
    ]
    default_path = list(range(chunk_count))

    # -------------------------------
    # BeatGraph (EchoNest-style)
    # -------------------------------
    seed_bg = _stable_u32(f"{job_id}:beatgraph:v4-structure-loop")
    rng_bg = np.random.default_rng(seed_bg)

    beat_jump_candidates: Dict[str, List[int]] = {}
    beat_edges: List[Dict[str, Any]] = []

    TOP_K_BEAT = 3
    MIN_JUMP_SIM = 0.90
    SAMPLE = 200

    beat_dur = np.asarray(beat_ends[:n_beats], dtype=float) - np.asarray(beat_starts[:n_beats], dtype=float)
    beat_dur = np.maximum(0.001, beat_dur)

    bar_index_for_beat = (np.arange(n_beats, dtype=int) // time_signature)
    phrase_bars = int(phrase_bars) if phrase_bars else 8
    phrase_pos_in_phrase_bar = (bar_index_for_beat % max(1, phrase_bars)).astype(int)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n_beats):
        key = (int(beat_pos[i]), int(phrase_pos_in_phrase_bar[i]))
        buckets.setdefault(key, []).append(int(i))

    def _allow_jump_from(i: int) -> bool:
        return int(beat_pos[i]) == 0

    def _pick_similar_beats(i: int) -> List[int]:
        if not _allow_jump_from(i):
            return []

        key = (int(beat_pos[i]), int(phrase_pos_in_phrase_bar[i]))
        pool = buckets.get(key, [])
        if len(pool) <= 1:
            return []

        pool = [j for j in pool if j != i]
        if not pool:
            return []

        k = min(SAMPLE, len(pool))
        cand = rng_bg.choice(pool, size=k, replace=False).tolist()

        src = beat_vecs[i]
        scored: List[Tuple[float, int]] = []
        for j in cand:
            if abs(int(j) - i) <= time_signature:
                continue
            sim = float(_cosine(src, beat_vecs[int(j)]))
            if sim >= MIN_JUMP_SIM:
                scored.append((sim, int(j)))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [int(j) for _, j in scored[:TOP_K_BEAT]]

    def _pick_loop_target(last_i: int) -> int:
        key = (int(beat_pos[last_i]), int(phrase_pos_in_phrase_bar[last_i]))
        pool = buckets.get(key, [])
        if not pool:
            return 0

        early_limit = max(8, int(0.2 * n_beats))
        pool_early = [j for j in pool if j < early_limit and j != last_i]
        pool_use = pool_early if pool_early else [j for j in pool if j != last_i]
        if not pool_use:
            return 0

        k = min(SAMPLE, len(pool_use))
        cand = rng_bg.choice(pool_use, size=k, replace=False).tolist()

        best_j = int(cand[0])
        best_sim = -1e9
        for j in cand:
            j = int(j)
            sim = float(_cosine(beat_vecs[last_i], beat_vecs[j]))
            if sim > best_sim:
                best_sim = sim
                best_j = j

        return int((best_j + 1) % n_beats)

    for i in range(n_beats):
        nxt = _pick_loop_target(i) if i == n_beats - 1 else (i + 1) % n_beats
        beat_edges.append({"from": int(i), "to": int(nxt), "type": "default", "weight": 1.0})

        picks = _pick_similar_beats(i)
        beat_jump_candidates[str(i)] = [int((j + 1) % n_beats) for j in picks]

        if picks:
            sims = [max(0.0, float(_cosine(beat_vecs[i], beat_vecs[j]))) for j in picks]
            total = float(sum(sims)) if sims else 1.0

            for j, sim in zip(picks, sims):
                j_next = int((j + 1) % n_beats)
                weight = max(0.001, sim / total) if total > 0 else 1.0
                beat_edges.append(
                    {
                        "from": int(i),
                        "to": int(j_next),
                        "type": "jump",
                        "weight": float(weight),
                        "sim": float(sim),
                    }
                )

    beatGraph: Dict[str, Any] = {
        "beatCount": int(n_beats),
        "beatStartSec": _round_list(beat_starts[:n_beats], 4),
        "beatDurSec": _round_list(beat_dur[:n_beats], 4),
        "beatPosInBar": [int(x) for x in beat_pos[:n_beats].tolist()],
        "features": {
            "chroma12": _round_list(chroma_mat_np.reshape(-1), 4),
            "mfcc": {"size": 12, "data": _round_list(mfcc_z.reshape(-1), 4)},
            "loudnessDb": _round_list(beat_loud_db_np[:n_beats], 3),
            "rms": _round_list(beat_rms_np[:n_beats], 5),
        },
        "edges": beat_edges,
        "jumpCandidates": beat_jump_candidates,
        "dj": {"jumpProbability": 0.08, "minSimilarity": float(MIN_JUMP_SIM)},
    }

    beats_list = [{"t": float(t), "c": 0.75} for t in beat_starts[:n_beats].tolist()]

    return {
        "version": 8,
        "jobId": job_id,
        "behavior": {"modes": ["default", "dj"]},
        "source": {},
        "timing": {
            "durationSec": float(duration_loopable),
            "sampleRateHz": int(sr),
            "tempoBpm": float(tempo) if tempo else 120.0,
            "tempoConfidence": 0.75,
            "timeSignature": int(time_signature),
            "phraseBars": int(phrase_bars),
            "timeSignatureConfidence": 0.9,
            "beats": beats_list,
            "bars": bars,
        },
        "chunking": {"barsPerChunk": int(bars_per_chunk), "chunkCount": int(chunk_count)},
        "chunks": chunks,
        "edges": chunk_edges,
        "defaultPath": default_path,
        "beatGraph": beatGraph,
    }


def analyze_file(audio_path: str, job_id: str | None = None) -> Dict[str, Any]:
    """
    Public API: analyze an audio file and return the JumpDeck analysis dict.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    jid = job_id or default_job_id_from_path(audio_path)
    return build_real_analysis(job_id=jid, local_path=audio_path)