"""
Phase 4 (start) — Motion Retrieval: link Phase 1 (audio analysis) to
Phase 3 (placeholder motion generator).

For every frame of the output timeline, this:
  1. Classifies the audio's energy at that moment as high or low, and
     picks "bounce" (high energy) or "sway" (low energy) accordingly.
  2. Computes where in its loop cycle the motion should be, syncing
     exactly one full cycle to each beat-to-beat interval — so a bounce
     or sway peak lands on the beat, not somewhere random.

This is the retrieval-selection piece. The fancier scoring/smoothing
algorithm (Phase 4, the "original contribution") builds on top of this
working baseline.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paths resolved relative to this file, not the current working
# directory — so this script runs correctly no matter where it's
# launched from.
THIS_DIR = Path(__file__).resolve().parent          # src/motion
SRC_DIR = THIS_DIR.parent                             # src
PROJECT_ROOT = SRC_DIR.parent                          # project root

sys.path.append(str(SRC_DIR / "audio"))
from beat_detection import detect_beats, compute_energy  # noqa: E402

from skeleton import SKELETON_JOINTS, BONES, generate_pose


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "default.yaml", "r") as f:
        return yaml.safe_load(f)


def build_beat_intervals(beat_times: np.ndarray):
    """
    Build (start, end) time intervals from consecutive beats, plus one
    padding interval before the first beat and after the last, so every
    timestamp in the song falls inside exactly one interval.
    """
    if len(beat_times) < 2:
        return [(0.0, float("inf"))]

    avg_interval = float(np.mean(np.diff(beat_times)))
    intervals = [(beat_times[0] - avg_interval, beat_times[0])]
    for i in range(len(beat_times) - 1):
        intervals.append((beat_times[i], beat_times[i + 1]))
    intervals.append((beat_times[-1], beat_times[-1] + avg_interval))
    return intervals


def classify_intervals(intervals, energy_times: np.ndarray, energy: np.ndarray):
    """
    One motion type per interval, based on that interval's *average*
    energy — not instantaneous per-frame energy. This is what prevents
    the motion from flickering between types multiple times within a
    single beat: the decision is made once per beat, like a real dancer
    picking a move and holding it.
    """
    threshold = float(np.mean(energy))
    types = []
    for start, end in intervals:
        mask = (energy_times >= start) & (energy_times < end)
        avg_energy = energy[mask].mean() if mask.any() else threshold
        types.append("bounce" if avg_energy >= threshold else "sway")
    return types, threshold


def find_interval_index(t: float, intervals) -> int:
    for i, (start, end) in enumerate(intervals):
        if start <= t < end:
            return i
    return len(intervals) - 1  # fallback: clamp to last interval


def build_motion_sequence(duration, beat_times, energy_times, energy, fps=30):
    """Returns (sequence, threshold). sequence is a list of (time, motion_type, pose)."""
    intervals = build_beat_intervals(beat_times)
    interval_types, threshold = classify_intervals(intervals, energy_times, energy)

    n_frames = int(duration * fps)
    sequence = []
    for i in range(n_frames):
        t = i / fps
        idx = find_interval_index(t, intervals)
        start, end = intervals[idx]
        length = (end - start) if end != float("inf") else 1.0
        phase = ((t - start) / length % 1.0) * 2 * np.pi
        motion_type = interval_types[idx]
        sequence.append((t, motion_type, generate_pose(motion_type, phase)))
    return sequence, threshold


def draw_frame(ax, t, motion_type, pose):
    ax.clear()
    xs = [pose[j][0] for j in SKELETON_JOINTS]
    ys = [pose[j][1] for j in SKELETON_JOINTS]
    ax.scatter(xs, ys, c="orange", s=20, zorder=3)
    for a, b in BONES:
        xa, ya = pose[a]
        xb, yb = pose[b]
        ax.plot([xa, xb], [ya, yb], c="royalblue", linewidth=2, zorder=2)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.1, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"t={t:.2f}s  |  {motion_type}", fontsize=10)


def save_sequence_gif(sequence, output_path: Path, fps=30):
    fig, ax = plt.subplots(figsize=(4, 5))

    def update(i):
        t, motion_type, pose = sequence[i]
        draw_frame(ax, t, motion_type, pose)
        return ax.artists

    anim = animation.FuncAnimation(fig, update, frames=len(sequence), interval=1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved motion sequence GIF to {output_path}")


def main():
    config = load_config()
    song_path = PROJECT_ROOT / config["song"]["path"]
    fps = config["render"]["fps"]

    print(f"Loading: {song_path}")
    y, sr, tempo, beat_times = detect_beats(str(song_path))
    energy_times, energy = compute_energy(y, sr)
    duration = len(y) / sr

    print(f"Duration: {duration:.2f}s | Beats: {len(beat_times)} | fps: {fps}")

    sequence, threshold = build_motion_sequence(duration, beat_times, energy_times, energy, fps=fps)

    intervals = build_beat_intervals(beat_times)
    interval_types, _ = classify_intervals(intervals, energy_times, energy)
    print(f"Energy threshold: {threshold:.4f}")
    print("Motion type per beat interval:")
    for (start, end), motion_type in zip(intervals, interval_types):
        print(f"  [{start:.2f}s - {end:.2f}s] -> {motion_type}")

    save_sequence_gif(sequence, PROJECT_ROOT / "outputs" / "motion_sequence.gif", fps=fps)


if __name__ == "__main__":
    main()