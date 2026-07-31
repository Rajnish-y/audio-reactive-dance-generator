"""
Phase 4/5 (real data) — multi-clip motion track: switch between DIFFERENT
real AIST++ clips based on energy tier, with an adaptive blend at each
switch point instead of an instant cut.

Builds on real_retrieval.py's single-clip beat alignment, but now:
  - each energy tier maps to a different real clip (e.g. calm clip for
    low energy, energetic clip for high energy)
  - each clip has its own independent "playhead" that keeps advancing
    across its own uses (so re-using the same clip later continues
    where it left off, rather than restarting)
  - when consecutive intervals use DIFFERENT clips, the seam is blended
    with aist_adapter.adaptive_ease_into_transition, which scales the
    blend length to the actual pose gap at that specific switch (a
    fixed-length blend was tested and found insufficient — some gaps
    measured 5-53x the clip's own normal per-frame movement); when they
    use the SAME clip, it's already continuous, no blending needed
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(str(Path(__file__).resolve().parent.parent / "audio"))
from beat_detection import detect_beats, compute_energy  # noqa: E402

from aist_adapter import (
    coco_sequence_to_poses,
    resample_sequence,
    normalize_pose_sequence,
    adaptive_ease_into_transition,
)
from skeleton import SKELETON_JOINTS, BONES
from retrieval import build_beat_intervals, classify_intervals

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent


def load_config():
    with open(PROJECT_ROOT / "config" / "default.yaml") as f:
        return yaml.safe_load(f)


def load_clip(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["keypoints3d"]


def extract_raw_snippet(coco_seq, start_frame, n_native_frames):
    """Grab n_native_frames starting at start_frame, wrapping if needed."""
    n_native = coco_seq.shape[0]
    indices = [(start_frame + i) % n_native for i in range(n_native_frames)]
    return coco_seq[indices], (start_frame + n_native_frames) % n_native


def build_multi_clip_track(song_path: str, tier_clip_paths: dict, native_fps: float = 60.0):
    """
    tier_clip_paths: dict mapping tier ("low"/"mid"/"high") -> pkl path.
    A tier with no entry falls back to the nearest available tier
    (mid -> low, high -> mid -> low) rather than crashing, so this still
    works with only 2 clips instead of requiring exactly 3.
    """
    config = load_config()
    target_fps = config["render"]["fps"]

    y, sr, tempo, beat_times = detect_beats(song_path)
    energy_times, energy = compute_energy(y, sr)
    duration = len(y) / sr

    intervals = build_beat_intervals(beat_times)
    tiers, intensities, cutoffs = classify_intervals(intervals, energy_times, energy)

    intervals = [(max(0.0, s), min(e, duration)) for s, e in intervals]
    if intervals:
        intervals[0] = (0.0, intervals[0][1])
        intervals[-1] = (intervals[-1][0], duration)

    # Load each distinct clip once.
    clips = {tier: load_clip(path) for tier, path in tier_clip_paths.items()}
    playheads = {tier: 0 for tier in clips}

    def resolve_tier(tier):
        # Fall back to an available tier if this one has no clip assigned.
        order = {"high": ["high", "mid", "low"], "mid": ["mid", "low", "high"], "low": ["low", "mid", "high"]}
        for candidate in order[tier]:
            if candidate in clips:
                return candidate
        raise ValueError("No clips available at all")

    segments = []  # list of (tier_used, pose_list)
    cumulative_target_frames = 0
    for (start, end), tier in zip(intervals, tiers):
        resolved = resolve_tier(tier)
        coco_seq = clips[resolved]

        ideal_cumulative = round(end * target_fps)
        n_target = max(1, ideal_cumulative - cumulative_target_frames)
        cumulative_target_frames = ideal_cumulative

        interval_duration = end - start
        n_native = max(1, int(round(interval_duration * native_fps)))
        raw_snippet, playheads[resolved] = extract_raw_snippet(coco_seq, playheads[resolved], n_native)
        resampled = resample_sequence(raw_snippet, native_fps, target_fps, n_target=n_target)
        poses = coco_sequence_to_poses(resampled)
        segments.append((resolved, poses))

    # Stitch segments together, crossfading only at points where the
    # clip actually changes.
    merged = segments[0][1]
    for i in range(1, len(segments)):
        prev_tier, _ = segments[i - 1]
        cur_tier, cur_poses = segments[i]
        if cur_tier != prev_tier:
            merged = adaptive_ease_into_transition(merged, cur_poses)
        else:
            merged = merged + cur_poses

    poses = normalize_pose_sequence(merged)
    print(f"Total frames: {len(poses)} at {target_fps}fps = {len(poses)/target_fps:.2f}s "
          f"(song duration: {duration:.2f}s)")
    print("Tier sequence:", [t for t, _ in segments])
    return poses, target_fps


def render_preview(poses, fps, output_path):
    fig, ax = plt.subplots(figsize=(4, 5))

    def update(i):
        ax.clear()
        pose = poses[i]
        xs = [pose[j][0] for j in SKELETON_JOINTS]
        ys = [pose[j][1] for j in SKELETON_JOINTS]
        ax.scatter(xs, ys, c="orange", s=20, zorder=3)
        for a, b in BONES:
            xa, ya = pose[a]
            xb, yb = pose[b]
            ax.plot([xa, xb], [ya, yb], c="royalblue", linewidth=2, zorder=2)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 2.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Multi-clip real motion — t={i / fps:.2f}s")
        return ax.artists

    anim = animation.FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python multi_clip_retrieval.py <song.mp3> <low_energy_clip.pkl> <high_energy_clip.pkl>")
        sys.exit(1)

    song_path, low_clip, high_clip = sys.argv[1], sys.argv[2], sys.argv[3]
    tier_clips = {"low": low_clip, "mid": low_clip, "high": high_clip}

    poses, fps = build_multi_clip_track(song_path, tier_clips)
    render_preview(poses, fps, PROJECT_ROOT / "outputs" / "multi_clip_real.gif")