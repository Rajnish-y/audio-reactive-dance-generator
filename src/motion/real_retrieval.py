"""
Phase 4/5 (real data) — align real AIST++ playback to our song's beat
grid.

Chops ONE continuous real AIST++ sequence into pieces matching each
beat interval's exact (and unequal) duration, advancing continuously
through the source clip frame by frame as it goes — no jumping around
within the source. This proves duration-alignment works with zero
jump-cuts, since nothing is actually being switched between yet, only
sliced from one continuous recording.

Switching BETWEEN different real clips based on energy tier is a
separate, harder problem (introduces genuine jump-cuts needing
crossfade/blending) and is intentionally NOT addressed here — see the
project roadmap / GitHub issue tracking the placeholder-to-real-data
upgrade.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(str(Path(__file__).resolve().parent.parent / "audio"))
from beat_detection import detect_beats  # noqa: E402

from aist_adapter import coco_sequence_to_poses, resample_sequence, normalize_pose_sequence
from skeleton import SKELETON_JOINTS, BONES
from retrieval import build_beat_intervals

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent


def load_config():
    with open(PROJECT_ROOT / "config" / "default.yaml") as f:
        return yaml.safe_load(f)


def extract_snippet(coco_seq, start_frame, duration_seconds, native_fps, target_fps, n_target=None):
    """
    Extract `duration_seconds` worth of native-fps frames starting at
    start_frame, wrapping around to the start of the sequence if it runs
    past the end (treats the source clip as a loop — reasonable since
    it's the only source we have right now). Returns the snippet
    resampled to target_fps, plus the frame index to resume from next
    time, so consecutive calls keep moving forward through the source
    instead of restarting at the same point each time.

    n_target, if given, is passed straight through to resample_sequence
    — see its docstring for why callers assembling many consecutive
    snippets need to control this directly rather than letting each
    snippet round its own frame count independently.
    """
    n_native = coco_seq.shape[0]
    frames_needed = max(1, int(round(duration_seconds * native_fps)))
    indices = [(start_frame + i) % n_native for i in range(frames_needed)]
    snippet = coco_seq[indices]
    resampled = resample_sequence(snippet, native_fps=native_fps, target_fps=target_fps, n_target=n_target)
    next_start_frame = (start_frame + frames_needed) % n_native
    return resampled, next_start_frame


def build_real_motion_track(song_path: str, pkl_path: str, native_fps: float = 60.0):
    config = load_config()
    target_fps = config["render"]["fps"]

    y, sr, tempo, beat_times = detect_beats(song_path)
    duration = len(y) / sr
    intervals = build_beat_intervals(beat_times)
    # Clamp both ends to the actual song duration — the first and last
    # intervals from build_beat_intervals are padding estimates that can
    # extend before 0 or fall short of the real duration.
    intervals = [(max(0.0, s), min(e, duration)) for s, e in intervals]
    intervals = [(s, e) for s, e in intervals if e > s]
    # Clamping alone doesn't guarantee full coverage — the last interval's
    # natural end can still fall short of the true duration (a gap, not
    # an overlap), so force exact coverage of [0, duration] explicitly.
    if intervals:
        intervals[0] = (0.0, intervals[0][1])
        intervals[-1] = (intervals[-1][0], duration)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coco_seq = data["keypoints3d"]

    all_frames = []
    start_frame = 0
    cumulative_target_frames = 0
    for start, end in intervals:
        # Cumulative rounding: decide this interval's frame count based
        # on "how many total frames should exist by the end of this
        # interval" minus "how many we've emitted so far" — not by
        # independently rounding this interval's own duration. The
        # latter causes small per-interval rounding errors to accumulate
        # into a noticeable total drift over many intervals; this way,
        # the total is always exactly correct by construction.
        ideal_cumulative = round(end * target_fps)
        n_target = ideal_cumulative - cumulative_target_frames
        cumulative_target_frames = ideal_cumulative

        snippet, start_frame = extract_snippet(
            coco_seq, start_frame, end - start, native_fps, target_fps, n_target=n_target
        )
        all_frames.append(snippet)

    full_track = np.concatenate(all_frames, axis=0)
    print(f"Total frames assembled: {len(full_track)} at {target_fps}fps "
          f"= {len(full_track) / target_fps:.2f}s (song duration: {duration:.2f}s)")

    poses_raw = coco_sequence_to_poses(full_track)
    poses = normalize_pose_sequence(poses_raw)
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
            xa, ya = pose[a][0], pose[a][1]
            xb, yb = pose[b][0], pose[b][1]
            ax.plot([xa, xb], [ya, yb], c="royalblue", linewidth=2, zorder=2)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 2.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Real motion aligned to beat grid — t={i / fps:.2f}s")
        return ax.artists

    anim = animation.FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python real_retrieval.py <song.mp3> <sequence.pkl>")
        sys.exit(1)

    poses, fps = build_real_motion_track(sys.argv[1], sys.argv[2])
    render_preview(poses, fps, PROJECT_ROOT / "outputs" / "real_beat_aligned.gif")