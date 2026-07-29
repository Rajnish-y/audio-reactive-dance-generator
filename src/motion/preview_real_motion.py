"""
Preview a real AIST++ sequence played back at our pipeline's target fps
(from config), instead of its native 60fps — proving the resampling
keeps real-time duration correct before this gets wired into retrieval.

This does NOT yet do beat-based clip selection/switching — that's a
separate, harder problem (a real clip doesn't loop cleanly like our
procedural sine motions). This step only proves playback timing itself
is correct.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from aist_adapter import coco_sequence_to_poses, resample_sequence, normalize_pose_sequence
from skeleton import SKELETON_JOINTS, BONES

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent


def load_config():
    with open(PROJECT_ROOT / "config" / "default.yaml") as f:
        return yaml.safe_load(f)


def preview_real_sequence(pkl_path: str, seconds: float = 5.0, native_fps: float = 60.0):
    config = load_config()
    target_fps = config["render"]["fps"]

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    kpts = data["keypoints3d"]  # (N, 17, 3), native_fps

    n_native_frames = int(seconds * native_fps)
    kpts = kpts[:n_native_frames]
    actual_seconds = len(kpts) / native_fps

    print(f"Loaded {len(kpts)} native frames ({actual_seconds:.2f}s at {native_fps} fps)")

    resampled = resample_sequence(kpts, native_fps=native_fps, target_fps=target_fps)
    print(f"Resampled to {len(resampled)} frames at {target_fps} fps "
          f"({len(resampled) / target_fps:.2f}s — should match {actual_seconds:.2f}s)")

    poses_raw = coco_sequence_to_poses(resampled)
    poses = normalize_pose_sequence(poses_raw)

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
        ax.set_title(f"Real AIST++ @ {target_fps}fps — t={i/target_fps:.2f}s")
        return ax.artists

    anim = animation.FuncAnimation(fig, update, frames=len(poses), interval=1000 / target_fps)
    output_path = PROJECT_ROOT / "outputs" / "real_motion_preview.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=target_fps)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_real_motion.py <path_to_sequence.pkl>")
        sys.exit(1)
    preview_real_sequence(sys.argv[1])