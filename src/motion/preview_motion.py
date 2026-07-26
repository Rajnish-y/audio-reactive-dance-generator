"""
Phase 3 — Dance Motion: preview the placeholder motions.

Generates two outputs per motion type:
  1. A static "filmstrip" PNG showing several poses across one loop
     cycle side by side (quick visual sanity check).
  2. An animated GIF looping the full motion (the actual demo output).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from skeleton import SKELETON_JOINTS, BONES, AVAILABLE_MOTIONS, generate_pose


def draw_pose(ax, pose: dict):
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


def save_filmstrip(motion_type: str, output_path: str, n_frames: int = 6):
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2, 3))
    for i, ax in enumerate(axes):
        phase = (i / n_frames) * 2 * np.pi
        pose = generate_pose(motion_type, phase)
        draw_pose(ax, pose)
        ax.set_title(f"{phase:.1f} rad", fontsize=9)

    fig.suptitle(f"Motion: {motion_type}")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved filmstrip to {output_path}")


def save_gif(motion_type: str, output_path: str, n_frames: int = 24):
    fig, ax = plt.subplots(figsize=(4, 5))

    def update(i):
        phase = (i / n_frames) * 2 * np.pi
        pose = generate_pose(motion_type, phase)
        draw_pose(ax, pose)
        return ax.artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer="pillow", fps=20)
    plt.close(fig)
    print(f"Saved animated GIF to {output_path}")


if __name__ == "__main__":
    for motion_type in AVAILABLE_MOTIONS:
        save_filmstrip(motion_type, f"../../outputs/motion_{motion_type}_filmstrip.png")
        save_gif(motion_type, f"../../outputs/motion_{motion_type}.gif")