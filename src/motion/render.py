"""
Shared rendering: turn a list of poses into a preview GIF.

Previously duplicated with small variations across retrieval.py,
real_retrieval.py, and multi_clip_retrieval.py — consolidated here so
there's one place to fix bugs or change rendering style, instead of
three that can silently drift apart from each other.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from skeleton import SKELETON_JOINTS, BONES


def render_pose_sequence_gif(poses: list, fps: float, output_path, title_prefix: str = "",
                              xlim=(-1.2, 1.2), ylim=(-0.5, 2.3)):
    """
    Render a list of pose dicts as an animated GIF.

    Args:
        poses: list of pose dicts (joint name -> [x, y, z])
        fps: frames per second for playback timing
        output_path: where to save the .gif (str or Path)
        title_prefix: text shown before the timestamp in each frame's title
        xlim, ylim: plot axis limits — defaults fit real AIST++ data after
            normalize_pose_sequence; procedural motion (tighter range)
            may look better with a smaller box if used standalone.

    Note: this is a flat 2D diagnostic preview (matplotlib) — Z is
    intentionally ignored here even though poses now carry real depth;
    the actual 3D render is the Three.js composite scene.
    """
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
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title_prefix}t={i / fps:.2f}s")
        return ax.artists

    anim = animation.FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved to {output_path}")