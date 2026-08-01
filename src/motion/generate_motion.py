"""
Main motion generation entry point — this is now the "official" path,
replacing direct use of retrieval.py or multi_clip_retrieval.py.

Behavior, driven by config/default.yaml's `motion` section:
  - If motion.use_real_data is true AND all referenced clip files
    actually exist on disk, use real AIST++ motion-capture data
    (multi_clip_retrieval.py) — beat-aligned, multi-clip, adaptively
    blended.
  - Otherwise, fall back to the procedural placeholder motions
    (retrieval.py) automatically, with a clear printed explanation of
    why. This matters because the real dataset (~800MB+) is gitignored
    and not part of a fresh clone — without this fallback, the project
    would be broken out of the box for anyone (including a future you,
    on a new machine) who hasn't separately downloaded it.
"""

import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent / "audio"))

from render import render_pose_sequence_gif

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent


def load_config():
    with open(PROJECT_ROOT / "config" / "default.yaml") as f:
        return yaml.safe_load(f)


def real_data_available(config) -> bool:
    motion_cfg = config.get("motion", {})
    if not motion_cfg.get("use_real_data", False):
        return False
    clips = motion_cfg.get("clips", {})
    if not clips:
        return False
    return all((PROJECT_ROOT / path).exists() for path in clips.values())


def generate():
    config = load_config()
    song_path = str(PROJECT_ROOT / config["song"]["path"])
    fps = config["render"]["fps"]

    if real_data_available(config):
        print("Real AIST++ clip files found — using real motion-capture data.")
        from multi_clip_retrieval import build_multi_clip_track
        clips = config["motion"]["clips"]
        clip_paths = {tier: str(PROJECT_ROOT / path) for tier, path in clips.items()}
        poses, fps = build_multi_clip_track(song_path, clip_paths)
        xlim, ylim = (-1.2, 1.2), (-0.5, 2.3)
        title_prefix = "Real motion — "
    else:
        motion_cfg = config.get("motion", {})
        if not motion_cfg.get("use_real_data", False):
            reason = "motion.use_real_data is false in config"
        else:
            missing = [p for p in motion_cfg.get("clips", {}).values() if not (PROJECT_ROOT / p).exists()]
            reason = f"real clip file(s) not found locally: {missing}" if missing else "no clips configured"
        print(f"Falling back to procedural placeholder motion ({reason}).")
        print("See the project roadmap / GitHub issue for the real-data setup steps if you want to use it.")

        from retrieval import build_motion_sequence, build_beat_intervals, classify_intervals
        sys.path.append(str(THIS_DIR.parent / "audio"))
        from beat_detection import detect_beats, compute_energy

        y, sr, tempo, beat_times = detect_beats(song_path)
        energy_times, energy = compute_energy(y, sr)
        duration = len(y) / sr
        sequence, cutoffs = build_motion_sequence(duration, beat_times, energy_times, energy, fps=fps)
        poses = [pose for _, _, pose in sequence]
        xlim, ylim = (-0.8, 0.8), (-0.1, 1.9)
        title_prefix = "Procedural motion — "

    output_path = PROJECT_ROOT / "outputs" / "generated_motion.gif"
    render_pose_sequence_gif(poses, fps, output_path, title_prefix=title_prefix, xlim=xlim, ylim=ylim)
    return poses, fps


if __name__ == "__main__":
    generate()