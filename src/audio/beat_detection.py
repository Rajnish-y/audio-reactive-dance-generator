"""
Phase 1 — Audio Analysis: beat detection.

Loads a song (path comes from config/default.yaml), detects tempo and
beat timestamps, and saves a plot of the waveform with beat markers.

This is intentionally minimal — just tempo + beat grid. Onset strength
and energy curves are a separate, later step, added once this works.
"""

import yaml
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Read the pipeline config so nothing is hardcoded here."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_beats(song_path: str):
    """
    Load an audio file and detect its tempo and beat timestamps.

    Returns:
        y: audio waveform (numpy array)
        sr: sample rate
        tempo: estimated tempo in BPM
        beat_times: array of timestamps (seconds) where beats occur
    """
    y, sr = librosa.load(song_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return y, sr, tempo, beat_times


def plot_beats(y, sr, tempo, beat_times, output_path: str):
    """Save a waveform plot with a vertical line at every detected beat."""
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)

    for t in beat_times:
        plt.axvline(x=t, color="r", linestyle="--", alpha=0.5)

    tempo_value = float(tempo) if hasattr(tempo, "__len__") is False else float(tempo[0])
    plt.title(f"Waveform with detected beats — estimated tempo: {tempo_value:.1f} BPM")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    config = load_config()
    song_path = config["song"]["path"]

    print(f"Loading: {song_path}")
    y, sr, tempo, beat_times = detect_beats(song_path)

    tempo_value = float(tempo) if hasattr(tempo, "__len__") is False else float(tempo[0])
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")
    print(f"Estimated tempo: {tempo_value:.1f} BPM")
    print(f"Number of beats detected: {len(beat_times)}")
    print(f"First 5 beat timestamps (s): {beat_times[:5]}")

    plot_beats(y, sr, tempo, beat_times, output_path="outputs/beat_plot.png")


if __name__ == "__main__":
    main()