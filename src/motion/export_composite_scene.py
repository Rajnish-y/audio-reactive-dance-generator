"""
Phase 9 — Compositing: combine the dancer, the reactive background, and
the beat-synced camera into ONE scene, driven by a single source of
truth instead of two independent systems.

Key design decision: the standalone reactive_background.html analyzed
audio LIVE (Web Audio AnalyserNode) while the character/camera used
PRECOMPUTED data baked at export time. Combining them naively would
mean two independent timing sources that could drift apart. Instead,
everything here — background energy, character pose, camera — is
precomputed per-frame in Python (we already trust this data; every
piece has been individually tested this session) and driven by the
SAME clock: the actual <audio> element's playback position
(audio.currentTime), not an independent JS timer. This guarantees the
visuals stay locked to what's actually audible, not just to each other.

The song audio itself is embedded as base64 so the whole thing is one
self-contained file — consistent with everything else being tied to
one specific baked song already (the character motion only makes sense
for the exact song it was generated from).
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR.parent / "audio"))

from generate_motion import generate as generate_motion, load_config  # noqa: E402
from skeleton import SKELETON_JOINTS, BONES  # noqa: E402
from beat_detection import detect_beats, compute_energy  # noqa: E402


def export_composite_scene(output_path):
    poses, fps = generate_motion()
    print(f"Generated {len(poses)} frames.")

    config = load_config()
    song_path = PROJECT_ROOT / config["song"]["path"]

    style_name = config.get("character", {}).get("style", "default")
    style_presets = config.get("style_presets", {})
    if style_name not in style_presets:
        print(f"WARNING: style '{style_name}' not found in style_presets, falling back to 'default'.")
        style_name = "default"
    style = style_presets.get(style_name, {
        "joint_color": "0xffa500", "bone_color": "0x4169e1",
        "bg_phase": [0.0, 2.0, 4.0], "bg_tint": [1.0, 1.0, 1.0],
    })
    print(f"Using style preset: '{style_name}'")

    y, sr, tempo, beat_times = detect_beats(str(song_path))
    energy_times, energy = compute_energy(y, sr)

    # Resample energy onto the SAME per-frame timeline as the poses
    # (same technique as aist_adapter's fps resampling: linear
    # interpolation onto target timestamps) so background pulsing and
    # character motion are perfectly frame-aligned, not just
    # approximately close.
    frame_times = np.arange(len(poses)) / fps
    energy_per_frame = np.interp(frame_times, energy_times, energy)
    e_min, e_max = energy_per_frame.min(), energy_per_frame.max()
    energy_norm = ((energy_per_frame - e_min) / (e_max - e_min + 1e-9)).tolist()

    frames = [[pose[j] for j in SKELETON_JOINTS] for pose in poses]

    audio_bytes = song_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_mime = "audio/mpeg" if song_path.suffix.lower() == ".mp3" else "audio/wav"
    print(f"Embedded audio: {len(audio_bytes)/1024:.0f}KB -> {len(audio_b64)/1024:.0f}KB base64")

    scene_json = json.dumps({
        "joints": SKELETON_JOINTS,
        "bones": BONES,
        "fps": fps,
        "frames": frames,
        "beat_times": [float(t) for t in beat_times],
        "energy": [float(e) for e in energy_norm],
        "duration": len(poses) / fps,
        "style": style,
    })

    html = (HTML_TEMPLATE
            .replace("__SCENE_DATA__", scene_json)
            .replace("__AUDIO_MIME__", audio_mime)
            .replace("__AUDIO_B64__", audio_b64))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Saved composite scene to {output_path}")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Composite Scene — Phase 9</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #000; }
  canvas { display: block; width: 100vw; height: 100vh; }
  #info {
    position: fixed; top: 12px; left: 12px; z-index: 10;
    font-family: system-ui, sans-serif; color: #fff; font-size: 13px;
    background: rgba(0,0,0,0.5); padding: 8px 12px; border-radius: 6px;
  }
  #overlay {
    position: fixed; inset: 0; z-index: 20;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.6);
  }
  #playBtn {
    font-size: 20px; padding: 16px 32px; border-radius: 8px; border: none;
    cursor: pointer; background: #ffa500; color: #111; font-weight: bold;
  }
</style>
</head>
<body>
<div id="overlay"><button id="playBtn">&#9654; Play</button></div>
<div id="info">Click play to start</div>
<audio id="audioEl" loop></audio>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SCENE_DATA = __SCENE_DATA__;
const AUDIO_MIME = "__AUDIO_MIME__";
const AUDIO_B64 = "__AUDIO_B64__";

// Decode the embedded base64 audio into a playable Blob URL. Fine at
// this song's size (a couple hundred KB); a much longer song would
// want a chunked conversion instead of this single-pass byte loop.
function base64ToBlob(base64, mime) {
  const byteChars = atob(base64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNumbers[i] = byteChars.charCodeAt(i);
  }
  return new Blob([new Uint8Array(byteNumbers)], { type: mime });
}

const audioEl = document.getElementById('audioEl');
audioEl.src = URL.createObjectURL(base64ToBlob(AUDIO_B64, AUDIO_MIME));

document.getElementById('playBtn').addEventListener('click', () => {
  audioEl.play();
  document.getElementById('overlay').style.display = 'none';
});

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
const BASE_CAMERA_Z = 4.5;
camera.position.set(0, 1.0, BASE_CAMERA_Z);
camera.lookAt(0, 1.0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Reactive background: plasma shader on a large plane behind the
// character, ported from reactive_background.html's raw WebGL1 into a
// Three.js ShaderMaterial so it lives in the same scene/depth buffer.
const bgVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const bgFragmentShader = `
  varying vec2 vUv;
  uniform float u_time;
  uniform float u_energy;
  uniform vec3 u_color_phase;
  uniform vec3 u_color_tint;

  void main() {
    vec2 uv = vUv * 2.0 - 1.0;
    float t = u_time * 0.5;
    float pulse = 0.5 + 0.9 * u_energy;

    float v = sin(uv.x * 10.0 + t)
            + sin(uv.y * 10.0 + t)
            + sin((uv.x + uv.y) * 10.0 + t)
            + sin(sqrt(uv.x * uv.x + uv.y * uv.y) * 10.0 - t);
    v *= pulse;

    vec3 color = vec3(
      0.5 + 0.5 * sin(v + t + u_color_phase.x),
      0.5 + 0.5 * sin(v + t + u_color_phase.y),
      0.5 + 0.5 * sin(v + t + u_color_phase.z)
    ) * u_color_tint;
    color *= (0.5 + u_energy * 0.9);

    gl_FragColor = vec4(color, 1.0);
  }
`;

const bgMaterial = new THREE.ShaderMaterial({
  vertexShader: bgVertexShader,
  fragmentShader: bgFragmentShader,
  uniforms: {
    u_time: { value: 0 },
    u_energy: { value: 0 },
    u_color_phase: { value: new THREE.Vector3(...SCENE_DATA.style.bg_phase) },
    u_color_tint: { value: new THREE.Vector3(...SCENE_DATA.style.bg_tint) },
  },
  depthWrite: false,
});
const bgPlane = new THREE.Mesh(new THREE.PlaneGeometry(40, 24), bgMaterial);
bgPlane.position.set(0, 1.0, -10);
scene.add(bgPlane);

// --- Character: same sphere/cylinder stick figure as character_viewer.html
const jointMeshes = {};
const sphereGeo = new THREE.SphereGeometry(0.035, 12, 12);
const jointMat = new THREE.MeshBasicMaterial({ color: parseInt(SCENE_DATA.style.joint_color, 16) });
SCENE_DATA.joints.forEach(name => {
  const mesh = new THREE.Mesh(sphereGeo, jointMat);
  scene.add(mesh);
  jointMeshes[name] = mesh;
});

const boneMat = new THREE.MeshBasicMaterial({ color: parseInt(SCENE_DATA.style.bone_color, 16) });
const cylinderGeo = new THREE.CylinderGeometry(0.018, 0.018, 1, 8);
const boneMeshes = SCENE_DATA.bones.map(() => {
  const mesh = new THREE.Mesh(cylinderGeo, boneMat);
  scene.add(mesh);
  return mesh;
});

function updateBone(mesh, a, b) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const length = dir.length();
  const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
  mesh.position.copy(mid);
  mesh.scale.set(1, Math.max(length, 0.0001), 1);
  if (length > 0.0001) {
    const up = new THREE.Vector3(0, 1, 0);
    mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize());
  }
}

function applyFrame(frameIdx) {
  const frame = SCENE_DATA.frames[frameIdx];
  SCENE_DATA.joints.forEach((name, i) => {
    const [x, y] = frame[i];
    jointMeshes[name].position.set(x, y, 0);
  });
  SCENE_DATA.bones.forEach((pair, i) => {
    const a = jointMeshes[pair[0]].position;
    const b = jointMeshes[pair[1]].position;
    updateBone(boneMeshes[i], a, b);
  });
}

// --- Camera: beat-synced punch zoom, same behavior as character_viewer.html
const PUNCH_STRENGTH = 0.35;
const PUNCH_DECAY = 6.0;

function timeSinceLastBeat(elapsed) {
  let last = null;
  for (let i = 0; i < SCENE_DATA.beat_times.length; i++) {
    if (SCENE_DATA.beat_times[i] <= elapsed) { last = SCENE_DATA.beat_times[i]; }
    else { break; }
  }
  return last === null ? Infinity : elapsed - last;
}

function updateCamera(loopElapsed) {
  const dt = timeSinceLastBeat(loopElapsed);
  const punch = dt === Infinity ? 0 : PUNCH_STRENGTH * Math.exp(-PUNCH_DECAY * dt);
  camera.position.z = BASE_CAMERA_Z - punch;
}

// --- Render loop - driven by the actual audio playback position
// (audioEl.currentTime), NOT an independent JS timer. This is the key
// design decision: it keeps everything locked to what's actually
// audible, correctly handles pause/seek, and avoids two clocks
// (audio + animation) that could drift apart from each other.
const infoEl = document.getElementById('info');

function animate() {
  requestAnimationFrame(animate);
  const loopElapsed = audioEl.currentTime % SCENE_DATA.duration;
  const frameIdx = Math.min(
    SCENE_DATA.frames.length - 1,
    Math.floor(loopElapsed * SCENE_DATA.fps)
  );

  applyFrame(frameIdx);
  updateCamera(loopElapsed);
  bgMaterial.uniforms.u_time.value = audioEl.currentTime;
  bgMaterial.uniforms.u_energy.value = SCENE_DATA.energy[frameIdx];

  infoEl.textContent = `t=${loopElapsed.toFixed(2)}s  frame ${frameIdx}/${SCENE_DATA.frames.length}  energy=${SCENE_DATA.energy[frameIdx].toFixed(2)}`;
  renderer.render(scene, camera);
}

applyFrame(0);
animate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    export_composite_scene(PROJECT_ROOT / "outputs" / "composite_scene.html")