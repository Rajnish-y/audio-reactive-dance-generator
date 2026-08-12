"""
Character rendering, step 1: render our existing pose data (2D joint
positions, from either procedural or real AIST++ motion) as a simple 3D
stick figure in Three.js — spheres at joints, cylinders for bones —
instead of matplotlib. This does NOT yet use real depth (Z) from AIST++
data (everything is placed on a flat plane, Z=0), and does NOT yet use
a rigged mesh character — both are separate, bigger future steps.

Generates a single self-contained HTML file with the motion data
embedded directly as JSON (no external fetch — keeps it viewable
standalone, e.g. as a chat artifact, without a local server).
"""

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.append(str(THIS_DIR))

from generate_motion import generate as generate_motion, load_config  # noqa: E402
from skeleton import SKELETON_JOINTS, BONES  # noqa: E402

sys.path.append(str(THIS_DIR.parent / "audio"))
from beat_detection import detect_beats  # noqa: E402


def export_character_viewer(output_path):
    poses, fps = generate_motion()
    print(f"Generated {len(poses)} frames.")

    # Re-detect beats for camera timing. Slightly redundant (motion
    # generation already did this internally), but keeps this script
    # simple and decoupled rather than threading beat data through every
    # function signature upstream for one new consumer.
    config = load_config()
    song_path = str(PROJECT_ROOT / config["song"]["path"])
    y, sr, tempo, beat_times = detect_beats(song_path)

    # Convert to a flat list-of-lists per frame (joint order fixed by
    # SKELETON_JOINTS) - more compact JSON than repeating joint names
    # in every single frame.
    frames = [[pose[j] for j in SKELETON_JOINTS] for pose in poses]

    motion_json = json.dumps({
        "joints": SKELETON_JOINTS,
        "bones": BONES,
        "fps": fps,
        "frames": frames,
        "beat_times": [float(t) for t in beat_times],
    })

    html = HTML_TEMPLATE.replace("__MOTION_DATA__", motion_json)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Saved viewer to {output_path} ({len(frames)} frames)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Character Viewer — Phase 6</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #111; }
  canvas { display: block; width: 100vw; height: 100vh; }
  #info {
    position: fixed; top: 12px; left: 12px; z-index: 10;
    font-family: system-ui, sans-serif; color: #fff; font-size: 13px;
    background: rgba(0,0,0,0.5); padding: 8px 12px; border-radius: 6px;
  }
</style>
</head>
<body>
<div id="info">t=0.00s</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const MOTION = __MOTION_DATA__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

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

// Joints: one sphere per joint, reused every frame (positions updated,
// not recreated) for performance.
const jointMeshes = {};
const sphereGeo = new THREE.SphereGeometry(0.035, 12, 12);
const jointMat = new THREE.MeshBasicMaterial({ color: 0xffa500 });
MOTION.joints.forEach(name => {
  const mesh = new THREE.Mesh(sphereGeo, jointMat);
  scene.add(mesh);
  jointMeshes[name] = mesh;
});

// Bones: one cylinder per connection, reused every frame (position,
// rotation, and scale recomputed each frame to connect its two joints).
const boneMat = new THREE.MeshBasicMaterial({ color: 0x4169e1 });
const cylinderGeo = new THREE.CylinderGeometry(0.018, 0.018, 1, 8);
const boneMeshes = MOTION.bones.map(() => {
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
  // Guard against a degenerate (near-zero-length) bone: normalizing a
  // zero vector produces NaN, which would silently corrupt this mesh's
  // orientation and could propagate visually broken state into future
  // frames. Skip the rotation update in that case - the near-zero scale
  // already makes the bone effectively invisible either way.
  if (length > 0.0001) {
    const up = new THREE.Vector3(0, 1, 0);
    mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize());
  }
}

const jointIndex = {};
MOTION.joints.forEach((name, i) => { jointIndex[name] = i; });

// Camera: beat-synced "punch" zoom - dollies subtly closer right after
// each beat, then eases back out exponentially, using the exact same
// beat_times the audio pipeline already computed. beat_times is sorted
// ascending (guaranteed by librosa's beat tracker), so a simple
// backward scan for "most recent beat <= elapsed" is enough - no need
// for a fancier search structure at this scale (a handful to a few
// hundred beats for a full song).
const PUNCH_STRENGTH = 0.35;   // max camera-forward distance at the instant of a beat
const PUNCH_DECAY = 6.0;       // higher = snaps back faster

function timeSinceLastBeat(elapsed) {
  let last = null;
  for (let i = 0; i < MOTION.beat_times.length; i++) {
    if (MOTION.beat_times[i] <= elapsed) {
      last = MOTION.beat_times[i];
    } else {
      break;
    }
  }
  return last === null ? Infinity : elapsed - last;
}

function updateCamera(elapsed) {
  const dt = timeSinceLastBeat(elapsed);
  const punch = dt === Infinity ? 0 : PUNCH_STRENGTH * Math.exp(-PUNCH_DECAY * dt);
  camera.position.z = BASE_CAMERA_Z - punch;
}

function applyFrame(frameIdx) {
  const frame = MOTION.frames[frameIdx];

  MOTION.joints.forEach((name, i) => {
    const [x, y, z] = frame[i];
    jointMeshes[name].position.set(x, y, z);
  });

  MOTION.bones.forEach((pair, i) => {
    const [nameA, nameB] = pair;
    const a = jointMeshes[nameA].position;
    const b = jointMeshes[nameB].position;
    updateBone(boneMeshes[i], a, b);
  });
}

const infoEl = document.getElementById('info');
const startTime = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const elapsed = (performance.now() - startTime) / 1000;
  const frameIdx = Math.floor(elapsed * MOTION.fps) % MOTION.frames.length;
  applyFrame(frameIdx);
  updateCamera(elapsed % (MOTION.frames.length / MOTION.fps));
  infoEl.textContent = `t=${elapsed.toFixed(2)}s  frame ${frameIdx}/${MOTION.frames.length}`;
  renderer.render(scene, camera);
}

applyFrame(0);
animate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    export_character_viewer(PROJECT_ROOT / "outputs" / "character_viewer.html")