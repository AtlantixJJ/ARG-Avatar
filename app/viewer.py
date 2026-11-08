"""Interactive 3D Gaussian splat viewer embedded in the Gradio demo.

The viewer is [Spark](https://sparkjs.dev) (World Labs' three.js Gaussian splatting
renderer, MIT) driven by three.js `OrbitControls` with pan and zoom disabled -- the user
can only change the viewing direction around the avatar. Spark needs nothing beyond
WebGL2, so it runs in desktop Chrome and in iOS Chrome/Safari (which are both WebKit;
ES-module import maps need iOS 16.4+).

The page is handed to `gr.HTML` as an ``<iframe srcdoc=...>``: an iframe is a separate
document, so its module scripts execute (scripts inside `gr.HTML` itself do not) and a
new reconstruction simply replaces the iframe, tearing down the old WebGL context. An
``about:srcdoc`` document inherits the parent page's base URL, so the relative
``gradio_api/file=...`` URL of the PLY resolves against the Gradio server, including
behind ``--share`` or a root path. three.js and Spark are loaded from jsDelivr / the
Spark release CDN, so the *viewer's browser* needs internet access; the server does not.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Final

THREE_VERSION: Final[str] = "0.180.0"
SPARK_VERSION: Final[str] = "2.2.0"

_VIEWER_PAGE: Final[str] = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #f4f5f7; }
  canvas { display: block; width: 100%; height: 100%; touch-action: none; outline: none; }
  #msg {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font: 14px/1.4 system-ui, -apple-system, sans-serif; color: #555; text-align: center;
    padding: 16px; pointer-events: none;
  }
  #hint {
    position: absolute; left: 0; right: 0; bottom: 10px; text-align: center;
    font: 12px system-ui, -apple-system, sans-serif; color: #777; pointer-events: none;
    transition: opacity .6s;
  }
</style>
<script type="importmap">
{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/",
  "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/__SPARK__/spark.module.js"
}}
</script>
</head>
<body>
<div id="msg">Loading 3D avatar&hellip;</div>
<div id="hint" style="opacity:0">Drag to rotate · Scroll outside the viewer</div>
<script type="module">
const msg = document.getElementById("msg");
const hint = document.getElementById("hint");
const fail = (text) => { msg.textContent = text; msg.style.display = "flex"; };

try {
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");

  const renderer = new THREE.WebGLRenderer({ antialias: false });
  if (!renderer.capabilities.isWebGL2) throw new Error("WebGL2 is not available in this browser.");
  // iOS reports devicePixelRatio 3; splatting cost scales with pixels, so cap it.
  const mobile = window.parent.matchMedia("(max-width: 767px), (pointer: coarse) and (max-width: 1023px)").matches;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2));
  renderer.setClearColor(0xf4f5f7, 1);
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  scene.add(new SparkRenderer({ renderer }));

  // Spark parses the URL with `new URL(url)` (no base), so a relative URL throws
  // "Failed to construct 'URL'". Resolve it against the srcdoc's inherited base URL.
  const plyUrl = new URL(__PLY_URL__, document.baseURI).href;
  const avatar = new SplatMesh({ url: plyUrl });
  // The PLY is in the COLMAP/OpenCV frame (y down, subject facing -z); a 180-degree turn
  // about x makes it y-up and facing +z, towards three.js's default camera.
  avatar.quaternion.set(1, 0, 0, 0);
  scene.add(avatar);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.rotateSpeed = 0.8;

  let radius = 1.0;
  const resize = () => {
    const w = Math.max(1, window.innerWidth), h = Math.max(1, window.innerHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Keep the whole avatar in frame whichever of width or height is the tighter fit.
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    const dist = 1.08 * radius / Math.sin(Math.min(vFov, hFov) / 2);
    camera.position.sub(controls.target).setLength(dist).add(controls.target);
    camera.updateProjectionMatrix();
  };
  camera.position.set(0, 0, 3);
  new ResizeObserver(resize).observe(document.body);
  resize();

  await avatar.initialized;
  const box = avatar.getBoundingBox ? avatar.getBoundingBox() : null;
  if (box && !box.isEmpty()) {
    box.applyMatrix4(avatar.matrixWorld.compose(avatar.position, avatar.quaternion, avatar.scale));
    const size = box.getSize(new THREE.Vector3());
    box.getCenter(controls.target);
    // Frame by the standing figure's height/width rather than the full 3D diagonal, so
    // an outstretched arm does not shrink the body.
    radius = 0.5 * Math.max(size.y, size.x, size.z);
    camera.position.set(controls.target.x, controls.target.y, controls.target.z + 3);
    resize();
  }
  msg.style.display = "none";
  hint.style.opacity = "1";
  const hideHint = () => { hint.style.opacity = "0"; };
  renderer.domElement.addEventListener("pointerdown", hideHint, { once: true });

  let contextLost = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) renderer.setAnimationLoop(null);
    else if (!contextLost) renderer.setAnimationLoop(renderFrame);
  });
  renderer.domElement.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    contextLost = true;
    renderer.setAnimationLoop(null);
    fail("3D viewer paused. Reload the page to view the avatar again.");
  });
  function renderFrame() {
    controls.update();
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(renderFrame);
} catch (err) {
  console.error(err);
  fail("Could not start the 3D viewer: " + (err && err.message ? err.message : err));
}
</script>
</body>
</html>
"""


def _placeholder(text: str) -> str:
    return (
        f'<div class="splat-viewer-placeholder">{html.escape(text)}</div>'
    )


PLACEHOLDER_HTML: Final[str] = _placeholder(
    "Reconstruct an avatar to explore it here in 3D (drag to rotate)."
)


def file_url(path: str | Path) -> str:
    """Gradio's file route for ``path``, relative to the app root.

    The file must live under ``allowed_paths``. During a Gradio callback it is
    registered to the requesting browser for the file middleware's access check.
    """
    from app.files import register_output
    register_output(path)
    return "gradio_api/file=" + str(Path(path).resolve())


def splat_viewer_html(ply_path: str | Path | None, message: str = "") -> str:
    """The `gr.HTML` value showing ``ply_path`` in the interactive viewer.

    With no PLY, returns a placeholder carrying ``message`` (or the default hint).
    """
    if not ply_path:
        return _placeholder(message) if message else PLACEHOLDER_HTML
    page = (
        _VIEWER_PAGE
        .replace("__THREE__", THREE_VERSION)
        .replace("__SPARK__", SPARK_VERSION)
        .replace("__PLY_URL__", json.dumps(file_url(ply_path)))
    )
    return (
        '<iframe class="splat-viewer" title="Interactive 3D avatar viewer" '
        f'style="width:100%;border:0;border-radius:8px;display:block" '
        f'srcdoc="{html.escape(page, quote=True)}"></iframe>'
    )
