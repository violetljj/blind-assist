import * as GaussianSplats3D from '/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js';
import * as THREE from 'three';

window.__canary = { ready: false, error: null };

try {
  const config = await fetch('/config.json').then((response) => response.json());
  const viewer = new GaussianSplats3D.Viewer({
    cameraUp: config.camera_up,
    initialCameraPosition: config.camera_position,
    initialCameraLookAt: config.camera_look_at,
    useBuiltInControls: false,
    sharedMemoryForWorkers: true,
    gpuAcceleratedSort: false,
    halfPrecisionCovariancesOnGPU: false,
    sphericalHarmonicsDegree: 0,
    inMemoryCompressionLevel: 0,
    freeIntermediateSplatData: false,
    sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
    renderMode: GaussianSplats3D.RenderMode.Always,
    antialiased: false
  });
  window.__viewer = viewer;
  viewer.camera.fov = config.vertical_fov_degrees;
  viewer.camera.updateProjectionMatrix();
  await viewer.addSplatScene('/scene.ply', {
    splatAlphaRemovalThreshold: config.splat_alpha_removal_threshold,
    showLoadingUI: false,
    progressiveLoad: false
  });
  viewer.start();

  const deadline = performance.now() + config.instance_wait_timeout_ms;
  while (viewer.splatMesh.geometry.instanceCount !== viewer.splatMesh.getSplatCount()) {
    if (performance.now() > deadline) throw new Error('splat instance submission timeout');
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  await new Promise((resolve) => setTimeout(resolve, 500));

  const box = viewer.splatMesh.computeBoundingBox(true);
  const gl = viewer.renderer.getContext();
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  window.__canary = {
    ready: true,
    error: null,
    cross_origin_isolated: window.crossOriginIsolated,
    canvas: { width: viewer.renderer.domElement.width, height: viewer.renderer.domElement.height },
    splat_count: viewer.splatMesh.getSplatCount(),
    splat_render_count: viewer.splatRenderCount,
    instance_count: viewer.splatMesh.geometry.instanceCount,
    bounding_box: { min: box.min.toArray(), max: box.max.toArray() },
    camera: {
      position: viewer.camera.position.toArray(),
      quaternion: viewer.camera.quaternion.toArray(),
      up: viewer.camera.up.toArray(),
      vertical_fov_degrees: viewer.camera.fov,
      near: viewer.camera.near,
      far: viewer.camera.far
    },
    webgl: {
      version: gl.getParameter(gl.VERSION),
      vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)
    }
  };
} catch (error) {
  window.__canary = { ready: false, error: String(error?.stack ?? error) };
}
