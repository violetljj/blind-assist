import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { PNG } from 'pngjs';

const SCHEMA = 'blindassist_abotn_webgl_render_canary_v0';
const DATASET_REVISION = 'fbb62cc3382d8ff84f7fe3b6a3e7d48e4c21e974';
const RENDERER_PACKAGE = '@mkkellogg/gaussian-splats-3d';
const RENDERER_VERSION = '0.4.7';
const root = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i += 2) values[argv[i].replace(/^--/, '')] = argv[i + 1];
  for (const required of ['ply', 'annotation', 'output-dir']) {
    if (!values[required]) throw new Error(`missing --${required}`);
  }
  values.chrome ??= 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  return values;
}

function sha256File(file) {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const buffer = Buffer.alloc(4 * 1024 * 1024);
  try {
    let count;
    while ((count = fs.readSync(fd, buffer)) > 0) hash.update(buffer.subarray(0, count));
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
}

function inspectPly(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const buffer = Buffer.alloc(16384);
    fs.readSync(fd, buffer);
    const marker = Buffer.from('end_header\n');
    const headerEnd = buffer.indexOf(marker) + marker.length;
    if (headerEnd < marker.length) throw new Error('PLY header terminator missing');
    const header = buffer.subarray(0, headerEnd).toString('utf8');
    if (!header.includes('format binary_little_endian 1.0')) throw new Error('PLY is not binary little endian');
    const vertexCount = Number(header.match(/element vertex (\d+)/)?.[1]);
    const properties = header.split('\n').filter((line) => line.startsWith('property '));
    const expected = headerEnd + vertexCount * properties.length * 4;
    const bytes = fs.statSync(file).size;
    if (properties.length !== 62 || expected !== bytes) throw new Error('unexpected ABotN 3DGS PLY layout');
    return { bytes, header_bytes: headerEnd, vertex_count: vertexCount, property_count: properties.length };
  } finally {
    fs.closeSync(fd);
  }
}

function cameraConfig(annotation) {
  const pose = annotation.trajectory?.[0];
  if (!pose) throw new Error('annotation has no initial trajectory pose');
  const sourcePosition = [pose.x, pose.y, pose.z];
  const position = [pose.y, -pose.x, pose.z];
  const sourceForward = [
    Math.cos(pose.yaw) * Math.cos(pose.pitch),
    Math.sin(pose.yaw) * Math.cos(pose.pitch),
    Math.sin(pose.pitch)
  ];
  const forward = [sourceForward[1], -sourceForward[0], sourceForward[2]];
  return {
    source_initial_position: sourcePosition,
    source_initial_euler_radians: [pose.roll, pose.pitch, pose.yaw],
    point_cloud_coordinate_mapping: '(x,y,z)_simulator -> (y,-x,z)_point_cloud_rotated',
    camera_position: position,
    camera_look_at: position.map((value, index) => value + forward[index]),
    camera_up: [0, 0, 1],
    vertical_fov_degrees: 50,
    splat_alpha_removal_threshold: 5,
    instance_wait_timeout_ms: 300000
  };
}

function pixelStats(buffer) {
  const png = PNG.sync.read(buffer);
  let count = 0;
  let mean = 0;
  let m2 = 0;
  let black = 0;
  let white = 0;
  const sampledColors = new Set();
  for (let i = 0, pixel = 0; i < png.data.length; i += 4, pixel += 1) {
    const luma = 0.2126 * png.data[i] + 0.7152 * png.data[i + 1] + 0.0722 * png.data[i + 2];
    count += 1;
    const delta = luma - mean;
    mean += delta / count;
    m2 += delta * (luma - mean);
    if (luma < 5) black += 1;
    if (luma > 250) white += 1;
    if (pixel % 16 === 0) sampledColors.add(`${png.data[i]},${png.data[i + 1]},${png.data[i + 2]}`);
  }
  return {
    width: png.width,
    height: png.height,
    luma_mean: mean,
    luma_stddev: Math.sqrt(m2 / count),
    black_fraction: black / count,
    white_fraction: white / count,
    sampled_distinct_rgb: sampledColors.size
  };
}

function contentType(file) {
  return new Map([['.html', 'text/html'], ['.js', 'text/javascript'], ['.wasm', 'application/wasm']]).get(path.extname(file)) ?? 'application/octet-stream';
}

async function atomicWrite(file, bytes) {
  const temporary = `${file}.tmp`;
  await fsp.writeFile(temporary, bytes);
  await fsp.rename(temporary, file);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const ply = path.resolve(args.ply);
  const annotationPath = path.resolve(args.annotation);
  const outputDir = path.resolve(args['output-dir']);
  const chrome = path.resolve(args.chrome);
  if (!fs.existsSync(chrome)) throw new Error(`Chrome executable missing: ${chrome}`);
  const annotationBytes = await fsp.readFile(annotationPath);
  const annotation = JSON.parse(annotationBytes.toString('utf8'));
  const camera = cameraConfig(annotation);
  const plyInspection = inspectPly(ply);
  await fsp.mkdir(outputDir, { recursive: true });

  const servedConfig = {
    camera_position: camera.camera_position,
    camera_look_at: camera.camera_look_at,
    camera_up: camera.camera_up,
    vertical_fov_degrees: camera.vertical_fov_degrees,
    splat_alpha_removal_threshold: camera.splat_alpha_removal_threshold,
    instance_wait_timeout_ms: camera.instance_wait_timeout_ms
  };
  let server;
  let browser;
  const consoleMessages = [];
  try {
    server = http.createServer((request, response) => {
      const pathname = new URL(request.url, `http://${request.headers.host}`).pathname;
      let file;
      let body;
      if (pathname === '/favicon.ico') {
        response.writeHead(204).end();
        return;
      }
      if (pathname === '/config.json') body = Buffer.from(JSON.stringify(servedConfig));
      else if (pathname === '/scene.ply') file = ply;
      else if (pathname === '/' || pathname === '/index.html') file = path.join(root, 'index.html');
      else if (pathname === '/app.js') file = path.join(root, 'app.js');
      else if (pathname.startsWith('/node_modules/')) file = path.resolve(root, pathname.slice(1));
      if (file && !file.startsWith(root) && file !== ply) file = undefined;
      const headers = {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'same-origin'
      };
      if (body) {
        response.writeHead(200, { ...headers, 'Content-Type': 'application/json', 'Content-Length': body.length });
        response.end(body);
      } else if (file && fs.existsSync(file) && fs.statSync(file).isFile()) {
        response.writeHead(200, { ...headers, 'Content-Type': contentType(file), 'Content-Length': fs.statSync(file).size });
        fs.createReadStream(file).pipe(response);
      } else response.writeHead(404, headers).end();
    });
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const port = server.address().port;
    browser = await chromium.launch({
      executablePath: chrome,
      headless: true,
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--enable-webgl', '--ignore-gpu-blocklist']
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
    page.on('console', (message) => consoleMessages.push(`${message.type()}: ${message.text()}`));
    page.on('pageerror', (error) => consoleMessages.push(`pageerror: ${error.stack ?? error}`));
    await page.goto(`http://127.0.0.1:${port}`, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForFunction(() => window.__canary?.ready || window.__canary?.error, null, { timeout: 360000 });
    const browserState = await page.evaluate(() => window.__canary);
    if (browserState.error) throw new Error(browserState.error);
    await page.evaluate(() => { window.__viewer.stop(); window.__viewer.update(); window.__viewer.render(); });
    await page.waitForTimeout(100);
    const png = await page.screenshot({ timeout: 120000 });
    const stats = pixelStats(png);
    const screenshotPath = path.join(outputDir, 'initial_view.png');
    await atomicWrite(screenshotPath, png);
    const gates = {
      cross_origin_isolated: browserState.cross_origin_isolated === true,
      canvas_1280x720: browserState.canvas?.width === 1280 && browserState.canvas?.height === 720,
      all_retained_splats_submitted: browserState.splat_count > 0 && browserState.instance_count === browserState.splat_count,
      nondegenerate_pixels: stats.luma_stddev >= 10 && stats.black_fraction < 0.9 && stats.sampled_distinct_rgb >= 256
    };
    const passed = Object.values(gates).every(Boolean);
    const receipt = {
      schema_version: SCHEMA,
      created_at_utc: new Date().toISOString(),
      mode: 'REVERSIBLE_EXPLORATION_CANARY_LITE',
      terminal: passed ? 'WEBGL_RENDER_TRANSPORT_CANARY_PASS' : 'WEBGL_RENDER_TRANSPORT_CANARY_FAIL',
      dataset: { id: 'acvlab/ABotN-POIBench', revision: DATASET_REVISION },
      frozen_task: {
        selection_rule: 'smallest_public_scene_payload_then_lexicographically_first_task',
        annotation_path: annotationPath,
        annotation_sha256: crypto.createHash('sha256').update(annotationBytes).digest('hex'),
        trajectory_observation_count: annotation.trajectory.length,
        rendered_observation_index: 0
      },
      scene: { path: ply, sha256: sha256File(ply), ...plyInspection },
      renderer: {
        kind: 'UNOFFICIAL_WEBGL_MECHANICS_CANARY',
        package: RENDERER_PACKAGE,
        version: RENDERER_VERSION,
        browser_executable: chrome,
        browser_version: await browser.version(),
        spherical_harmonics_degree: 0,
        sort_backend: 'SHARED_MEMORY_CPU',
        render_backend: browserState.webgl
      },
      camera,
      provider_firewall: {
        renderer_served_fields: Object.keys(servedConfig).sort(),
        goal_instruction_served: false,
        target_position_served: false,
        distance_to_goal_served: false,
        teacher_outputs_served: false,
        private_truth_served: false
      },
      browser_state: browserState,
      screenshot: { path: screenshotPath, sha256: crypto.createHash('sha256').update(png).digest('hex'), bytes: png.length, pixel_stats: stats },
      gates,
      console_messages: consoleMessages.filter((message) => !message.includes('favicon.ico')),
      render_calls: 1,
      teacher_calls: 0,
      provider_calls: 0,
      baseline_calls: 0,
      truth_status: 'METRIC_ARRIVAL_ENDPOINT_ONLY_FUNCTIONAL_PIXEL_REGION_NOT_ESTABLISHED',
      claim_ceiling: 'WEBGL_RENDER_TRANSPORT_AND_ARRIVAL_SUBSTRATE_MECHANICS_ONLY_NOT_OFFICIAL_RENDERER_EQUIVALENCE',
      next_action: passed ? 'FREEZE_ONE_TASK_ARRIVAL_ONLY_PROVIDER_FIREWALL_CANARY_BEFORE_ANY_BASELINE' : 'STOP_AND_DIAGNOSE_RENDER_TRANSPORT'
    };
    await atomicWrite(path.join(outputDir, 'receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`);
    console.log(JSON.stringify({ terminal: receipt.terminal, output_dir: outputDir, gates, screenshot: receipt.screenshot }, null, 2));
    if (!passed) process.exitCode = 2;
  } finally {
    if (browser) await browser.close();
    if (server) await new Promise((resolve) => server.close(resolve));
  }
}

await main();
