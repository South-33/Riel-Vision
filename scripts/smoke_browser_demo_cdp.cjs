const fs = require("node:fs");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const DEFAULT_EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const CLASS_NAMES = [
  "USD_1",
  "USD_5",
  "USD_10",
  "USD_20",
  "USD_50",
  "USD_100",
  "KHR_500",
  "KHR_1000",
  "KHR_2000",
  "KHR_5000",
  "KHR_10000",
  "KHR_20000",
  "KHR_50000",
];
const VALUES = {
  USD_1: 1,
  USD_5: 5,
  USD_10: 10,
  USD_20: 20,
  USD_50: 50,
  USD_100: 100,
  KHR_500: 500,
  KHR_1000: 1000,
  KHR_2000: 2000,
  KHR_5000: 5000,
  KHR_10000: 10000,
  KHR_20000: 20000,
  KHR_50000: 50000,
};

function parseArgs(argv) {
  const args = {
    image: "/data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png",
    port: 8787,
    debugPort: 9223,
    timeoutMs: 120000,
    screenshot: "",
    outCsv: "",
    labels: "",
    matchIou: 0.5,
    edge: process.env.EDGE_PATH || DEFAULT_EDGE,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--image") {
      args.image = value;
      index += 1;
    } else if (key === "--port") {
      args.port = Number(value);
      index += 1;
    } else if (key === "--debug-port") {
      args.debugPort = Number(value);
      index += 1;
    } else if (key === "--timeout-ms") {
      args.timeoutMs = Number(value);
      index += 1;
    } else if (key === "--screenshot") {
      args.screenshot = value;
      index += 1;
    } else if (key === "--out-csv") {
      args.outCsv = value;
      index += 1;
    } else if (key === "--labels") {
      args.labels = value;
      index += 1;
    } else if (key === "--match-iou") {
      args.matchIou = Number(value);
      index += 1;
    } else if (key === "--edge") {
      args.edge = value;
      index += 1;
    } else if (key === "--help" || key === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${key}`);
    }
  }
  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/smoke_browser_demo_cdp.cjs [options]

Options:
  --image PATH        Repo-root browser image URL path.
  --labels PATH       Optional YOLO detect labels for smoke evaluation.
  --match-iou NUMBER  IoU threshold for --labels evaluation. Default: 0.5.
  --screenshot PATH   Optional PNG screenshot output path.
  --out-csv PATH      Optional CSV output for browser-side detections.
  --port NUMBER       Local static server port. Default: 8787.
  --debug-port NUMBER Edge DevTools port. Default: 9223.
  --timeout-ms NUMBER Autorun wait timeout. Default: 120000.
  --edge PATH         Microsoft Edge executable path.`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function killTree(child) {
  if (!child || !child.pid) return;
  if (process.platform === "win32") {
    spawnSync("powershell", ["-NoProfile", "-Command", `Stop-Process -Id ${child.pid} -Force -ErrorAction SilentlyContinue`], {
      stdio: "ignore",
      windowsHide: true,
    });
  } else {
    child.kill("SIGTERM");
  }
}

function stopWindowsProcessesByCommandLine(pattern) {
  if (process.platform !== "win32") return;
  const escaped = pattern.replaceAll("'", "''");
  spawnSync(
    "powershell",
    [
      "-NoProfile",
      "-Command",
      `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*${escaped}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`,
    ],
    { stdio: "ignore", windowsHide: true },
  );
}

async function waitForHttp(url, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function connectCdp(debugPort, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const pages = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
      const page = pages.find((item) => (item.url || "").includes("/demo/browser/")) || pages[0];
      if (page?.webSocketDebuggerUrl) {
        const ws = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((resolve, reject) => {
          ws.onopen = resolve;
          ws.onerror = reject;
        });
        return ws;
      }
    } catch {
      // Edge is still starting.
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for Edge CDP on port ${debugPort}`);
}

function createCdpClient(ws) {
  let id = 0;
  const pending = new Map();
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  };
  return function send(method, params = {}) {
    const messageId = ++id;
    ws.send(JSON.stringify({ id: messageId, method, params }));
    return new Promise((resolve) => pending.set(messageId, resolve));
  };
}

function readYoloLabels(labelsPath, imageWidth, imageHeight) {
  if (!labelsPath) return null;
  const absolute = path.resolve(ROOT, labelsPath);
  const text = fs.readFileSync(absolute, "utf8");
  const labels = [];
  text.split(/\r?\n/).forEach((line, lineIndex) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return;
    const parts = trimmed.split(/\s+/).map(Number);
    if (parts.length !== 5 || parts.some((part) => Number.isNaN(part))) {
      throw new Error(`${labelsPath}: line ${lineIndex + 1} must be YOLO detect format`);
    }
    const [classId, cx, cy, width, height] = parts;
    labels.push({
      classId,
      name: CLASS_NAMES[classId] || `class_${classId}`,
      x1: (cx - width / 2) * imageWidth,
      y1: (cy - height / 2) * imageHeight,
      x2: (cx + width / 2) * imageWidth,
      y2: (cy + height / 2) * imageHeight,
    });
  });
  return labels;
}

function boxIou(left, right) {
  const x1 = Math.max(left.x1, right.x1);
  const y1 = Math.max(left.y1, right.y1);
  const x2 = Math.min(left.x2, right.x2);
  const y2 = Math.min(left.y2, right.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  const union = leftArea + rightArea - intersection;
  return union ? intersection / union : 0;
}

function greedyMatch(detections, labels, matchIou, requireSameClass) {
  const usedLabels = new Set();
  let matches = 0;
  [...detections].sort((left, right) => Number(right.score || 0) - Number(left.score || 0)).forEach((detection) => {
    let bestIndex = -1;
    let bestIou = 0;
    labels.forEach((label, index) => {
      if (usedLabels.has(index)) return;
      if (requireSameClass && detection.name !== label.name) return;
      const iou = boxIou(detection, label);
      if (iou > bestIou) {
        bestIou = iou;
        bestIndex = index;
      }
    });
    if (bestIndex >= 0 && bestIou >= matchIou) {
      usedLabels.add(bestIndex);
      matches += 1;
    }
  });
  return matches;
}

function sourceDetections(rawDetections, nameKey, scoreKey) {
  return rawDetections
    .filter((detection) => detection[nameKey])
    .map((detection) => ({
      ...detection,
      name: detection[nameKey],
      score: Number(detection[scoreKey] || 0),
    }));
}

function evaluateSource(detections, labels, matchIou) {
  const matchedSameClass = greedyMatch(detections, labels, matchIou, true);
  const matchedAnyClass = greedyMatch(detections, labels, matchIou, false);
  return {
    predCount: detections.length,
    countError: detections.length - labels.length,
    matchedSameClass,
    matchedAnyClass,
    recallSameClass: labels.length ? matchedSameClass / labels.length : 0,
    recallAnyClass: labels.length ? matchedAnyClass / labels.length : 0,
  };
}

function evaluateDetections(value, args) {
  const labels = readYoloLabels(args.labels, value.imageWidth, value.imageHeight);
  if (!labels) return null;
  const detections = value.detections || [];
  const sources = {
    final: evaluateSource(sourceDetections(detections, "name", "score"), labels, args.matchIou),
    detector: evaluateSource(sourceDetections(detections, "detectorName", "detectorScore"), labels, args.matchIou),
    fragment: evaluateSource(sourceDetections(detections, "fragmentName", "fragmentScore"), labels, args.matchIou),
  };
  return {
    labels: path.relative(ROOT, path.resolve(ROOT, args.labels)),
    matchIou: args.matchIou,
    gtCount: labels.length,
    ...sources.final,
    sources,
  };
}

function summarize(value, args) {
  const counts = {};
  let khrValue = 0;
  let usdValue = 0;
  for (const detection of value.detections || []) {
    counts[detection.name] = (counts[detection.name] || 0) + 1;
    if (detection.name.startsWith("KHR_")) {
      khrValue += VALUES[detection.name] || 0;
    } else if (detection.name.startsWith("USD_")) {
      usdValue += VALUES[detection.name] || 0;
    }
  }
  return {
    status: value.status,
    totalCount: Number(value.totalCount || 0),
    khrValue,
    usdValue,
    predClasses: Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right))),
    debug: value.debug || {},
    evaluation: evaluateDetections(value, args),
    detections: value.detections || [],
  };
}

async function readBrowserState(send, timeoutMs) {
  await send("Runtime.enable");
  const expression = `(() => JSON.stringify({
    status: document.getElementById('modelStatus')?.textContent?.trim(),
    runButton: document.getElementById('runButton')?.textContent?.trim(),
    totalCount: document.getElementById('totalCount')?.textContent?.trim(),
    imageWidth: typeof state !== 'undefined' && state.image ? state.image.width : 0,
    imageHeight: typeof state !== 'undefined' && state.image ? state.image.height : 0,
    detections: typeof state !== 'undefined' ? state.detections : null,
    debug: typeof state !== 'undefined' ? state.debug : null
  }))()`;
  const started = Date.now();
  let value = null;
  while (Date.now() - started < timeoutMs) {
    const result = await send("Runtime.evaluate", { expression, returnByValue: true });
    if (result.result.exceptionDetails) {
      throw new Error(result.result.exceptionDetails.text || "CDP evaluation failed");
    }
    value = JSON.parse(result.result.result.value);
    if (value.totalCount !== "0" && value.runButton === "Run") return value;
    if (value.status === "Model load failed") return value;
    await sleep(1000);
  }
  throw new Error(`Timed out waiting ${timeoutMs}ms for browser autorun`);
}

async function captureScreenshot(send, outputPath) {
  if (!outputPath) return;
  const result = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const absolute = path.resolve(ROOT, outputPath);
  fs.mkdirSync(path.dirname(absolute), { recursive: true });
  fs.writeFileSync(absolute, Buffer.from(result.result.data, "base64"));
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeDetectionCsv(outputPath, detections) {
  if (!outputPath) return;
  const absolute = path.resolve(ROOT, outputPath);
  fs.mkdirSync(path.dirname(absolute), { recursive: true });
  const fieldnames = [
    "index",
    "x1",
    "y1",
    "x2",
    "y2",
    "detector_class",
    "detector_conf",
    "fragment_class",
    "fragment_conf",
    "final_class",
    "final_score",
  ];
  const lines = [fieldnames.join(",")];
  detections.forEach((detection, index) => {
    const row = {
      index,
      x1: detection.x1,
      y1: detection.y1,
      x2: detection.x2,
      y2: detection.y2,
      detector_class: detection.detectorName,
      detector_conf: detection.detectorScore,
      fragment_class: detection.fragmentName,
      fragment_conf: detection.fragmentScore,
      final_class: detection.name,
      final_score: detection.score,
    };
    lines.push(fieldnames.map((field) => csvCell(row[field])).join(","));
  });
  fs.writeFileSync(absolute, `${lines.join("\n")}\n`);
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(args.edge)) {
    throw new Error(`Edge executable not found: ${args.edge}`);
  }
  const profileDir = path.join(ROOT, ".cache_runtime", "edge-cdp-smoke-profile");
  fs.rmSync(profileDir, { recursive: true, force: true });
  fs.mkdirSync(profileDir, { recursive: true });

  const server = spawn("python", ["-m", "http.server", String(args.port), "--bind", "127.0.0.1"], {
    cwd: ROOT,
    stdio: "ignore",
    windowsHide: true,
  });
  let edge = null;
  try {
    await waitForHttp(`http://127.0.0.1:${args.port}/demo/browser/index.html`, 15000);
    const url = `http://127.0.0.1:${args.port}/demo/browser/?image=${args.image}&autorun=1`;
    edge = spawn(
      args.edge,
      [
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        `--remote-debugging-port=${args.debugPort}`,
        `--user-data-dir=${profileDir}`,
        "--window-size=1280,900",
        url,
      ],
      { stdio: "ignore", windowsHide: true },
    );
    const ws = await connectCdp(args.debugPort, 15000);
    const send = createCdpClient(ws);
    const value = await readBrowserState(send, args.timeoutMs);
    await captureScreenshot(send, args.screenshot);
    writeDetectionCsv(args.outCsv, value.detections || []);
    ws.close();
    console.log(JSON.stringify(summarize(value, args), null, 2));
  } finally {
    killTree(edge);
    killTree(server);
    stopWindowsProcessesByCommandLine(profileDir);
    stopWindowsProcessesByCommandLine(`http.server ${args.port}`);
  }
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
