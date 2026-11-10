/* ---------- where are we running? ---------- */
// Inside the Android app, Capacitor serves the page from https://localhost.
const IS_NATIVE =
  Boolean(window.Capacitor?.isNativePlatform?.()) || location.origin === 'https://localhost';

// Browser: the library comes from the CDN (this exact URL must match LIB_URL in sw.js).
// Android app: a local copy that the GitHub workflow places in /vendor/.
const LIB_URL = IS_NATIVE
  ? '/vendor/transformers.min.js'
  : 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
const { pipeline, env } = await import(LIB_URL);

const CONFIG = {
  task: 'image-classification',
  modelId: 'onnx-community/face-emotion-detection-ONNX',
  device: 'wasm',     // CPU through WebAssembly
  dtype: 'q8',        // -> onnx/model_quantized.onnx (87 MB)
  cropRatio: 0.6,     // camera: centered square = 60% of the frame's shorter side
  loopDelayMs: 150,   // pause between camera predictions
  apkFile: 'face-emotion-detection.apk', // must match APK_NAME in the GitHub workflow
  apkUrl: '',         // optional: full APK link; leave empty on GitHub Pages (it is worked out for you)
};

// config.json only contains LABEL_0 ... LABEL_6. This order comes from the model card.
const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

if (IS_NATIVE) {
  // Everything is inside the app: never use the network or the browser cache.
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = '/models/';   // the workflow copies the model to /models/<model id>/
  env.useBrowserCache = false;
  env.useWasmCache = false;
  const ort = `${location.origin}/ort/`; // ONNX Runtime files copied by the workflow
  env.backends.onnx.wasm.wasmPaths = {
    mjs: `${ort}ort-wasm-simd-threaded.asyncify.mjs`,
    wasm: `${ort}ort-wasm-simd-threaded.asyncify.wasm`,
  };
  env.backends.onnx.wasm.numThreads = 1;
} else {
  env.allowLocalModels = false; // always use the Hugging Face Hub (skips a local-file check)
  env.useBrowserCache = true;   // keep downloaded model files in Cache Storage
  env.useWasmCache = true;      // also keep ONNX Runtime's WASM files (needed for offline use)
}

const $ = (id) => document.getElementById(id);
const el = {
  net: $('net-status'), model: $('model-status'), progress: $('progress'),
  cache: $('cache-status'), shell: $('shell-status'), storage: $('storage-status'),
  persist: $('persist-status'), btnPersist: $('btn-persist'),
  video: $('video'), guide: $('guide'), btnCamera: $('btn-camera'),
  btnUpload: $('btn-upload'), file: $('file'), crop: $('crop'),
  top: $('top-result'), bars: $('bars'),
  apkCard: $('apk-card'), apkInfo: $('apk-info'), btnApk: $('btn-apk'),
};
const cropCtx = el.crop.getContext('2d', { willReadFrequently: true });

let classifier = null;
let stream = null;
let cameraOn = false;
let queue = Promise.resolve();

/* ---------- status panel ---------- */
function updateNet() {
  el.net.textContent = navigator.onLine ? 'Online' : 'Offline';
}

async function modelInBrowserCache() {
  try {
    if (typeof caches === 'undefined') return false;
    const cache = await caches.open(env.cacheKey); // 'transformers-cache'
    const keys = await cache.keys();
    return keys.some((r) => r.url.includes(CONFIG.modelId) && r.url.includes('.onnx'));
  } catch {
    return false;
  }
}

async function showStorage() {
  try {
    const { usage } = await navigator.storage.estimate();
    const persistent = await navigator.storage.persisted();
    el.storage.textContent = `${Math.round(usage / 1048576)} MB${persistent ? ' (persistent)' : ''}`;
  } catch {
    el.storage.textContent = 'n/a';
  }
}

// Shows whether the browser has promised not to delete this site's storage and, if asked, requests that promise.
async function protectStorage(ask) {
  try {
    if (!navigator.storage?.persisted) {
      el.persist.textContent = 'Not supported by this browser';
      return;
    }
    let isProtected = await navigator.storage.persisted();
    if (!isProtected && ask && navigator.storage.persist) {
      isProtected = await navigator.storage.persist(); // the browser decides; it may say no
    }
    el.persist.textContent = isProtected ? 'Yes' : 'No (the browser may delete the model if disk space runs low)';
    el.btnPersist.hidden = isProtected;
  } catch {
    el.persist.textContent = 'n/a';
  }
  showStorage();
}

async function setupServiceWorker() {
  if (IS_NATIVE) return; // the Android app has everything built in
  if (!('serviceWorker' in navigator)) {
    el.shell.textContent = 'Not supported';
    return;
  }
  try {
    await navigator.serviceWorker.register('./sw.js');
    await navigator.serviceWorker.ready;
    const hit = await caches.match(new URL('./index.html', location.href).href);
    el.shell.textContent = hit ? 'Yes' : 'No';
  } catch (err) {
    console.warn('Service worker failed:', err);
    el.shell.textContent = 'Failed (see console)';
  }
}

// Inside the Android app the browser-cache rows make no sense, so hide them.
function hideBrowserOnlyRows() {
  for (const id of ['cache-status', 'shell-status', 'storage-status', 'persist-status', 'btn-persist']) {
    const node = document.getElementById(id);
    const target = node?.closest('.row') ?? node;
    target?.style.setProperty('display', 'none', 'important');
  }
}

/* ---------- "Get the Android app" card (Android browsers only) ---------- */
function apkRepo() {
  const m = /^([a-z0-9-]+)\.github\.io$/i.exec(location.hostname);
  if (!m) return null; // not on GitHub Pages
  return { owner: m[1], repo: location.pathname.split('/')[1] || `${m[1]}.github.io` };
}

async function setupApkButton() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isInstalledPwa = window.matchMedia('(display-mode: standalone)').matches;
  if (IS_NATIVE || isInstalledPwa || !isAndroid || !navigator.onLine) return;

  let url = CONFIG.apkUrl;
  let info = '';
  if (!url) {
    const gh = apkRepo();
    if (!gh) return; // not on GitHub Pages and no apkUrl set
    url = `https://github.com/${gh.owner}/${gh.repo}/releases/latest/download/${CONFIG.apkFile}`;
    try {
      // Ask GitHub if a release with the APK exists; this also gives us the version and the file size.
      const res = await fetch(`https://api.github.com/repos/${gh.owner}/${gh.repo}/releases/latest`);
      if (res.status === 404) return; // no release yet, so do not show a dead link
      if (res.ok) {
        const release = await res.json();
        const asset = release.assets?.find((a) => a.name === CONFIG.apkFile);
        if (!asset) return; // the latest release has no APK
        info = `Version ${release.tag_name} · ${Math.round(asset.size / 1048576)} MB download`;
      }
      // any other answer (for example "rate limited"): show the button anyway
    } catch {
      // network problem while asking: show the button anyway
    }
  }
  el.apkInfo.textContent = info;
  el.apkCard.hidden = false;
  el.btnApk.addEventListener('click', () => { location.href = url; });
}

/* ---------- model ---------- */
async function loadModel() {
  const cachedBefore = IS_NATIVE ? false : await modelInBrowserCache();
  el.model.textContent = IS_NATIVE
    ? 'Loading the model built into the app…'
    : cachedBefore ? 'Loading from browser cache…' : 'Downloading (first visit only)…';
  el.progress.hidden = false;

  const options = { device: CONFIG.device, dtype: CONFIG.dtype };
  if (!IS_NATIVE) {
    options.progress_callback = (e) => {
      if (e.status === 'progress_total' && typeof e.progress === 'number') {
        el.progress.value = Math.min(100, Math.max(0, e.progress));
      }
    };
  }
  classifier = await pipeline(CONFIG.task, CONFIG.modelId, options);

  el.progress.hidden = true;
  if (IS_NATIVE) {
    el.model.textContent = 'Ready (built into the app, works offline)';
    return;
  }
  el.model.textContent = cachedBefore ? 'Ready (loaded from cache)' : 'Ready (downloaded and cached)';
  el.cache.textContent = (await modelInBrowserCache()) ? 'Yes' : 'No';
}

/* ---------- prediction ---------- */
function drawCrop(source, sw, sh, ratio) {
  const side = Math.min(sw, sh) * ratio;
  cropCtx.drawImage(source, (sw - side) / 2, (sh - side) / 2, side, side, 0, 0, el.crop.width, el.crop.height);
}

function labelName(label) {
  const m = /^LABEL_(\d+)$/.exec(label);
  return m ? (EMOTIONS[Number(m[1])] ?? label) : label;
}

function showResults(results) {
  const items = results.map((r) => ({ name: labelName(r.label), pct: r.score * 100 }));
  el.top.textContent = `${items[0].name} · ${items[0].pct.toFixed(1)}%`;
  el.bars.innerHTML = items.map((i) => `
    <div class="bar-row">
      <span>${i.name}</span>
      <div class="bar"><div class="fill" style="width:${i.pct.toFixed(1)}%"></div></div>
      <span>${i.pct.toFixed(1)}%</span>
    </div>`).join('');
}

// Classifies whatever is currently drawn in the 224x224 canvas.
// Jobs are queued so two predictions never overlap.
function classifyCanvas() {
  queue = queue
    .then(async () => {
      const results = await classifier(el.crop, { top_k: EMOTIONS.length });
      showResults(results);
    })
    .catch((err) => console.error('Prediction failed:', err));
  return queue;
}

/* ---------- camera ---------- */
function positionGuide() {
  const { videoWidth: vw, videoHeight: vh } = el.video;
  if (!vw || !vh) return;
  const side = Math.min(vw, vh) * CONFIG.cropRatio;
  el.guide.style.width = `${(side / vw) * 100}%`;
  el.guide.style.height = `${(side / vh) * 100}%`;
}

async function cameraLoop() {
  while (cameraOn) {
    const v = el.video;
    if (v.readyState >= 2 && v.videoWidth) {
      drawCrop(v, v.videoWidth, v.videoHeight, CONFIG.cropRatio);
      await classifyCanvas();
    }
    await new Promise((r) => setTimeout(r, CONFIG.loopDelayMs));
  }
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  el.video.srcObject = stream;
  await el.video.play();
  positionGuide();
  el.guide.hidden = false;
  cameraOn = true;
  el.btnCamera.textContent = 'Stop camera';
  cameraLoop();
}

function stopCamera() {
  cameraOn = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  el.video.srcObject = null;
  el.guide.hidden = true;
  el.btnCamera.textContent = 'Start camera';
}

el.btnCamera.addEventListener('click', async () => {
  if (cameraOn) return stopCamera();
  try {
    await startCamera();
  } catch (err) {
    console.error(err);
    stopCamera();
    alert('Could not start the camera: ' + err.message);
  }
});

el.btnUpload.addEventListener('click', () => el.file.click());

el.file.addEventListener('change', async () => {
  const file = el.file.files?.[0];
  if (!file) return;
  stopCamera();
  const bmp = await createImageBitmap(file);
  drawCrop(bmp, bmp.width, bmp.height, 1); // uploads: centered square of the whole photo
  bmp.close();
  await classifyCanvas();
  el.file.value = '';
});

el.btnPersist.addEventListener('click', () => protectStorage(true));

/* ---------- start ---------- */
async function init() {
  updateNet();
  addEventListener('online', updateNet);
  addEventListener('offline', updateNet);
  setupApkButton().catch((err) => console.warn('Android card:', err));
  if (IS_NATIVE) {
    hideBrowserOnlyRows();
  } else {
    setupServiceWorker(); // runs in parallel
    showStorage();
  }

  try {
    await loadModel();
  } catch (err) {
    console.error(err);
    el.progress.hidden = true;
    const reason = String(err?.message || err).slice(0, 140);
    const prefix = IS_NATIVE
      ? 'Failed to load the built-in model'
      : navigator.onLine ? 'Failed to load' : 'Failed to load while offline (is the model cached yet?)';
    el.model.textContent = `${prefix}: ${reason}`;
    return;
  }

  el.btnCamera.disabled = false;
  el.btnUpload.disabled = false;
  if (!IS_NATIVE) await protectStorage(true);
}

init();