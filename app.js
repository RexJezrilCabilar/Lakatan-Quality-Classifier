// The version in this URL must match LIB_URL in sw.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const CONFIG = {
  task: 'image-classification',
  modelId: 'onnx-community/face-emotion-detection-ONNX',
  device: 'wasm',     // CPU through WebAssembly
  dtype: 'q8',        // -> onnx/model_quantized.onnx (87 MB)
  cropRatio: 0.6,     // camera: centered square = 60% of the frame's shorter side
  loopDelayMs: 150,   // pause between camera predictions
};

// config.json only contains LABEL_0 ... LABEL_6. This order comes from the model card.
const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

env.allowLocalModels = false; // always use the Hugging Face Hub (skips a local-file check)
env.useBrowserCache = true;   // keep downloaded model files in Cache Storage
env.useWasmCache = true;      // also keep ONNX Runtime's WASM files (needed for offline use)

const $ = (id) => document.getElementById(id);
const el = {
  net: $('net-status'), model: $('model-status'), progress: $('progress'),
  cache: $('cache-status'), shell: $('shell-status'), storage: $('storage-status'),
  video: $('video'), guide: $('guide'), btnCamera: $('btn-camera'),
  btnUpload: $('btn-upload'), file: $('file'), crop: $('crop'),
  top: $('top-result'), bars: $('bars'),
  persist: $('persist-status'), btnPersist: $('btn-persist'),
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

/* ---------- model ---------- */
async function loadModel() {
  const cachedBefore = await modelInBrowserCache();
  el.model.textContent = cachedBefore ? 'Loading from browser cache…' : 'Downloading (first visit only)…';
  el.progress.hidden = false;

  classifier = await pipeline(CONFIG.task, CONFIG.modelId, {
    device: CONFIG.device,
    dtype: CONFIG.dtype,
    progress_callback: (e) => {
      if (e.status === 'progress_total' && typeof e.progress === 'number') {
        el.progress.value = Math.min(100, Math.max(0, e.progress));
      }
    },
  });

  el.progress.hidden = true;
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
  setupServiceWorker(); // runs in parallel
  showStorage();

  try {
    await loadModel();
  } catch (err) {
    console.error(err);
    el.progress.hidden = true;
    el.model.textContent = navigator.onLine
      ? 'Failed to load (open the console for details)'
      : 'Offline, and the model is not cached in this browser yet';
    return;
  }

  el.btnCamera.disabled = false;
  el.btnUpload.disabled = false;
  await protectStorage(true);
}

init();