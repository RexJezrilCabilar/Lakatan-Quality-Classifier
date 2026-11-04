// Caches the app's own files plus files from cdn.jsdelivr.net so the page opens offline.
// Hugging Face model files are NOT handled here: Transformers.js stores them itself
// in Cache Storage under the name "transformers-cache".

const VERSION = 'v3';                          // bump this every time you change a file in the repo
const SHELL_CACHE = `emotion-shell-${VERSION}`;

// Must be exactly the same URL as the import in app.js
const LIB_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const PRECACHE = [
  './', './index.html', './style.css', './app.js',
  './manifest.webmanifest',
  './icons/icon-192.png', './icons/icon-512.png', './icons/icon-maskable-512.png', './icons/apple-touch-icon.png',
  LIB_URL,
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    // cache: 'reload' skips the browser's HTTP cache, so we never store a stale copy
    await cache.addAll(PRECACHE.map((url) => new Request(url, { cache: 'reload' })));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(
      names
        .filter((n) => n.startsWith('emotion-shell-') && n !== SHELL_CACHE) // never touch "transformers-cache"
        .map((n) => caches.delete(n))
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  const isOwnFile = url.origin === self.location.origin;
  const isCdnFile = url.hostname === 'cdn.jsdelivr.net';
  if (!isOwnFile && !isCdnFile) return; // everything else (Hugging Face, etc.) goes straight to the network

  event.respondWith((async () => {
    const cache = await caches.open(SHELL_CACHE);
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;

    try {
      const response = await fetch(request);
      if (isCdnFile && response.status === 200) {
        event.waitUntil(cache.put(request, response.clone())); // also keep CDN files such as ONNX Runtime WASM
      }
      return response;
    } catch (err) {
      if (request.mode === 'navigate') {
        const fallback = await cache.match('./index.html');
        if (fallback) return fallback;
      }
      throw err;
    }
  })());
});