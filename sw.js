// Service worker mínimo — só serve para a app ser "instalável".
// Não faz cache do conteúdo da Streamlit (que é dinâmico).

const CACHE = 'btc-shell-v1';
const SHELL = [
  './',
  './index.html',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/apple-touch-icon.png'
];

self.addEventListener('install', function (e) {
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE).then(function (c) { return c.addAll(SHELL); })
  );
});

self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(
        keys.filter(function (k) { return k !== CACHE; })
            .map(function (k) { return caches.delete(k); })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function (e) {
  const url = new URL(e.request.url);
  // Só serve do cache os ficheiros do shell (mesmo domínio).
  // Tudo o resto (Streamlit) vai sempre à rede.
  if (url.origin === self.location.origin) {
    e.respondWith(
      caches.match(e.request).then(function (r) {
        return r || fetch(e.request);
      })
    );
  }
});
