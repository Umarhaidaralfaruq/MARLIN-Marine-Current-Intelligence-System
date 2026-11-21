const CACHE_NAME = 'marlin-pwa-v2'; // Ubah angka versi ini
const urlsToCache = [
  './',
  './index.html',
  './forecaster.html',
  './logo_bmkg2.png',
  './radar-logo.jpg'
];

// Install Service Worker dan simpan file ke Cache
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
      })
  );
});

// Ambil file dari Cache saat offline/loading
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Kembalikan file dari cache jika ada, jika tidak ambil dari jaringan (API)
        return response || fetch(event.request);
      })
  );
});

// Hapus Cache lama jika ada update versi
self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});