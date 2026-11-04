// The data source's Web Worker, so that none of this holds up the page: it downloads or reads the file in a load,
// { load, source, hdu } with a File or URL as the source, and fits-worker holds the cube and answers the viewer's
// requests. ArrayBuffers both ways are the source's messages.
import init, { held, open, request, work } from "./pkg/fits_worker.js";

const ready = init();
ready.catch((e) => postMessage({ failed: `could not start: ${e.message ?? e}` }));

// A job at a time, each in a task of its own, so that requests (a cursor value, cubelets no longer wanted) are
// taken in between.
const jobs = new MessageChannel();
let scheduled = false;
jobs.port1.onmessage = () => {
  scheduled = false;
  if (work()) schedule();
};
function schedule() {
  if (!scheduled) {
    scheduled = true;
    jobs.port2.postMessage(null);
  }
}

onmessage = async ({ data }) => {
  await ready;
  if (data instanceof ArrayBuffer) {
    request(new Uint8Array(data));
    schedule();
  } else {
    load(data);
  }
};

// Only the latest load is read, so a slow download can't replace a file picked after it started.
let latest;

async function load({ load: id, source, hdu }) {
  latest?.abort();
  const current = (latest = new AbortController());
  try {
    const bytes = typeof source === "string" ? await download(source, current.signal, reporter(id)) : await source.bytes();
    if (current.signal.aborted) return;
    postMessage({ load: id, reading: true });
    const start = performance.now();
    const info = open(id, bytes, hdu);
    postMessage({ load: id, info, seconds: (performance.now() - start) / 1000 }, [info]);
  } catch (e) {
    // With the load whose file it still holds, which may not be the one the page shows.
    if (!current.signal.aborted) postMessage({ load: id, error: e.message ?? String(e), held: held() });
  }
}

// Download progress for the page's status, ten times a second at most.
function reporter(id) {
  let last = 0;
  return (got, total) => {
    const now = performance.now();
    if (now - last >= 100) {
      last = now;
      postMessage({ load: id, got, total });
    }
  };
}

// A big file comes down as four ranges at once: in Chrome one stream only reached about half the line speed.
const PARTS = 4;
async function download(url, signal, progress) {
  const head = await fetch(url, { method: "HEAD", signal }).catch(() => null);
  const total = head?.ok && !head.headers.get("Content-Encoding") ? Number(head.headers.get("Content-Length")) : 0;
  if (total >= PARTS << 23) {
    try {
      return await downloadRanges(url, total, signal, progress);
    } catch (e) {
      if (signal.aborted) throw e;
    }
  }
  return downloadStream(url, signal, progress);
}

// A BYOB read transfers the buffer it fills, so the ranges can't share one: each reads into its own scratch
// buffer and copies from it into the result.
async function downloadRanges(url, total, signal, progress) {
  const abort = new AbortController();
  signal.addEventListener("abort", () => abort.abort(), { once: true });
  const bytes = new Uint8Array(total);
  const size = Math.ceil(total / PARTS);
  let got = 0;
  try {
    await Promise.all(Array.from({ length: PARTS }, async (_, i) => {
      const start = i * size, end = Math.min(total, start + size);
      const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` }, signal: abort.signal });
      if (res.status !== 206) throw new Error(`no range support (${res.status})`);
      const reader = res.body.getReader({ mode: "byob" });
      let scratch = new ArrayBuffer(1 << 20);
      for (let at = start; at < end; ) {
        const { done, value } = await reader.read(new Uint8Array(scratch));
        if (done || at + value.length > end) throw new Error("range doesn't match its request");
        bytes.set(value, at);
        at += value.length;
        got += value.length;
        scratch = value.buffer;
        progress(got, total);
      }
    }));
  } catch (e) {
    abort.abort();
    throw e;
  }
  return bytes;
}

// Reads into one buffer sized from Content-Length, so the file is held once rather than as chunks plus a copy.
async function downloadStream(url, signal, progress) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  // With a Content-Encoding it's the compressed size, so the buffer grows instead.
  const total = res.headers.get("Content-Encoding") ? 0 : Number(res.headers.get("Content-Length"));
  let got = 0;
  // A BYOB reader fills the buffer in place, where the default reader hands back a new array per chunk that
  // piles up as garbage until the load is over.
  let byob = null;
  try {
    if (total) byob = res.body.getReader({ mode: "byob" });
  } catch {}
  if (byob) {
    let buffer = new ArrayBuffer(total);
    while (got < total) {
      const { done, value } = await byob.read(new Uint8Array(buffer, got));
      if (done) break;
      // Each read transfers the buffer, so carry on with the one it hands back.
      buffer = value.buffer;
      got += value.length;
      progress(got, total);
    }
    return new Uint8Array(buffer, 0, got);
  }
  const reader = res.body.getReader();
  let bytes = new Uint8Array(total || 1 << 20);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > bytes.length) {
      const grown = new Uint8Array(Math.max(2 * bytes.length, got + value.length));
      grown.set(bytes.subarray(0, got));
      bytes = grown;
    }
    bytes.set(value, got);
    got += value.length;
    progress(got, total);
  }
  return bytes.subarray(0, got);
}
