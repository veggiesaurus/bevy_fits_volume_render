// The data source's Web Worker, so that none of this holds up the page: it downloads or reads the file in a load,
// { load, source, hdu } with a File or URL as the source, and fits-worker holds the cube and answers the viewer's
// requests, until { release } says the viewer no longer shows it. A URL ending in .cubelets is a baked cube, whose
// cubelets are read by range request as they're asked for. ArrayBuffers both ways are the source's messages. It also
// holds the socket to a fits-server, opened with { server, token, id, memory }, and { remote } messages both ways are
// the viewer's for that connection.
import init, {
  baked_attach, baked_failed, baked_fetched, baked_fetches, baked_open, baked_wanted, held, open, release, remote_attach,
  remote_closed, remote_command, remote_pack, remote_receive, remote_start, remote_work, request, work,
} from "./pkg/fits_worker.js";

const ready = init();
ready.catch((e) => postMessage({ failed: `could not start: ${e.message ?? e}` }));

// A job at a time, each in a task of its own, so that requests (a cursor value, cubelets no longer wanted) are
// taken in between. A server's cubelets that were in the cache are decoded the same way.
const jobs = new MessageChannel();
let scheduled = false;
jobs.port1.onmessage = () => {
  scheduled = false;
  const sent = remote_work();
  if (sent) toServer(sent);
  const more = work();
  readBaked();
  if (more || sent) schedule();
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
    readBaked();
    schedule();
  } else if (data.remote) {
    toServer(remote_command(new Uint8Array(data.remote)));
    schedule();
  } else if (data.release !== undefined) {
    release(data.release);
  } else if (data.server) {
    connectServer(data);
  } else {
    load(data);
  }
};

// The server's socket. fits-worker turns what comes in into the viewer's messages, and the viewer's commands into
// what goes out. The token goes in the first message rather than the URL, which browsers print when a connection
// fails.
let socket;
let unsent = [];

function connectServer({ server: url, token, id, memory }) {
  closeServer("another connection took its place");
  unsent = [remote_start(id, url, token, memory, clearPacks)];
  let ws;
  try {
    ws = socket = new WebSocket(url);
  } catch (e) {
    socket = undefined;
    remote_closed(`not a server URL: ${e.message ?? e}`);
    return;
  }
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    for (const message of unsent) ws.send(message);
    unsent = [];
  };
  ws.onmessage = ({ data }) => {
    try {
      toServer(remote_receive(new Uint8Array(data)));
    } catch (e) {
      closeServer(e.message ?? String(e));
    }
    const pack = remote_pack();
    if (pack) {
      openPack(pack).then(([handle, room]) => remote_attach(pack, handle, room), (e) => {
        console.warn(`Cubelets are kept in memory only: ${e.message ?? e}`);
        remote_attach(pack, null, 0);
      });
    }
  };
  // A browser doesn't say why a socket failed, only that it closed.
  ws.onclose = ({ code, reason }) =>
    closeServer(reason || (code === 1006 ? "couldn't connect, or the connection was lost" : `closed (${code})`));
}

function toServer(messages) {
  for (const message of messages) {
    if (socket?.readyState === WebSocket.OPEN) socket.send(message);
    else unsent.push(message);
  }
}

function closeServer(reason) {
  const ws = socket;
  socket = undefined;
  unsent = [];
  if (ws) {
    ws.onopen = ws.onmessage = ws.onclose = null;
    ws.close();
    remote_closed(reason);
  }
}

// The server's cubelets are kept in a pack per cube in OPFS, under a cap on them all, the least recently used going
// first to make room for the one opened. Without OPFS they're kept in memory only.
const CAP = 4 * 2 ** 30, RESERVE = 2 ** 30;
const packs = (async () => (await navigator.storage.getDirectory()).getDirectoryHandle("cubelets", { create: true }))()
  .catch(() => null);

// The pack's file and how long it may grow. Another tab with the same cube open holds the file, and this one goes
// without.
async function openPack(name) {
  const dir = await packs;
  const all = [];
  for await (const [other, handle] of dir.entries()) {
    const file = await handle.getFile().catch(() => null);
    if (file) all.push({ name: other, size: file.size, used: file.lastModified });
  }
  all.sort((a, b) => a.used - b.used);
  const current = all.find((p) => p.name === name)?.size ?? 0;
  let others = all.reduce((sum, p) => sum + p.size, 0) - current;
  for (const p of all) {
    if (others + current + RESERVE <= CAP) break;
    if (p.name !== name) await dir.removeEntry(p.name).then(() => (others -= p.size), () => {});
  }
  const file = await dir.getFileHandle(name, { create: true });
  return [await file.createSyncAccessHandle(), CAP - others];
}

// Every pack but the one open, which the client has emptied.
async function clearPacks(open) {
  const dir = await packs;
  if (!dir) return;
  const names = [];
  for await (const name of dir.keys()) if (name !== open) names.push(name);
  for (const name of names) await dir.removeEntry(name).catch(() => {});
}

// Only the latest load is read, so a slow download can't replace a file picked after it started.
let latest;

async function load({ load: id, source, hdu }) {
  latest?.abort();
  const current = (latest = new AbortController());
  try {
    if (typeof source === "string" && new URL(source).pathname.endsWith(".cubelets")) {
      return await loadBaked(id, source, current.signal);
    }
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

// A baked cube: the index at the start of its file, then its pack in OPFS, opened before the viewer is told so that
// it asks for cubelets with the cache in place.
let baked;
async function loadBaked(id, url, signal) {
  const start = performance.now();
  baked = { load: id, url };
  // Most of an index is the histogram, so 256 kB takes in all of one, and the rest is read if it doesn't.
  let index = await range(url, 0, 1 << 18, signal);
  const size = new DataView(index.buffer, index.byteOffset).getUint32(0, true);
  // baked.rs's MAX_INDEX, so that a file that isn't one can't have this allocate gigabytes.
  if (size > 4 << 20) throw new Error("not a baked cube");
  if (size > index.length) {
    const whole = new Uint8Array(size);
    whole.set(index);
    whole.set(await range(url, index.length, size, signal), index.length);
    index = whole;
  }
  if (signal.aborted) return;
  const [pack, info, levels, approximate] = baked_open(id, index, url, baked.whole?.length ?? baked.size);
  const [handle, room] = await openPack(pack).catch((e) => {
    console.warn(`Cubelets aren't kept in OPFS: ${e.message ?? e}`);
    return [null, 0];
  });
  baked_attach(id, handle, room);
  if (signal.aborted) return;
  postMessage({ load: id, info, levels, approximate, seconds: (performance.now() - start) / 1000 }, [info]);
}

// The cubelets the baked cube's source asks for, a few range requests at a time, each made only if it's still wanted.
const READS = 16;
let reading = 0;
const reads = [];
function readBaked() {
  const [load, ...ranges] = baked_fetches();
  // Those for a cube that's on its way out are dropped.
  if (load === baked?.load) {
    for (let i = 0; i < ranges.length; i += 3) reads.push({ load, url: baked.url, token: ranges[i], start: ranges[i + 1], end: ranges[i + 2] });
  }
  while (reading < READS && reads.length) {
    const { load, url, token, start, end } = reads.shift();
    if (!baked_wanted(load, token)) continue;
    reading++;
    retried(() => range(url, start, end))
      .then((bytes) => baked_fetched(load, token, bytes), (e) => {
        console.error(`Reading ${url}: ${e.message ?? e}`);
        baked_failed(load, token);
      })
      .finally(() => {
        reading--;
        readBaked();
      });
  }
}

// Three tries, half a second and then a second apart.
async function retried(read) {
  for (let tries = 1; ; tries++) {
    try {
      return await read();
    } catch (e) {
      if (tries === 3) throw e;
      await new Promise((resolve) => setTimeout(resolve, 500 * tries));
    }
  }
}

// Bytes `start` to `end` of `url`. OPFS is the cache, so the browser's isn't asked. A server that ignores ranges, as
// python3 -m http.server does, sends the whole file, which is kept and read from after.
async function range(url, start, end, signal) {
  if (baked?.url === url && baked.whole) return baked.whole.subarray(start, end);
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` }, cache: "no-store", signal });
  // The file's size, for checking its index, where the server lets the page see it.
  const total = /\/(\d+)$/.exec(res.headers.get("Content-Range") ?? "")?.[1];
  if (baked?.url === url && total) baked.size = Number(total);
  if (res.status === 200) {
    const whole = new Uint8Array(await res.arrayBuffer());
    if (baked?.url === url) baked.whole = whole;
    return whole.subarray(start, end);
  }
  if (res.status !== 206) throw new Error(`${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
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
