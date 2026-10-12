'use strict';
/*
 * server.js — tiny web server for the trainer. No npm packages needed.
 *
 *   node server.js        then open http://localhost:3000
 *
 * Routes
 *   GET  /                          the UI (public/index.html)
 *   POST /api/upload                upload the CSV (raw body) -> data/dataset.csv
 *   GET  /api/dataset               info about the uploaded CSV
 *   POST /api/train                 start training in a worker thread
 *   POST /api/stop                  stop training
 *   GET  /api/events                live training messages (Server-Sent Events)
 *   GET  /api/runs                  list of runs in ./runs
 *   GET  /api/runs/<run>/files      files of one run
 *   GET  /runs/<run>/<file>         read a log file
 *   POST /api/predict               run a sentence through a saved checkpoint
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');
const { readDataset } = require('./dataset');
const text = require('./text');
const { Network } = require('./nn');
const { DEFAULTS } = require('./trainer');

const PORT = Number(process.env.PORT) || 3000;
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, 'data');
const RUNS_DIR = path.join(ROOT, 'runs');
const CSV_PATH = path.join(DATA_DIR, 'dataset.csv');
const INFO_PATH = path.join(DATA_DIR, 'dataset.info.json');
fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(RUNS_DIR, { recursive: true });

// ---------------- training job + live events ----------------
let job = { running: false, messages: [], worker: null, run: null };
const clients = new Set();

function push(msg) {
  job.messages.push(msg);
  if (msg.type === 'start') job.run = msg.run;
  if (msg.type === 'done' || msg.type === 'error') job.running = false;
  const line = `data: ${JSON.stringify(msg)}\n\n`;
  for (const res of clients) res.write(line);
}

setInterval(() => { for (const res of clients) res.write(': ping\n\n'); }, 20000).unref();

// ---------------- helpers ----------------
function sendJSON(res, status, obj) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(obj));
}

function readBody(req, limit = 1e6) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', c => {
      size += c.length;
      if (size > limit) { reject(new Error('Request body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// Keep a path inside a base folder (blocks "../" tricks).
function safeJoin(base, rel) {
  const full = path.resolve(base, '.' + path.sep + rel);
  return full === base || full.startsWith(base + path.sep) ? full : null;
}

function inspectDataset(fileName) {
  const ds = readDataset(fs.readFileSync(CSV_PATH, 'utf8'));
  const pos = ds.docs.filter(d => d.label === 1).length;
  const words = ds.docs.reduce((s, d) => s + d.text.split(/\s+/).length, 0);
  const info = {
    fileName,
    bytes: fs.statSync(CSV_PATH).size,
    uploadedAt: new Date().toISOString(),
    rows: ds.docs.length,
    skipped: ds.skipped,
    columns: ds.header,
    textColumn: ds.textColumn,
    labelColumn: ds.labelColumn,
    positive: pos,
    negative: ds.docs.length - pos,
    avgWords: ds.docs.length ? Math.round(words / ds.docs.length) : 0,
    samples: ds.docs.slice(0, 3).map(d => ({
      label: d.label ? 'positive' : 'negative',
      text: d.text.replace(/<br\s*\/?>/g, ' ').slice(0, 300),
    })),
  };
  fs.writeFileSync(INFO_PATH, JSON.stringify(info, null, 2));
  return info;
}

function walk(dir, base = dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(full, base));
    else out.push({ path: path.relative(base, full).split(path.sep).join('/'), bytes: fs.statSync(full).size });
  }
  return out;
}

const naturalSort = (a, b) => a.localeCompare(b, undefined, { numeric: true });

function listRuns() {
  return fs.readdirSync(RUNS_DIR, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => {
      const dir = path.join(RUNS_DIR, e.name);
      const epochs = fs.readdirSync(dir).filter(f => /^epoch_\d+$/.test(f)).sort(naturalSort);
      let summary = null;
      try { summary = JSON.parse(fs.readFileSync(path.join(dir, '99_run_summary.json'), 'utf8')); } catch { /* still running or stopped */ }
      return {
        name: e.name,
        epochs,
        finished: !!summary,
        running: job.running && job.run === e.name,
        bestTestAccuracy: summary ? summary.best_test_accuracy : null,
      };
    })
    .sort((a, b) => naturalSort(b.name, a.name));
}

// ---------------- prediction ----------------
const modelCache = new Map();

function loadModel(run, epoch) {
  const rel = epoch === 'final' ? 'final_model.json' : `epoch_${Number(epoch)}/checkpoint.json`;
  const file = safeJoin(RUNS_DIR, `${run}/${rel}`);
  if (!file || !fs.existsSync(file)) throw new Error(`No checkpoint at runs/${run}/${rel}`);
  const key = `${file}:${fs.statSync(file).mtimeMs}`;
  if (!modelCache.has(key)) {
    const cp = JSON.parse(fs.readFileSync(file, 'utf8'));
    modelCache.clear();
    modelCache.set(key, {
      net: Network.fromCheckpoint(cp, cp.vocabulary),
      vocab: text.vocabFromLists(cp.vocabulary, cp.idf),
      epoch: cp.epoch,
    });
  }
  return modelCache.get(key);
}

function predict(model, sentence) {
  const { net, vocab } = model;
  const det = text.tokenize(sentence, true);
  const vec = text.vectorize(det.tokens, vocab, true);
  const score = net.forward(vec.vector);
  const hidden = net.layers.slice(1, -1).map(L => ({
    layer: L.name,
    outputs: L.neurons.map(n => Math.round(n.output * 1e4) / 1e4),
  }));
  // Occlusion: how much does the score change if this token is removed?
  const rows = vec.table.map(t => {
    const without = net.forward(text.vectorize(det.tokens.filter(x => x !== t.token), vocab));
    return {
      token: t.token,
      input_neuron: t.input_neuron,
      tf: t.tf,
      idf: +t.idf.toFixed(4),
      normalized: +t.normalized.toFixed(4),
      score_without: +without.toFixed(4),
      impact: +(score - without).toFixed(4),
    };
  }).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  return {
    epoch: model.epoch,
    score: +score.toFixed(4),
    label: score >= 0.5 ? 'positive' : 'negative',
    cleaned: det.cleaned,
    tokens: det.tokens,
    stopwordsRemoved: det.stopwordsRemoved,
    unknown: vec.unknown,
    rows,
    hidden,
  };
}

// ---------------- config from the UI ----------------
function buildConfig(body) {
  const num = (v, d, min, max) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : d;
  };
  const hidden = String(body.hidden ?? DEFAULTS.hidden.join(','))
    .split(/[\s,]+/).map(Number).filter(n => Number.isInteger(n) && n > 0).slice(0, 5).map(n => Math.min(n, 256));
  return {
    __trainer: true,
    csvPath: CSV_PATH,
    csvName: JSON.parse(fs.readFileSync(INFO_PATH, 'utf8')).fileName,
    runsDir: RUNS_DIR,
    maxRows: Math.round(num(body.maxRows, 0, 0, 1e7)),
    testSplit: num(body.testSplit, DEFAULTS.testSplit, 0.05, 0.5),
    maxVocab: Math.round(num(body.maxVocab, DEFAULTS.maxVocab, 50, 50000)),
    minDf: Math.round(num(body.minDf, DEFAULTS.minDf, 1, 1000)),
    hidden,
    epochs: Math.round(num(body.epochs, DEFAULTS.epochs, 1, 100)),
    learningRate: num(body.learningRate, DEFAULTS.learningRate, 0.0001, 10),
    seed: Math.round(num(body.seed, DEFAULTS.seed, 0, 2 ** 31)),
    curveEvery: Math.round(num(body.curveEvery, DEFAULTS.curveEvery, 10, 1e6)),
    pauseMs: Math.round(num(body.pauseMs, DEFAULTS.pauseMs, 0, 60000)),
  };
}

// ---------------- routes ----------------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const p = decodeURIComponent(url.pathname);
  try {
    if (req.method === 'GET' && (p === '/' || p === '/index.html')) {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      return fs.createReadStream(path.join(ROOT, 'public', 'index.html')).pipe(res);
    }

    if (req.method === 'POST' && p === '/api/upload') {
      if (job.running) return sendJSON(res, 409, { error: 'Training is running. Stop it before uploading a new file.' });
      const fileName = String(req.headers['x-filename'] || 'dataset.csv').replace(/[^\w.\- ]/g, '_');
      const tmp = CSV_PATH + '.uploading';
      await new Promise((resolve, reject) => {
        const out = fs.createWriteStream(tmp);
        req.pipe(out);
        out.on('finish', resolve);
        out.on('error', reject);
        req.on('error', reject);
      });
      fs.renameSync(tmp, CSV_PATH);
      return sendJSON(res, 200, inspectDataset(fileName));
    }

    if (req.method === 'GET' && p === '/api/dataset') {
      if (!fs.existsSync(INFO_PATH) || !fs.existsSync(CSV_PATH)) return sendJSON(res, 200, null);
      return sendJSON(res, 200, JSON.parse(fs.readFileSync(INFO_PATH, 'utf8')));
    }

    if (req.method === 'POST' && p === '/api/train') {
      if (job.running) return sendJSON(res, 409, { error: 'Training is already running.' });
      if (!fs.existsSync(CSV_PATH)) return sendJSON(res, 400, { error: 'Upload a CSV first.' });
      const cfg = buildConfig(JSON.parse((await readBody(req)) || '{}'));
      job = { running: true, messages: [], worker: null, run: null };
      for (const c of clients) c.write(`data: ${JSON.stringify({ type: 'reset' })}\n\n`);
      const worker = new Worker(path.join(ROOT, 'trainer.js'), { workerData: cfg });
      job.worker = worker;
      const thisJob = job;
      worker.on('message', m => { if (job === thisJob) push(m); });
      worker.on('error', e => { if (job === thisJob) push({ type: 'error', message: e.message }); });
      worker.on('exit', () => {
        if (job === thisJob && job.running) { job.running = false; push({ type: 'stopped' }); }
      });
      return sendJSON(res, 200, { ok: true, config: { ...cfg, csvPath: undefined, runsDir: undefined, __trainer: undefined } });
    }

    if (req.method === 'POST' && p === '/api/stop') {
      if (job.worker && job.running) {
        await job.worker.terminate();
        job.running = false;
        push({ type: 'stopped' });
      }
      return sendJSON(res, 200, { ok: true });
    }

    if (req.method === 'GET' && p === '/api/events') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
      res.write(`data: ${JSON.stringify({ type: 'hello', running: job.running })}\n\n`);
      for (const m of job.messages) res.write(`data: ${JSON.stringify(m)}\n\n`);
      clients.add(res);
      req.on('close', () => clients.delete(res));
      return;
    }

    if (req.method === 'GET' && p === '/api/runs') return sendJSON(res, 200, listRuns());

    let m;
    if (req.method === 'GET' && (m = p.match(/^\/api\/runs\/([^/]+)\/files$/))) {
      const dir = safeJoin(RUNS_DIR, m[1]);
      if (!dir || !fs.existsSync(dir)) return sendJSON(res, 404, { error: 'Run not found' });
      return sendJSON(res, 200, walk(dir).sort((a, b) => naturalSort(a.path, b.path)));
    }

    if (req.method === 'GET' && p.startsWith('/runs/')) {
      const file = safeJoin(RUNS_DIR, p.slice('/runs/'.length));
      if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404); return res.end('Not found'); }
      const type = file.endsWith('.json') ? 'application/json' : 'text/plain';
      res.writeHead(200, { 'Content-Type': `${type}; charset=utf-8` });
      return fs.createReadStream(file).pipe(res);
    }

    if (req.method === 'POST' && p === '/api/predict') {
      const body = JSON.parse((await readBody(req)) || '{}');
      if (!body.run) return sendJSON(res, 400, { error: 'Choose a run.' });
      const model = loadModel(body.run, body.epoch ?? 'final');
      return sendJSON(res, 200, predict(model, String(body.text || '')));
    }

    sendJSON(res, 404, { error: 'Not found' });
  } catch (e) {
    sendJSON(res, 500, { error: e.message });
  }
});

server.listen(PORT, () => {
  console.log(`Sentiment trainer running at http://localhost:${PORT}`);
  console.log(`Logs are written to ${RUNS_DIR}`);
});
