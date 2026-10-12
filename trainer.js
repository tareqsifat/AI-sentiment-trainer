'use strict';
/*
 * trainer.js — runs one training session and writes everything it does to
 *
 *   runs/<date_time>/
 *     README.md                  what every file means
 *     00_config.json             settings used
 *     01_dataset_stats.json      the data: sizes, label balance, lengths
 *     02_vocabulary.json         every input neuron's word, its document frequency and IDF
 *     03_tfidf_examples.json     3 reviews walked through every preprocessing step
 *     04_architecture.json       layers, neurons, links, parameter count
 *     epoch_0/                   the untrained (random) network
 *     epoch_1/ ... epoch_N/      the network after each pass over the training data
 *     final_model.json           the trained weights, ready to load
 *     99_run_summary.json        one row per epoch
 *
 * Runs three ways:
 *   - from server.js, inside a worker thread (the web UI)
 *   - from the command line:  node trainer.js path/to/IMDB.csv [epochs]
 *   - required as a module:   require('./trainer').runTraining(config, onMessage)
 */

const fs = require('fs');
const path = require('path');
const { Network, mulberry32 } = require('./nn');
const text = require('./text');
const { readDataset } = require('./dataset');

const DEFAULTS = {
  maxRows: 0,          // 0 = use every row
  testSplit: 0.2,      // share of rows held back for testing
  maxVocab: 5000,      // number of input neurons
  minDf: 5,            // a word must appear in at least this many training reviews
  hidden: [16, 8],     // hidden layer sizes
  epochs: 5,
  learningRate: 0.3,
  seed: 42,
  curveEvery: 1000,    // log running loss/accuracy every N training reviews
  pauseMs: 400,        // sleep between the stages of an epoch (train -> evaluate/write -> next epoch) to go easy on the PC
};

// Blocking sleep. Safe here because training runs in its own worker thread.
const sleepMs = ms => { if (ms > 0) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); };

const r6 = x => Math.round(x * 1e6) / 1e6;
const r4 = x => Math.round(x * 1e4) / 1e4;
const pct = x => Math.round(x * 10000) / 100;

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function stamp() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

function bce(p, y) {
  p = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
  return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
}

const snippet = (s, n = 220) => {
  const t = String(s).replace(/<br\s*\/?>/g, ' ').replace(/\s+/g, ' ').trim();
  return t.length > n ? t.slice(0, n) + '…' : t;
};

// Readable-but-compact JSON for big checkpoints: one top-level key per line, long
// arrays wrapped into rows of `perRow` values. Still plain, valid JSON.
function formatCheckpoint(obj, perRow = 10) {
  const fmtArray = a => {
    if (a.length <= perRow) return JSON.stringify(a);
    const rows = [];
    for (let i = 0; i < a.length; i += perRow) rows.push('    ' + a.slice(i, i + perRow).map(v => JSON.stringify(v)).join(', '));
    return `[\n${rows.join(',\n')}\n  ]`;
  };
  const lines = Object.entries(obj).map(([k, v]) => `  ${JSON.stringify(k)}: ${Array.isArray(v) ? fmtArray(v) : JSON.stringify(v)}`);
  return `{\n${lines.join(',\n')}\n}\n`;
}

// final_model.json layout: layers/neurons indented, word->weight maps and long
// arrays wrapped several to a line so the file stays navigable.
function formatModel(m, perRow = 5) {
  const rows = (items, pad) => {
    const out = [];
    for (let i = 0; i < items.length; i += perRow) out.push(pad + items.slice(i, i + perRow).join(', '));
    return out.join(',\n');
  };
  const weights = w => {
    if (Array.isArray(w)) return w.length <= 16 ? JSON.stringify(w) : `[\n${rows(w.map(v => JSON.stringify(v)), '          ')}\n        ]`;
    const pairs = Object.entries(w).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`);
    return `{\n${rows(pairs, '          ')}\n        }`;
  };
  const layers = m.layers.map(L => {
    const neurons = L.neurons.map(n =>
      `      {\n        "neuron": ${n.neuron},\n        "bias": ${JSON.stringify(n.bias)},\n        "weights": ${weights(n.weights)}\n      }`);
    return `    {\n      "layer": ${JSON.stringify(L.layer)},\n      "inputs_from": ${JSON.stringify(L.inputs_from)},\n      "neurons": [\n${neurons.join(',\n')}\n      ]\n    }`;
  });
  const arr = a => `[\n${rows(a.map(v => JSON.stringify(v)), '    ')}\n  ]`;
  return `{\n  "_explain": ${JSON.stringify(m._explain)},\n  "epoch": ${m.epoch},\n  "sizes": ${JSON.stringify(m.sizes)},\n  "layers": [\n${layers.join(',\n')}\n  ],\n  "vocabulary": ${arr(m.vocabulary)},\n  "idf": ${arr(m.idf)}\n}\n`;
}

function runTraining(userConfig, emit = () => {}) {
  const C = { ...DEFAULTS, ...userConfig };
  const started = Date.now();
  const log = t => emit({ type: 'log', text: t });

  const runName = stamp();
  const runDir = path.join(C.runsDir, runName);
  fs.mkdirSync(runDir, { recursive: true });
  emit({ type: 'start', run: runName });

  const writeText = (rel, content) => {
    const full = path.join(runDir, rel);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, content);
    emit({ type: 'file', run: runName, path: rel, bytes: Buffer.byteLength(content) });
  };
  const write = (rel, obj, pretty = true) => writeText(rel, pretty ? JSON.stringify(obj, null, 2) : JSON.stringify(obj));
  const writeCheckpoint = (rel, obj) => writeText(rel, formatCheckpoint(obj));

  writeText('README.md', RUN_README);

  // ------------------------------------------------------------------
  // 1. Read and split the data
  // ------------------------------------------------------------------
  log(`Reading ${C.csvName || path.basename(C.csvPath)} ...`);
  const ds = readDataset(fs.readFileSync(C.csvPath, 'utf8'));
  log(`Parsed ${ds.docs.length} labelled reviews (text column "${ds.textColumn}", label column "${ds.labelColumn}", ${ds.skipped} rows skipped).`);
  if (ds.docs.length < 20) throw new Error('Need at least 20 labelled rows to train.');

  const rng = mulberry32(C.seed);
  let docs = ds.docs.map((d, i) => ({ row: i + 1, text: d.text, label: d.label }));
  shuffle(docs, rng);
  if (C.maxRows > 0 && docs.length > C.maxRows) docs = docs.slice(0, C.maxRows);
  const testCount = Math.max(1, Math.round(docs.length * C.testSplit));
  const testDocs = docs.slice(0, testCount);
  const trainDocs = docs.slice(testCount);
  log(`Using ${docs.length} reviews: ${trainDocs.length} for training, ${testDocs.length} for testing.`);

  write('00_config.json', {
    _explain: 'The settings this run used. Same settings + same seed + same CSV = the exact same run and the exact same numbers in every file.',
    csv_file: C.csvName || path.basename(C.csvPath),
    max_rows: C.maxRows || 'all',
    test_split: C.testSplit,
    vocabulary_size_max: C.maxVocab,
    min_document_frequency: C.minDf,
    hidden_layers: C.hidden,
    epochs: C.epochs,
    learning_rate: C.learningRate,
    seed: C.seed,
    curve_every_n_reviews: C.curveEvery,
    pause_ms_between_stages: C.pauseMs,
    started_at: new Date(started).toISOString(),
  });

  // ------------------------------------------------------------------
  // 2. Vocabulary: count in how many TRAINING reviews each token appears
  // ------------------------------------------------------------------
  log('Tokenizing training reviews and counting document frequency ...');
  const dfMap = new Map();
  for (const d of trainDocs) {
    const toks = text.tokenize(d.text);
    for (const t of toks) {
      let s = dfMap.get(t);
      if (!s) { s = { df: 0, dfPos: 0, dfNeg: 0, count: 0 }; dfMap.set(t, s); }
      s.count++;
    }
    for (const t of new Set(toks)) {
      const s = dfMap.get(t);
      s.df++;
      if (d.label === 1) s.dfPos++; else s.dfNeg++;
    }
  }
  const vocab = text.buildVocabulary(dfMap, trainDocs.length, C);
  dfMap.clear();
  log(`Vocabulary: kept ${vocab.tokens.length} of ${vocab.distinctTokens} distinct tokens (min document frequency ${C.minDf}, max ${C.maxVocab}).`);

  write('02_vocabulary.json', {
    _explain: [
      'One entry per input neuron. input_neuron = its index in the input layer.',
      'df = in how many training reviews the token appears (counted once per review).',
      'df_positive / df_negative = the same count split by label. positive_share = df_positive / df.',
      'A token with positive_share far from the overall positive rate is a sentiment clue the network can pick up.',
      'idf = ln((1 + N) / (1 + df)) + 1, with N = number of training reviews. Rare tokens get a larger idf.',
      'Counted on training reviews only: the test set must stay unseen, otherwise the test score is too optimistic (data leakage).',
    ],
    training_reviews: trainDocs.length,
    distinct_tokens_seen: vocab.distinctTokens,
    tokens_dropped_as_too_rare: vocab.droppedRare,
    tokens_kept: vocab.tokens.length,
    tokens: vocab.entries.map((e, i) => ({
      input_neuron: i,
      token: e.token,
      df: e.df,
      df_positive: e.dfPos,
      df_negative: e.dfNeg,
      positive_share: r4(e.dfPos / e.df),
      total_occurrences: e.count,
      idf: r6(vocab.idf[i]),
    })),
  });

  // ------------------------------------------------------------------
  // 3. TF-IDF vector for every review
  // ------------------------------------------------------------------
  log('Turning every review into a TF-IDF vector ...');
  let knownOcc = 0, allOcc = 0, activeSum = 0;
  for (const d of docs) {
    const toks = text.tokenize(d.text);
    d.tokenCount = toks.length;
    d.wordCount = d.text.split(/\s+/).length;
    d.x = text.vectorize(toks, vocab);
    allOcc += toks.length;
    knownOcc += d.x.known;
    activeSum += d.x.idx.length;
  }

  const countLabels = arr => ({ positive: arr.filter(d => d.label === 1).length, negative: arr.filter(d => d.label === 0).length });
  const lengths = docs.map(d => d.wordCount).sort((a, b) => a - b);
  write('01_dataset_stats.json', {
    _explain: 'What the network is trained and tested on. Check that positive/negative are roughly balanced: if 90% were positive, a model that always says "positive" would already score 90%.',
    csv_file: C.csvName || path.basename(C.csvPath),
    text_column: ds.textColumn,
    label_column: ds.labelColumn,
    labelled_rows_in_file: ds.docs.length,
    rows_skipped: ds.skipped,
    rows_used: docs.length,
    labels_all: countLabels(docs),
    train: { reviews: trainDocs.length, labels: countLabels(trainDocs) },
    test: { reviews: testDocs.length, labels: countLabels(testDocs) },
    review_length_in_words: {
      min: lengths[0],
      median: lengths[Math.floor(lengths.length / 2)],
      mean: Math.round(lengths.reduce((s, v) => s + v, 0) / lengths.length),
      max: lengths[lengths.length - 1],
    },
    avg_tokens_per_review_after_cleaning: Math.round(allOcc / docs.length),
    avg_active_input_neurons_per_review: Math.round(activeSum / docs.length),
    share_of_token_occurrences_covered_by_vocabulary: pct(knownOcc / allOcc) + '%',
    note_on_sparsity: `Each review switches on about ${Math.round(activeSum / docs.length)} of ${vocab.tokens.length} input neurons; the rest output 0.`,
    samples: docs.slice(0, 3).map(d => ({ row: d.row, label: d.label ? 'positive' : 'negative', text: snippet(d.text, 500) })),
  });

  // Walk 3 short training reviews through every step.
  const exampleDocs = trainDocs.filter(d => d.tokenCount >= 20 && d.tokenCount <= 70).slice(0, 3);
  while (exampleDocs.length < 3 && exampleDocs.length < trainDocs.length) exampleDocs.push(trainDocs[exampleDocs.length]);
  write('03_tfidf_examples.json', {
    _explain: [
      'How a review becomes the numbers the input layer sees.',
      'step1 raw text -> step2 cleaned (lowercase, html and punctuation removed) -> step3 words',
      '-> step4 stopwords dropped and negation merged ("not good" -> "not_good") -> step5 tokens',
      '-> step6 tf = how many times the token occurs in this review',
      '-> step7 tf_x_idf = tf × idf (frequent here AND rare elsewhere = important)',
      '-> step8 normalized = tf_x_idf / l2_norm, so every review vector has length 1 regardless of review length.',
      'The normalized value is exactly what input neuron #input_neuron outputs. Every other input neuron outputs 0.',
    ],
    examples: exampleDocs.map(d => {
      const det = text.tokenize(d.text, true);
      const vec = text.vectorize(det.tokens, vocab, true);
      return {
        row: d.row,
        label: d.label ? 'positive' : 'negative',
        step1_raw_text: snippet(d.text, 1500),
        step2_cleaned: det.cleaned.slice(0, 1500),
        step3_words: det.words,
        step4_stopwords_removed: det.stopwordsRemoved,
        step4_negations_merged: det.negations,
        step5_tokens: det.tokens,
        step6_tokens_not_in_vocabulary: vec.unknown,
        l2_norm: r6(vec.norm),
        active_input_neurons: vec.table.length,
        silent_input_neurons: vocab.tokens.length - vec.table.length,
        step6_to_8_tfidf_table: vec.table
          .sort((a, b) => b.normalized - a.normalized)
          .map(t => ({ ...t, idf: r6(t.idf), tf_x_idf: r6(t.tf_x_idf), normalized: r6(t.normalized) })),
      };
    }),
  });

  // ------------------------------------------------------------------
  // 4. Build the network
  // ------------------------------------------------------------------
  const sizes = [vocab.tokens.length, ...C.hidden, 1];
  const net = new Network(sizes, { seed: C.seed, inputLabels: vocab.tokens });
  const groups = [];
  for (let l = 0; l < net.layers.length - 1; l++) {
    groups.push({ from: net.layers[l].name, to: net.layers[l + 1].name, links: sizes[l] * sizes[l + 1] });
  }
  const totalBiases = sizes.slice(1).reduce((s, v) => s + v, 0);
  log(`Network: ${sizes.join(' → ')} = ${net.links.length} links + ${totalBiases} biases = ${net.links.length + totalBiases} learnable numbers.`);

  write('04_architecture.json', {
    _explain: 'The shape of the network. The shape never changes during training; only the numbers (weights and biases) do.',
    layer_sizes: sizes,
    layers: net.layers.map(L => ({
      name: L.name,
      neurons: L.neurons.length,
      role: L.index === 0 ? 'one neuron per vocabulary token; outputs that token\'s TF-IDF value (0 if absent)'
        : L.name === 'output' ? 'one neuron; output = probability the review is positive (>= 0.5 means positive)'
          : 'learns combinations of words; nobody tells it what to detect',
      first_neurons: L.neurons.slice(0, 5).map(n => n.label),
    })),
    link_groups: groups,
    total_links: net.links.length,
    total_biases: totalBiases,
    total_learnable_parameters: net.links.length + totalBiases,
    activation: 'sigmoid(z) = 1 / (1 + e^-z) in every non-input neuron',
    loss: 'binary cross-entropy: -(y·ln(p) + (1-y)·ln(1-p))',
    optimizer: 'plain stochastic gradient descent, one review at a time (batch size 1)',
    weight_init: `uniform random in [-0.5, 0.5] from seed ${C.seed}; biases start at 0`,
    links_in_creation_order: 'link id k: for each layer, for each sending neuron, for each receiving neuron. checkpoint.json weights[k] belongs to link k.',
  });

  // ------------------------------------------------------------------
  // Helpers that describe the network's current state
  // ------------------------------------------------------------------
  const nodeName = n => (n.layerIndex === 0 ? n.label : n.name);
  const linkName = k => `${k.from.layerIndex === 0 ? `input "${k.from.label}"` : k.from.name} → ${k.to.name}`;

  function writeLayerFiles(dir) {
    for (let l = 1; l < net.layers.length; l++) {
      const layer = net.layers[l];
      const k = Math.min(10, Math.ceil(sizes[l - 1] / 2));
      write(`${dir}/layer_${l}_${layer.name}.json`, {
        _explain: [
          `Every learnable number of layer "${layer.name}" at this point.`,
          'bias: added to the neuron\'s sum before sigmoid.',
          'all_incoming_weights: one entry per link into this neuron, keyed by the sending neuron (a word for the first hidden layer).',
          'Positive weight = that input pushes this neuron to fire; negative = it holds the neuron back.',
          'Compare the same neuron across epoch folders to watch it learn.',
        ],
        layer: layer.name,
        neurons: layer.neurons.length,
        incoming_links_per_neuron: sizes[l - 1],
        neuron_details: layer.neurons.map(n => {
          const inc = n.incoming.map(link => [nodeName(link.from), link.weight]);
          const sorted = [...inc].sort((a, b) => b[1] - a[1]);
          return {
            name: n.name,
            bias: r6(n.bias),
            strongest_positive_links: sorted.slice(0, k).map(([from, w]) => ({ from, weight: r6(w) })),
            strongest_negative_links: sorted.slice(-k).reverse().map(([from, w]) => ({ from, weight: r6(w) })),
            all_incoming_weights: Object.fromEntries(inc.map(([f, w]) => [f, r6(w)])),
          };
        }),
      });
    }
  }

  function evaluate(set) {
    let tp = 0, tn = 0, fp = 0, fn = 0, loss = 0;
    const hidden = net.layers.slice(1, -1);
    const act = hidden.map(L => L.neurons.map(() => [0, 0])); // [sum on positive, sum on negative]
    const nPos = set.filter(d => d.label === 1).length || 1;
    const nNeg = set.length - nPos || 1;
    const samples = [], wrong = [];
    for (const d of set) {
      const p = net.forward(d.x);
      loss += bce(p, d.label);
      const pred = p >= 0.5 ? 1 : 0;
      if (pred === 1 && d.label === 1) tp++;
      else if (pred === 0 && d.label === 0) tn++;
      else if (pred === 1) fp++;
      else fn++;
      hidden.forEach((L, li) => L.neurons.forEach((n, ni) => { act[li][ni][d.label === 1 ? 0 : 1] += n.output; }));
      const rec = { row: d.row, actual: d.label ? 'positive' : 'negative', predicted: pred ? 'positive' : 'negative', score: r4(p), correct: pred === d.label, text: snippet(d.text) };
      if (samples.length < 20) samples.push(rec);
      if (pred !== d.label) wrong.push(rec);
    }
    wrong.sort((a, b) => Math.abs(b.score - 0.5) - Math.abs(a.score - 0.5));
    const precision = tp / (tp + fp || 1);
    const recall = tp / (tp + fn || 1);
    return {
      metrics: {
        reviews: set.length,
        loss: r6(loss / set.length),
        accuracy: r4((tp + tn) / set.length),
        precision: r4(precision),
        recall: r4(recall),
        f1: r4((2 * precision * recall) / (precision + recall || 1)),
        confusion_matrix: { true_positive: tp, true_negative: tn, false_positive: fp, false_negative: fn },
      },
      samples,
      mostConfidentMistakes: wrong.slice(0, 15),
      activations: hidden.map((L, li) => L.neurons.map((n, ni) => ({ name: n.name, pos: act[li][ni][0] / nPos, neg: act[li][ni][1] / nNeg }))),
    };
  }

  function neuronReport(ev) {
    const layers = net.layers.slice(1, -1);
    return {
      _explain: [
        'An interpretation of each hidden neuron, measured on the test reviews.',
        'mean_activation_on_positive / _negative: the neuron\'s average output (0..1) on positive vs negative reviews.',
        'A neuron with a big difference has become a sentiment detector. Nobody programmed that: backpropagation discovered it.',
        'excites_most / inhibits_most: the inputs with the largest positive / negative weights into this neuron.',
        'weight_to_next_layer: how the next layer uses this neuron (for the last hidden layer this is its vote to the output).',
        'In epoch_0 everything is random, so differences are near zero. Watch them grow.',
      ],
      layers: layers.map((L, li) => ({
        layer: L.name,
        neurons: L.neurons.map((n, ni) => {
          const a = ev.activations[li][ni];
          const diff = a.pos - a.neg;
          const sorted = [...n.incoming].sort((x, y) => y.weight - x.weight);
          return {
            name: n.name,
            mean_activation_on_positive: r4(a.pos),
            mean_activation_on_negative: r4(a.neg),
            difference: r4(diff),
            reading: diff > 0.05 ? 'fires more on POSITIVE reviews' : diff < -0.05 ? 'fires more on NEGATIVE reviews' : 'no clear preference',
            excites_most: sorted.slice(0, 10).map(k => `${nodeName(k.from)} (${r4(k.weight)})`),
            inhibits_most: sorted.slice(-10).reverse().map(k => `${nodeName(k.from)} (${r4(k.weight)})`),
            weight_to_next_layer: n.outgoing.map(k => ({ to: k.to.name, weight: r4(k.weight) })),
          };
        }),
      })),
    };
  }

  function wordInfluence() {
    const empty = net.forward({ idx: [], val: [] });
    const scores = vocab.tokens.map((t, i) => [t, net.forward({ idx: [i], val: [1] })]);
    scores.sort((a, b) => b[1] - a[1]);
    const fmt = ([token, s]) => ({ token, score: r4(s) });
    return {
      _explain: [
        'Feeds each vocabulary token ALONE into the network (that one input neuron = 1, all others 0) and records the output.',
        'score near 1 = the network reads this word as positive, near 0 = negative.',
        'empty_review_score = output when every input is 0, i.e. what the biases alone produce.',
        'In epoch_0 all scores sit near the same value. After training they spread out.',
      ],
      empty_review_score: r4(empty),
      most_positive: scores.slice(0, 50).map(fmt),
      most_negative: scores.slice(-50).reverse().map(fmt),
      all_tokens: Object.fromEntries(scores.map(([t, s]) => [t, r4(s)])),
    };
  }

  function weightChanges(before, biasBefore) {
    const per = new Map();
    const all = [];
    net.links.forEach((k, i) => {
      const key = `${net.layers[k.from.layerIndex].name} → ${net.layers[k.to.layerIndex].name}`;
      let g = per.get(key);
      if (!g) { g = { links: 0, changed: 0, sumAbs: 0, max: 0, normBefore: 0, normAfter: 0 }; per.set(key, g); }
      const d = k.weight - before[i];
      g.links++;
      if (d !== 0) g.changed++;
      g.sumAbs += Math.abs(d);
      g.max = Math.max(g.max, Math.abs(d));
      g.normBefore += before[i] * before[i];
      g.normAfter += k.weight * k.weight;
      all.push(i);
    });
    all.sort((a, b) => Math.abs(net.links[b].weight - before[b]) - Math.abs(net.links[a].weight - before[a]));
    return {
      _explain: [
        'How much training moved the numbers during this epoch.',
        'mean_abs_change: average |new - old| over all links in the group. Large early, smaller as the network settles.',
        'weight_norm: sqrt(sum of squared weights). Steady growth means the network is becoming more confident.',
        'links_changed: input links of words that never appeared in a training review this epoch cannot change.',
      ],
      per_link_group: [...per].map(([group, g]) => ({
        group,
        links: g.links,
        links_changed: g.changed,
        mean_abs_change: r6(g.sumAbs / g.links),
        max_abs_change: r6(g.max),
        weight_norm_before: r4(Math.sqrt(g.normBefore)),
        weight_norm_after: r4(Math.sqrt(g.normAfter)),
      })),
      biggest_weight_changes: all.slice(0, 30).map(i => {
        const k = net.links[i];
        return { link: linkName(k), before: r6(before[i]), after: r6(k.weight), change: r6(k.weight - before[i]) };
      }),
      bias_changes: net.layers.slice(1).flatMap((L, li) => L.neurons.map((n, ni) => ({
        neuron: n.name, before: r6(biasBefore[li][ni]), after: r6(n.bias), change: r6(n.bias - biasBefore[li][ni]),
      }))),
    };
  }

  // One training step with everything written down.
  function tracedStep(doc, epoch) {
    const lr = C.learningRate;
    const p = net.forward(doc.x);

    const forward = net.layers.slice(1).map(L => ({
      layer: L.name,
      neurons: L.neurons.map(n => {
        const rec = { name: n.name, bias: r6(n.bias), z_sum: r6(n.sum), output_sigmoid_z: r6(n.output) };
        if (L.index === 1) {
          const c = n.incoming.filter(k => k.from.output !== 0).map(k => ({
            token: k.from.label, input_value: r6(k.from.output), weight: r6(k.weight), contribution: r6(k.from.output * k.weight),
          }));
          c.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
          rec.how_z_was_computed = `bias + sum of ${c.length} contributions (input_value × weight) from the active input neurons`;
          rec.top_contributions = c.slice(0, 8);
        } else {
          rec.inputs = n.incoming.map(k => ({ from: k.from.name, value: r6(k.from.output), weight: r6(k.weight), contribution: r6(k.from.output * k.weight) }));
        }
        return rec;
      }),
    }));

    net.backward(doc.label);
    const out = net.outputLayer.neurons[0];
    const backward = net.layers.slice(1).reverse().map(L => ({
      layer: L.name,
      neurons: L.neurons.map(n => {
        if (n === out) return { name: n.name, delta: r6(n.delta), formula: `delta = prediction - target = ${r6(p)} - ${doc.label}` };
        const parts = n.outgoing.map(k => ({ to: k.to.name, weight: r6(k.weight), their_delta: r6(k.to.delta), product: r6(k.weight * k.to.delta) }));
        const blame = n.outgoing.reduce((s, k) => s + k.weight * k.to.delta, 0);
        return {
          name: n.name,
          blame_from_next_layer: r6(blame),
          sigmoid_slope: r6(n.output * (1 - n.output)),
          delta: r6(n.delta),
          formula: 'delta = (sum of weight × their_delta) × output × (1 - output)',
          blame_parts: parts,
        };
      }),
    }));

    // Links to watch while updating.
    let watch;
    const activeTop = neuron => neuron.incoming.filter(k => k.from.output !== 0)
      .sort((a, b) => Math.abs(b.from.output * b.weight) - Math.abs(a.from.output * a.weight)).slice(0, 8);
    if (net.layers.length === 2) watch = activeTop(out);
    else {
      watch = [...out.incoming];
      for (let l = 2; l < net.layers.length - 1; l++) watch.push(...net.layers[l].neurons[0].incoming);
      watch.push(...activeTop(net.layers[1].neurons[0]));
    }
    const wBefore = watch.map(k => k.weight);
    const grads = watch.map(k => k.to.delta * k.from.output);
    const bBefore = net.layers.slice(1).map(L => L.neurons.map(n => n.bias));

    net.update(lr);

    const updates = watch.map((k, i) => ({
      link: linkName(k),
      weight_before: r6(wBefore[i]),
      gradient: r6(grads[i]),
      change: r6(-lr * grads[i]),
      weight_after: r6(k.weight),
    }));
    const biasUpdates = net.layers.slice(1).flatMap((L, li) => L.neurons.slice(0, 3).map((n, ni) => ({
      neuron: n.name, bias_before: r6(bBefore[li][ni]), delta: r6(n.delta), bias_after: r6(n.bias),
    })));

    const pAfter = net.forward(doc.x);
    const det = text.tokenize(doc.text, true);
    return {
      prediction: p,
      trace: {
        _explain: [
          'One complete training step on one real review, written down number by number.',
          'The same review is traced in every epoch, so you can compare how the network treats it over time.',
          '1. forward: input values flow through the links, each neuron computes z = bias + Σ input × weight, then output = sigmoid(z).',
          '2. loss: how wrong the prediction was.',
          '3. backward: each neuron\'s delta (its share of the blame), from the output back to the first hidden layer.',
          '4. update: new_weight = old_weight - learning_rate × delta(receiving neuron) × output(sending neuron). Only a sample of links is shown.',
          '5. after_update: the same review run through the updated network. The prediction moves toward the target.',
        ],
        epoch,
        review: { row: doc.row, label: doc.label ? 'positive (target = 1)' : 'negative (target = 0)', text: snippet(doc.text, 1000), tokens: det.tokens },
        input_layer: {
          active_neurons: doc.x.idx.length,
          values: Array.from(doc.x.idx, (i, k) => ({ input_neuron: i, token: vocab.tokens[i], value: r6(doc.x.val[k]) })).sort((a, b) => b.value - a.value),
        },
        step1_forward: forward,
        step2_loss: {
          prediction: r6(p),
          target: doc.label,
          loss: r6(bce(p, doc.label)),
          formula: '-(target × ln(prediction) + (1 - target) × ln(1 - prediction))',
        },
        step3_backward: backward,
        step4_update: { learning_rate: lr, links: updates, biases: biasUpdates },
        step5_after_update: {
          prediction_before: r6(p),
          prediction_after: r6(pAfter),
          moved_toward_target: Math.abs(pAfter - doc.label) < Math.abs(p - doc.label),
        },
      },
    };
  }

  function writeEpoch(epoch, extra) {
    const dir = `epoch_${epoch}`;
    const ev = evaluate(testDocs);
    write(`${dir}/evaluation.json`, {
      _explain: [
        'How the network performs on the TEST reviews (never used for training).',
        'accuracy = share of correct predictions. precision = of the reviews called positive, share that were positive.',
        'recall = of the positive reviews, share that were found. f1 = balance of precision and recall.',
        'confusion_matrix: true_positive = positive called positive, false_positive = negative called positive, etc.',
        'most_confident_mistakes: wrong answers with the score farthest from 0.5. Reading them shows what bag-of-words cannot handle (sarcasm, "good but...").',
      ],
      epoch,
      test_set: ev.metrics,
      sample_predictions: ev.samples,
      most_confident_mistakes: ev.mostConfidentMistakes,
    });
    writeLayerFiles(dir);
    write(`${dir}/neuron_report.json`, neuronReport(ev));
    write(`${dir}/word_influence.json`, wordInfluence());

    if (extra) {
      write(`${dir}/summary.json`, extra.summaryFile(ev));
      write(`${dir}/training_curve.json`, extra.curve);
      write(`${dir}/weight_changes.json`, extra.changes);
      if (extra.trace) write(`${dir}/sample_trace.json`, extra.trace);
    }

    writeCheckpoint(`${dir}/checkpoint.json`, {
      _explain: 'Everything needed to rebuild this exact network: layer sizes, every weight (weights[k] = link k), every bias, plus the vocabulary and IDF values to turn new text into inputs.',
      epoch,
      sizes,
      vocabulary: vocab.tokens,
      idf: Array.from(vocab.idf, r6),
      weights: net.getWeights().length ? Array.from(net.getWeights()) : [],
      biases: net.getBiases(),
    });
    return ev;
  }

  // ------------------------------------------------------------------
  // 5. Epoch 0: the untrained network
  // ------------------------------------------------------------------
  log('Writing epoch_0 (random weights, before any training) ...');
  const ev0 = writeEpoch(0, null);
  log(`epoch 0  test accuracy ${pct(ev0.metrics.accuracy)}%  (random guessing ≈ 50%)`);
  sleepMs(C.pauseMs);

  // ------------------------------------------------------------------
  // 6. Train
  // ------------------------------------------------------------------
  const traceDoc = trainDocs.find(d => d.tokenCount >= 15 && d.tokenCount <= 60 && d.x.idx.length >= 12) || trainDocs[0];
  const order = trainDocs.map((_, i) => i);
  const epochRows = [{ epoch: 0, train_loss: null, train_accuracy: null, test_loss: ev0.metrics.loss, test_accuracy: ev0.metrics.accuracy, seconds: 0 }];
  let prev = ev0.metrics;

  for (let epoch = 1; epoch <= C.epochs; epoch++) {
    const t0 = Date.now();
    shuffle(order, rng);
    const wBefore = net.getWeights();
    const bBefore = net.getBiases();
    let sumLoss = 0, correct = 0, winLoss = 0, winCorrect = 0, winN = 0;
    const curve = [];
    let trace = null;
    log(`epoch ${epoch}/${C.epochs}: training on ${order.length} reviews ...`);

    for (let s = 0; s < order.length; s++) {
      const doc = trainDocs[order[s]];
      let p;
      if (doc === traceDoc) {
        const t = tracedStep(doc, epoch);
        p = t.prediction;
        trace = t.trace;
      } else {
        p = net.train(doc.x, doc.label, C.learningRate);
      }
      const loss = bce(p, doc.label);
      const ok = (p >= 0.5 ? 1 : 0) === doc.label ? 1 : 0;
      sumLoss += loss; correct += ok; winLoss += loss; winCorrect += ok; winN++;

      if ((s + 1) % C.curveEvery === 0 || s === order.length - 1) {
        const point = { reviews_seen: s + 1, loss_last_window: r4(winLoss / winN), accuracy_last_window: r4(winCorrect / winN), loss_so_far: r4(sumLoss / (s + 1)) };
        curve.push(point);
        emit({ type: 'progress', epoch, epochs: C.epochs, seen: s + 1, total: order.length, loss: point.loss_last_window, acc: point.accuracy_last_window });
        winLoss = 0; winCorrect = 0; winN = 0;
      }
    }

    const seconds = (Date.now() - t0) / 1000;
    sleepMs(C.pauseMs); // training finished -> evaluation + file writing
    const trainLoss = sumLoss / order.length;
    const trainAcc = correct / order.length;
    const changes = weightChanges(wBefore, bBefore);

    const ev = writeEpoch(epoch, {
      curve: {
        _explain: [
          `Running loss and accuracy while the network trained, one point every ${C.curveEvery} reviews.`,
          'Each prediction is made BEFORE the network updates on that review, so this is an honest score on reviews it had not yet learned from in this epoch.',
          '*_last_window = average over the last window only. loss_so_far = average since the start of the epoch.',
        ],
        epoch,
        points: curve,
      },
      changes,
      trace,
      summaryFile: e => ({
        _explain: 'One-glance result of this epoch. train_* is measured while training, test_* on the held-out reviews after the epoch ended. If train accuracy keeps rising while test accuracy falls, the network is memorizing (overfitting).',
        epoch,
        learning_rate: C.learningRate,
        seconds: r4(seconds),
        train_reviews: order.length,
        train_loss: r6(trainLoss),
        train_accuracy: r4(trainAcc),
        test_loss: e.metrics.loss,
        test_accuracy: e.metrics.accuracy,
        test_accuracy_change_vs_previous_epoch: r4(e.metrics.accuracy - prev.accuracy),
        test_loss_change_vs_previous_epoch: r6(e.metrics.loss - prev.loss),
        precision: e.metrics.precision,
        recall: e.metrics.recall,
        f1: e.metrics.f1,
        confusion_matrix: e.metrics.confusion_matrix,
        mean_abs_weight_change_per_group: Object.fromEntries(changes.per_link_group.map(g => [g.group, g.mean_abs_change])),
      }),
    });

    const row = { epoch, train_loss: r6(trainLoss), train_accuracy: r4(trainAcc), test_loss: ev.metrics.loss, test_accuracy: ev.metrics.accuracy, seconds: r4(seconds) };
    epochRows.push(row);
    prev = ev.metrics;
    emit({ type: 'epoch', row });
    log(`epoch ${epoch}  train loss ${row.train_loss.toFixed(4)}  train acc ${pct(row.train_accuracy)}%  test loss ${row.test_loss.toFixed(4)}  test acc ${pct(row.test_accuracy)}%  (${seconds.toFixed(1)}s)`);
    sleepMs(C.pauseMs); // epoch finished -> next epoch
  }

  // ------------------------------------------------------------------
  // 7. Final model and summary
  // ------------------------------------------------------------------
  const finalCp = JSON.parse(fs.readFileSync(path.join(runDir, `epoch_${C.epochs}`, 'checkpoint.json'), 'utf8'));
  // final_model.json: every neuron with its own bias and incoming weights.
  const finalNet = Network.fromCheckpoint(finalCp, finalCp.vocabulary);
  writeText('final_model.json', formatModel({
    _explain: 'The trained network, neuron by neuron. For each layer after the input: every neuron has a bias and the weight on each incoming link. Neuron output = sigmoid(bias + sum(input x weight)). Weights into hidden1 are keyed by word (input = TF-IDF value of that word). vocabulary and idf turn new text into those inputs.',
    epoch: finalCp.epoch,
    sizes: finalCp.sizes,
    layers: finalNet.toNeuronJSON(finalCp.vocabulary),
    vocabulary: finalCp.vocabulary,
    idf: finalCp.idf,
  }));
  const best = epochRows.slice(1).reduce((b, r) => (r.test_accuracy > b.test_accuracy ? r : b), epochRows[1] || epochRows[0]);
  const summary = {
    _explain: 'One row per epoch. epoch 0 is the untrained network. final_model.json holds the weights from the last epoch.',
    run: runName,
    total_seconds: r4((Date.now() - started) / 1000),
    best_epoch_by_test_accuracy: best.epoch,
    best_test_accuracy: best.test_accuracy,
    epochs: epochRows,
  };
  write('99_run_summary.json', summary);
  log(`Done in ${summary.total_seconds}s. Best test accuracy ${pct(best.test_accuracy)}% at epoch ${best.epoch}. Logs: runs/${runName}`);
  emit({ type: 'done', run: runName, summary });
  return { runName, runDir, summary };
}

const RUN_README = `# Training run

Suggested reading order:

1. **00_config.json** – the settings.
2. **01_dataset_stats.json** – how many reviews, label balance, review lengths.
3. **03_tfidf_examples.json** – three reviews walked from raw text to the exact numbers the input layer sees.
4. **02_vocabulary.json** – every input neuron: its token, document frequency, positive share and IDF.
5. **04_architecture.json** – layers, links and parameter count.
6. **epoch_0/** – the random, untrained network. Test accuracy ≈ 50%.
7. **epoch_1/** and later – the same files after each pass over the training data:

| file | what it shows |
|---|---|
| summary.json | loss/accuracy for the epoch, change vs the previous one |
| training_curve.json | running loss/accuracy every N reviews during the epoch |
| sample_trace.json | ONE training step on one review: forward sums, loss, every delta, weight updates, prediction before/after |
| layer_*.json | every weight and bias of each layer |
| neuron_report.json | what each hidden neuron reacts to (positive vs negative reviews, strongest words) |
| word_influence.json | each word fed alone into the network: how positive/negative it reads |
| weight_changes.json | how far the weights moved this epoch, biggest single changes |
| evaluation.json | test-set accuracy, precision, recall, confusion matrix, most confident mistakes |
| checkpoint.json | the full network (weights + biases + vocabulary + IDF), loadable |

Good comparisons:
- \`epoch_0/word_influence.json\` vs \`epoch_1/word_influence.json\`: words go from meaningless to meaningful.
- \`sample_trace.json\` across epochs: the same review, an increasingly confident prediction.
- \`neuron_report.json\` across epochs: hidden neurons turning into positive/negative detectors.
- \`summary.json\` train vs test accuracy: when they drift apart the network has started memorizing.
`;

// --- worker-thread entry (used by server.js) ---
const wt = require('worker_threads');
if (!wt.isMainThread && wt.workerData && wt.workerData.__trainer) {
  try {
    runTraining(wt.workerData, m => wt.parentPort.postMessage(m));
  } catch (e) {
    wt.parentPort.postMessage({ type: 'error', message: e.message, stack: e.stack });
  }
}

// --- command-line entry ---
if (require.main === module && wt.isMainThread) {
  const csv = process.argv[2];
  if (!csv) { console.log('Usage: node trainer.js path/to/IMDB.csv [epochs] [maxRows]'); process.exit(1); }
  runTraining({
    csvPath: path.resolve(csv),
    runsDir: path.join(__dirname, 'runs'),
    epochs: Number(process.argv[3]) || DEFAULTS.epochs,
    maxRows: Number(process.argv[4]) || 0,
  }, m => {
    if (m.type === 'log') console.log(m.text);
    if (m.type === 'progress' && m.seen % (DEFAULTS.curveEvery * 10) === 0) console.log(`  ${m.seen}/${m.total}  loss ${m.loss}  acc ${m.acc}`);
  });
}

module.exports = { runTraining, DEFAULTS, formatCheckpoint, formatModel };
