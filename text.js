'use strict';
/*
 * text.js — turning a review into numbers.
 *
 *   raw review
 *     -> clean      (lowercase, strip <br />, punctuation)
 *     -> tokenize   (split into words, drop stopwords, mark negation: "not good" -> "not_good")
 *     -> TF-IDF     (how often the word is in THIS review × how rare it is across ALL reviews)
 *     -> L2 normalize (so long and short reviews produce vectors of the same length)
 *
 * The result is the value each input neuron outputs.
 */

// Words that flip the meaning of the next word.
const NEGATORS = new Set(['not', 'no', 'never', 'nor', 'cannot']);

// Very common words that say nothing about sentiment. Negators are NOT in here.
const STOPWORDS = new Set(`
a about above after again against all am an and any are as at be because been before being below between both
but by could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom why will with would
you your yours yourself yourselves also im ive id ill youre youve youd theyre theyve weve were hes shes thats theres
whats lets us one get got even may might must shall can
`.trim().split(/\s+/));

function clean(raw) {
  return String(raw).toLowerCase()
    .replace(/<br\s*\/?>/g, ' ')          // IMDB reviews contain <br /><br />
    .replace(/<[^>]*>/g, ' ')             // any other html tag
    .replace(/[’‘`]/g, "'")
    .replace(/[.!?;:,()"[\]{}]+/g, ' . ') // sentence breaks become "." (they stop a negation)
    .replace(/[^a-z'. ]+/g, ' ')          // drop digits and other symbols
    .replace(/\s+/g, ' ')
    .trim();
}

// details=true returns every intermediate step (used by the logs and the "Try it" box).
function tokenize(raw, details = false) {
  const cleaned = clean(raw);
  const words = cleaned.split(' ').filter(Boolean);
  const tokens = [];
  const stopwordsRemoved = [];
  const negations = [];
  let negate = false;

  for (let w of words) {
    if (w === '.') { negate = false; continue; }
    w = w.replace(/^'+|'+$/g, '');
    if (!w) continue;

    if (NEGATORS.has(w) || w.endsWith("n't")) { negate = true; continue; }
    if (w.endsWith("'s")) w = w.slice(0, -2);
    const bare = w.replace(/'/g, '');

    if (bare.length < 2 || STOPWORDS.has(bare)) {
      if (details) stopwordsRemoved.push(bare || w);
      continue;
    }
    if (negate) {
      tokens.push('not_' + bare);
      if (details) negations.push('not_' + bare);
      negate = false;
    } else {
      tokens.push(bare);
    }
  }

  if (!details) return tokens;
  return { cleaned, words: words.filter(w => w !== '.'), stopwordsRemoved, negations, tokens };
}

// dfMap: token -> { df, dfPos, dfNeg, count }, counted on the TRAINING reviews only.
function buildVocabulary(dfMap, numDocs, { maxVocab, minDf }) {
  const all = [];
  for (const [token, s] of dfMap) all.push({ token, ...s });
  const kept = all
    .filter(e => e.df >= minDf)
    .sort((a, b) => b.df - a.df || (a.token < b.token ? -1 : 1))
    .slice(0, maxVocab);

  const tokens = kept.map(e => e.token);
  // Smoothed IDF (same formula scikit-learn uses): rare words get a big number, common words a small one.
  const idf = Float64Array.from(kept, e => Math.log((1 + numDocs) / (1 + e.df)) + 1);
  return {
    tokens,
    index: new Map(tokens.map((t, i) => [t, i])),
    idf,
    entries: kept,
    numDocs,
    distinctTokens: all.length,
    droppedRare: all.filter(e => e.df < minDf).length,
  };
}

function vocabFromLists(tokens, idf) {
  return { tokens, index: new Map(tokens.map((t, i) => [t, i])), idf: Float64Array.from(idf) };
}

// Returns only the non-zero inputs: { idx, val }.
function vectorize(tokens, vocab, details = false) {
  const counts = new Map();
  const unknown = [];
  for (const t of tokens) {
    const i = vocab.index.get(t);
    if (i === undefined) unknown.push(t);
    else counts.set(i, (counts.get(i) || 0) + 1);
  }
  const idx = [...counts.keys()].sort((a, b) => a - b);
  const raw = idx.map(i => counts.get(i) * vocab.idf[i]);
  const norm = Math.sqrt(raw.reduce((s, v) => s + v * v, 0)) || 1;
  const vector = {
    idx: Int32Array.from(idx),
    val: Float32Array.from(raw, v => v / norm),
    known: tokens.length - unknown.length,
  };
  if (!details) return vector;
  return {
    vector,
    norm,
    unknown,
    table: idx.map((i, k) => ({
      token: vocab.tokens[i],
      input_neuron: i,
      tf: counts.get(i),
      idf: vocab.idf[i],
      tf_x_idf: raw[k],
      normalized: raw[k] / norm,
    })),
  };
}

module.exports = { clean, tokenize, buildVocabulary, vocabFromLists, vectorize, STOPWORDS, NEGATORS };
