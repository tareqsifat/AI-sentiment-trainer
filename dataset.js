'use strict';
/*
 * dataset.js — reading the IMDB CSV.
 *
 * The Kaggle "IMDB Dataset of 50K Movie Reviews" file looks like:
 *   review,sentiment
 *   "One of the other reviewers has mentioned ... <br /><br />...",positive
 *
 * Reviews contain commas, quotes ("" inside quotes) and sometimes newlines,
 * so a proper CSV parser is needed, not split(',').
 */

function parseCSV(str) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  let i = str.charCodeAt(0) === 0xfeff ? 1 : 0; // skip BOM
  const n = str.length;

  while (i < n) {
    if (inQuotes) {
      const q = str.indexOf('"', i);
      if (q === -1) { field += str.slice(i); i = n; break; }
      field += str.slice(i, q);
      if (str[q + 1] === '"') { field += '"'; i = q + 2; }   // "" = escaped quote
      else { inQuotes = false; i = q + 1; }
      continue;
    }
    const c = str[i];
    if (c === '"') { inQuotes = true; i++; continue; }
    if (c === ',') { row.push(field); field = ''; i++; continue; }
    if (c === '\r') { i++; continue; }
    if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; i++; continue; }
    let j = i;
    while (j < n) {
      const d = str.charCodeAt(j);
      if (d === 44 || d === 10 || d === 13) break; // , \n \r
      j++;
    }
    field += str.slice(i, j);
    i = j;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows;
}

const TEXT_NAMES = ['review', 'text', 'sentence', 'comment', 'content'];
const LABEL_NAMES = ['sentiment', 'label', 'polarity', 'class', 'target'];

function labelOf(value) {
  const s = String(value).trim().toLowerCase();
  if (['positive', 'pos', '1', 'good'].includes(s)) return 1;
  if (['negative', 'neg', '0', 'bad'].includes(s)) return 0;
  return null;
}

function readDataset(csvString) {
  const rows = parseCSV(csvString).filter(r => !(r.length === 1 && r[0].trim() === ''));
  if (!rows.length) throw new Error('The CSV file is empty.');

  const header = rows[0].map(h => h.trim().toLowerCase());
  let textCol = header.findIndex(h => TEXT_NAMES.includes(h));
  let labelCol = header.findIndex(h => LABEL_NAMES.includes(h));
  const hasHeader = textCol >= 0 || labelCol >= 0;
  if (textCol < 0) textCol = labelCol === 0 ? 1 : 0;
  if (labelCol < 0) labelCol = textCol === 0 ? 1 : 0;

  const docs = [];
  let skipped = 0;
  for (let r = hasHeader ? 1 : 0; r < rows.length; r++) {
    const row = rows[r];
    const label = labelOf(row[labelCol]);
    const text = row[textCol];
    if (label === null || !text || !text.trim()) { skipped++; continue; }
    docs.push({ text, label });
  }
  return {
    docs,
    skipped,
    header: hasHeader ? rows[0] : null,
    textColumn: hasHeader ? rows[0][textCol] : `column ${textCol + 1}`,
    labelColumn: hasHeader ? rows[0][labelCol] : `column ${labelCol + 1}`,
  };
}

module.exports = { parseCSV, readDataset, labelOf };
