// Read only report metadata and representative records; never build a string
// containing the entire spectrum array (which can exceed V8's string limit).
const fs = require('fs');
const cache = new Map();
const MAX_VALUE_BYTES = 32 * 1024 * 1024;

function representativeIndices(entries) {
  return new Set(Object.values(entries || {}).flatMap(levels =>
    Object.values(levels).map(item => item?.spectrum_index)
  ).filter(index => Number.isInteger(index) && index >= 0));
}

async function scan(filename, requested = null) {
  const stream = fs.createReadStream(filename, { encoding: 'utf8' });
  const metadata = {}, spectra = new Map();
  let wanted = requested, depth = 0, quoted = false, escaped = false;
  let key = null, expectKey = false, expectValue = false;
  let inSpectra = false, recordActive = false, recordIndex = 0;
  let capture = null;
  const start = (kind, index, shape) => {
    capture = { kind, index, shape, base: depth, parts: [], length: 0 };
  };
  const append = part => {
    capture.length += Buffer.byteLength(part);
    if (capture.length > MAX_VALUE_BYTES) throw new Error('One validation record exceeds the 32 MB display limit.');
    capture.parts.push(part);
  };
  const finish = () => {
    const current = capture;
    const value = JSON.parse(current.parts.join(''));
    capture = null;
    if (current.kind === 'key') { key = value; expectKey = false; }
    else if (current.kind === 'record') spectra.set(current.index, value);
    else {
      metadata[current.index] = value;
      if (current.index === 'representatives' && requested === null) wanted = representativeIndices(value);
    }
  };
  const complete = () => Object.hasOwn(metadata, 'summary') && Object.hasOwn(metadata, 'representatives') &&
    wanted !== null && [...wanted].every(index => spectra.has(index));
  try {
    for await (const chunk of stream) {
      let segment = capture ? 0 : -1;
      for (let i = 0; i < chunk.length; i++) {
        const ch = chunk[i], whitespace = ch === ' ' || ch === '\n' || ch === '\r' || ch === '\t';
        if (!quoted && capture?.shape === 'primitive' && (whitespace || ',}]'.includes(ch))) {
          append(chunk.slice(segment, i)); finish(); segment = -1;
          if (complete()) return { ...metadata, spectra };
        }
        if (!quoted && !capture) {
          if (depth === 1 && expectKey && ch === '"') {
            start('key', null, 'string'); segment = i;
          } else if (depth === 1 && expectValue && !whitespace) {
            expectValue = false;
            if (key === 'summary' || key === 'representatives') {
              start('field', key, '{['.includes(ch) ? 'container' : ch === '"' ? 'string' : 'primitive'); segment = i;
            } else if (key === 'spectra' && ch === '[') inSpectra = true;
          } else if (inSpectra && depth === 2 && !recordActive && !whitespace && ch !== ',' && ch !== ']') {
            if (ch !== '{') throw new Error('Validation spectra must contain JSON objects.');
            recordActive = true;
            if (wanted?.has(recordIndex)) { start('record', recordIndex, 'container'); segment = i; }
          }
        }
        let closedString = false;
        if (quoted) {
          if (escaped) escaped = false;
          else if (ch === '\\') escaped = true;
          else if (ch === '"') { quoted = false; closedString = true; }
        } else if (ch === '"') quoted = true;
        else if (ch === '{' || ch === '[') { depth++; if (depth === 1) expectKey = true; }
        else if (ch === '}' || ch === ']') {
          if (inSpectra && depth === 2 && ch === ']') inSpectra = false;
          depth--;
          if (recordActive && depth === 2) { recordActive = false; recordIndex++; }
        } else if (depth === 1 && ch === ':') expectValue = true;
        else if (depth === 1 && ch === ',') { expectKey = true; key = null; }
        if (capture && ((capture.shape === 'string' && closedString) ||
            (capture.shape === 'container' && !quoted && depth === capture.base && '}]'.includes(ch)))) {
          append(chunk.slice(segment, i + 1)); finish(); segment = -1;
          if (complete()) return { ...metadata, spectra };
        }
      }
      if (capture) append(chunk.slice(segment));
    }
    if (quoted || depth !== 0 || capture) throw new Error('Validation JSON is incomplete; retry when writing finishes.');
    return { ...metadata, spectra };
  } finally { stream.destroy(); }
}

async function readValidation(filename) {
  // New writers publish a compact sidecar last. Older runs use the streaming
  // path, including files larger than V8's maximum string length.
  const sidecar = filename.replace(/\.json$/, '.summary.json');
  let source = filename, compact = false;
  try {
    const [small, full] = await Promise.all([fs.promises.stat(sidecar), fs.promises.stat(filename)]);
    if (small.mtimeMs >= full.mtimeMs) { source = sidecar; compact = true; }
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const stat = await fs.promises.stat(source);
  const version = `${stat.size}:${stat.mtimeMs}`;
  if (cache.get(source)?.version === version) return cache.get(source).value;
  let value;
  if (compact) value = JSON.parse(await fs.promises.readFile(source, 'utf8'));
  else {
    let parsed = await scan(filename);
    const wanted = representativeIndices(parsed.representatives);
    // JSON field order is not a contract. If representatives follow spectra,
    // make a second bounded-memory pass for the selected indices.
    if ([...wanted].some(index => !parsed.spectra.has(index))) {
      const second = await scan(filename, wanted);
      for (const [index, record] of second.spectra) parsed.spectra.set(index, record);
    }
    const representatives = {};
    for (const [metric, levels] of Object.entries(parsed.representatives || {})) {
      representatives[metric] = {};
      for (const [quantile, item] of Object.entries(levels)) {
        const spectrum = parsed.spectra.get(item.spectrum_index);
        if (spectrum) representatives[metric][quantile] = { ...item, spectrum };
      }
    }
    value = { summary: parsed.summary || {}, representatives };
  }
  cache.set(source, { version, value });
  if (cache.size > 8) cache.delete(cache.keys().next().value);
  return value;
}

module.exports = { readValidation };
