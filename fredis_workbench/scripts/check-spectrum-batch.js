const assert = require('assert/strict');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function(request, parent, main) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, main);
};
const feature = require('../src/features/spectrum-prediction/editor');

const args = feature.buildBatchArgs({
  input: '/data/source database.msds', outputDir: '/data/prediction output',
  modelPath: '/models/model.pt', device: 'cuda', db: 'MoNA', maxSamples: 16,
  specIdColumn: 'SpecID', smilesColumn: 'SMILES', precursorMzColumn: 'PrecursorMZ',
  adductTypeColumn: 'AdductType', collisionEnergyColumn: 'CollisionEnergy',
  instrumentColumn: 'InstrumentType', overwrite: true
});
assert.deepEqual(args.slice(0, 2), ['-m', 'fredis.ml.specgen.predict_spectrum']);
assert.deepEqual(args.slice(args.indexOf('--db'), args.indexOf('--db') + 2), ['--db', 'MoNA']);
assert.deepEqual(args.slice(args.indexOf('--max-samples'), args.indexOf('--max-samples') + 2), ['--max-samples', '16']);
assert.deepEqual(args.slice(args.indexOf('--output-dir'), args.indexOf('--output-dir') + 2), ['--output-dir', '/data/prediction output']);
assert(!args.includes('--output-name'));
assert(!args.includes('--precompute-workers'));
assert(args.includes('--overwrite'));
assert(!args.includes('--keep-temp'));
assert(!args.includes('--filter-low-intensity-peaks'));
assert(feature.buildBatchArgs({filterLowIntensityPeaks:true}).includes('--filter-low-intensity-peaks'));
assert(feature.html().includes('name="filterLowIntensityPeaks" type="checkbox"'));
assert(args.includes('--instrument-column'));
assert.deepEqual(args.slice(args.indexOf('--num-workers'), args.indexOf('--num-workers') + 2), ['--num-workers', '1']);
assert.deepEqual(args.slice(args.indexOf('--chunk-size'), args.indexOf('--chunk-size') + 2), ['--chunk-size', '1']);
const parallelArgs = feature.buildBatchArgs({ input: 'x.msds', outputDir: 'out', modelPath: 'm.pt', numWorkers: 4, chunkSize: 8, keepTemp: true });
assert.equal(parallelArgs[parallelArgs.indexOf('--num-workers') + 1], '4');
assert.equal(parallelArgs[parallelArgs.indexOf('--chunk-size') + 1], '8');
assert(parallelArgs.includes('--keep-temp'));
assert(feature.shellDisplay('python', args).includes("'/data/source database.msds'"));
assert(feature.html().includes('Predict MSDataset'));
assert(feature.html().includes('predictLoadConfig'));
assert(feature.html().includes('predictLoadConfigFile'));
assert(feature.html().includes('predictSaveConfig'));
assert(feature.html().includes('batchMaxSamples'));
assert(feature.html().includes('batchNumWorkers'));
assert(feature.html().includes('batchChunkSize'));
assert(feature.html().includes('batchKeepTemp'));
assert(feature.script().includes("postMessage({ type: 'predictLoadConfigFile' }"));
new Function(feature.script());

const { workbenchHtml } = require('../src/extension');
Module._load = originalLoad;
const page = workbenchHtml({}, '{}', {});
assert(page.includes('id="predictBatchSection"'));
assert(page.includes('batchSpecIdColumn'));
assert(page.includes('batchOutputDir'));
for (const match of page.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
(async () => {
  // predictLoadConfigFile: reads prediction.pft (as predict_spectrum.py's
  // main() writes it) via a real file dialog + fs read, and relays it through
  // the same predictConfigLoaded message the existing Load Configuration button
  // already posts, so the client applies it with no further changes.
  const fs = require('fs'), os = require('os'), path = require('path');
  const pftPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'fredis-pft-')), 'prediction.pft');
  const pftConfig = { application: 'spectrum-prediction', input: '/data/in.msds', outputDir: '/data/out', modelPath: '/models/m.pt', db: 'MoNA', numWorkers: 4, chunkSize: 8 };
  fs.writeFileSync(pftPath, JSON.stringify(pftConfig));
  const posted = [];
  const isolatedMock = { workspace: { getConfiguration: () => ({ get: (key, fallback) => fallback }) },
    window: { showOpenDialog: async () => [{ fsPath: pftPath }] } };
  const previousLoad = Module._load;
  Module._load = function(request, parent, main) { if (request === 'vscode') return isolatedMock; return previousLoad.call(this, request, parent, main); };
  delete require.cache[require.resolve('../src/features/spectrum-prediction/editor')];
  const isolatedFeature = require('../src/features/spectrum-prediction/editor');
  Module._load = previousLoad;
  delete require.cache[require.resolve('../src/features/spectrum-prediction/editor')];
  const fakePanel = { webview: { postMessage: value => posted.push(value), onDidReceiveMessage: fn => { fakePanel._handler = fn; } }, onDidDispose: () => {} };
  isolatedFeature.attach(fakePanel, {}, () => '/root', { show() {}, appendLine() {} });
  await fakePanel._handler({ type: 'predictLoadConfigFile' });
  const loaded = posted.find(message => message.type === 'predictConfigLoaded');
  assert(loaded, 'predictLoadConfigFile did not post predictConfigLoaded');
  assert.equal(loaded.path, pftPath);
  assert.deepEqual(loaded.config, pftConfig);
  console.log('Load From Run (.pft) round-trip check passed.');
})().then(() => {
  console.log('Batch spectrum prediction CLI and Workbench checks passed.');
}).catch(error => { console.error(error); process.exit(1); });
