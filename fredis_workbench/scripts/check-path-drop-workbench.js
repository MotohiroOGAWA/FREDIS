// Drives real drop events through the Workbench page and checks that each
// drop follows the same spectrum_batch path as Browse.
const assert = require('assert/strict');
const Module = require('module');
const { JSDOM, VirtualConsole } = require('jsdom');
const mock = { workspace: { getConfiguration: () => ({ get: () => 'python' }) }, env: { clipboard: { writeText: async () => {} } } };
const load = Module._load;
Module._load = function (name, parent, main) { return name === 'vscode' ? mock : load.call(this, name, parent, main); };
const extension = require('../src/extension');
const preset = require('./fixtures').model;

const plain = value => JSON.parse(JSON.stringify(value));
const settle = async () => { for (let i = 0; i < 5; i++) await new Promise(resolve => setTimeout(resolve, 0)); };

(async () => {
  const messages = [], errors = [], virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', error => errors.push(error.detail?.stack || error.message));
  const dom = new JSDOM(extension.workbenchHtml(require('./fixtures').dataConfig, { modelConfig: preset, epochs: 1, batchSize: 4, lr: 0.0001, adapterWidth: 8 }), {
    runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole,
    beforeParse(w) {
      w.acquireVsCodeApi = () => ({ postMessage: message => messages.push(message), getState: () => ({}), setState() {} });
      w.HTMLElement.prototype.scrollIntoView = function () {};
      w.ResizeObserver = class { observe() {} disconnect() {} };
      w.matchMedia = () => ({ matches: false, addEventListener() {} });
      w.CSS = { escape: s => s };
    }
  });
  const w = dom.window, d = w.document;
  try {
    // The host's fs.stat() check is covered by check-path-drop.js; here every path is accepted.
    const statKinds = [];
    w.validateDroppedPaths = (paths, kind) => { statKinds.push(kind); return paths; };
    const drop = async (target, data = {}, files = [], shiftKey = target.matches('[data-file]')) => {
      const event = new w.Event('drop', { bubbles: true, cancelable: true });
      Object.defineProperty(event, 'shiftKey', { value: shiftKey });
      Object.defineProperty(event, 'dataTransfer', { value: { types: Object.keys(data), files, getData: type => data[type] || '' } });
      target.dispatchEvent(event);
      await settle();
      return event;
    };

    assert.equal(d.getElementById('pathDropDiagnostics'), null);

    // Data Preparation: dataset drop -> same 'picked' flow as Browse -> column detection / preview.
    const form = d.getElementById('form');
    let datasetZone = d.querySelector('.preparation-dataset-card [data-file]');
    assert(datasetZone, 'Preparation dataset path input');
    assert.equal(d.querySelector('.preparation-dataset-card .dataset-drop'),null);
    assert.equal(d.querySelector('.preparation-dataset-card [data-browse]').textContent,'Browse');
    await drop(datasetZone, {'text/uri-list':'file:///ignored.msds'}, [], false);
    assert.equal(datasetZone.value,'','file drop requires Shift');
    messages.length = 0;
    const event = await drop(datasetZone.querySelector('*') || datasetZone, { 'text/uri-list': 'vscode-remote://dev-container%2Babc/workspaces/FREDIS/data/train.msds' });
    assert(event.defaultPrevented);
    assert.equal(d.querySelector('.preparation-dataset-card [data-file]').value, '/workspaces/FREDIS/data/train.msds');
    const inspections = messages.filter(message => /^data\/(columns|preview)$/.test(message.type));
    assert(inspections.length, 'a dropped dataset is inspected like a browsed one');
    assert(inspections.every(message => message.path === '/workspaces/FREDIS/data/train.msds'));
    assert.equal(new Set(messages.filter(message => message.type === 'data/columns').map(message => message.requestId)).size, messages.filter(message => message.type === 'data/columns').length, 'one inspection request per drop');
    assert.equal(statKinds.at(-1), 'file');

    // The same flow from Browse produces the same inspection requests.
    d.getElementById('addDataset').click();
    const secondCard=d.querySelectorAll('.preparation-dataset-card')[1];
    datasetZone=d.querySelector('.preparation-dataset-card [data-file]');
    messages.length = 0;
    w.dispatchEvent(new w.MessageEvent('message', { data: { type: 'picked', form: 'data', field: secondCard.id, value: '/workspaces/FREDIS/data/val.msds' } }));
    await settle();
    assert.equal(secondCard.querySelector('[data-file]').value, '/workspaces/FREDIS/data/val.msds');
    const browsed = messages.filter(message => /^data\/(columns|preview)$/.test(message.type)).map(message => message.type);
    messages.length = 0;
    await drop(secondCard.querySelector('[data-file]'), { 'text/uri-list': 'file:///workspaces/FREDIS/data/val2.msds' });
    assert.equal(secondCard.querySelector('[data-file]').value, '/workspaces/FREDIS/data/val2.msds');
    assert.deepEqual(messages.filter(message => /^data\/(columns|preview)$/.test(message.type)).map(message => message.type), browsed, 'Drop and Browse trigger identical inspections');

    // OS drop without a path uploads the dataset and continues with the stored copy.
    const uploads = [];
    w.uploadDataset = async (file, progress) => { uploads.push(file.name); progress(100); return '/storage/datasets/upload.msds'; };
    messages.length = 0;
    await drop(datasetZone, {}, [{ name: 'external.msds', size: 3 }]);
    assert.deepEqual(uploads, ['external.msds']);
    assert.equal(d.querySelector('.preparation-dataset-card [data-file]').value, '/storage/datasets/upload.msds');
    assert(messages.some(message => message.type === 'data/columns' && message.path === '/storage/datasets/upload.msds'));
    assert.equal(datasetZone.parentElement.querySelector('[data-browse]').textContent, 'Browse');

    // Fragment Tree Training: folder fields are validated as folders and follow the Browse flow.
    const trainingForm = d.getElementById('trainingForm');
    const trainDirZone = trainingForm.elements.trainDir.closest('.path').querySelector('.file-drop-zone');
    assert.equal(trainDirZone.textContent, 'Drop a folder here or click to Browse');
    await drop(trainDirZone, { 'text/uri-list': 'file:///workspaces/FREDIS/prepared/train' });
    assert.equal(trainingForm.elements.trainDir.value, '/workspaces/FREDIS/prepared/train');
    assert.equal(statKinds.at(-1), 'folder');
    uploads.length = 0;
    await drop(trainDirZone, {}, [{ name: 'train', size: 0 }]);
    assert.deepEqual(uploads, [], 'folders are never uploaded');
    assert.match(trainDirZone.parentElement.querySelector('.path-drop-error').textContent, /A folder is required/);
    const resumeZone = trainingForm.elements.resume.closest('.path').querySelector('.file-drop-zone');
    // Disabled fields (resume in New-model mode) ignore drops just like their disabled Browse button.
    assert(resumeZone.disabled);
    await drop(resumeZone, { 'text/uri-list': 'file:///models/ignored.pt' });
    assert.equal(trainingForm.elements.resume.value, '');
    const resumeMode = trainingForm.querySelector('input[name=trainingMode][value=resume]');
    resumeMode.checked = true;
    resumeMode.dispatchEvent(new w.Event('change', { bubbles: true }));
    await settle();
    assert(!resumeZone.disabled);
    await drop(resumeZone, { 'text/uri-list': 'file:///models/run/checkpoint.pt' }, [{ name: 'checkpoint.pt', size: 10 }]);
    assert.equal(trainingForm.elements.resume.value, '/models/run/checkpoint.pt');
    assert(messages.some(message => message.type === 'training/checkpoint' && message.path === '/models/run/checkpoint.pt'), 'checkpoint settings load as after Browse');
    assert.deepEqual(uploads, [], 'checkpoints dropped from VS Code are used in place');
    assert.equal(statKinds.at(-1), 'file');

    // Spectrum Prediction: dataset drop -> same inspection as Browse.
    const predictForm = d.getElementById('predictForm');
    messages.length = 0;
    await drop(d.getElementById('predictDatasetDrop'), { CodeFiles: JSON.stringify(['/workspaces/FREDIS/data/batch.msds']) });
    assert.equal(predictForm.elements.batchInput.value, '/workspaces/FREDIS/data/batch.msds');
    assert(messages.some(message => message.type === 'data/columns' && message.target === 'prediction' && message.path === '/workspaces/FREDIS/data/batch.msds'));
    assert.equal(messages.filter(message => message.type === 'data/columns').length, 1, 'one inspection request per drop');
    const modelZone = predictForm.elements.modelPath.closest('.path').querySelector('.file-drop-zone');
    await drop(modelZone, { 'text/uri-list': 'file:///models/model.pt' });
    assert.equal(predictForm.elements.modelPath.value, '/models/model.pt');

    // Reaction Pattern Analysis: the result must keep its original location.
    uploads.length = 0;
    messages.length = 0;
    await drop(d.getElementById('rpaResultDrop'), {}, [{ name: 'result.reactionpatternanalysis.pft', size: 10 }]);
    assert.deepEqual(uploads, []);
    assert.match(d.getElementById('rpaResultInfo').textContent, /must be dropped from VS Code Explorer so its companion compounds\/ and details\/ directories remain accessible/);
    await drop(d.getElementById('rpaResultDrop'), { 'text/uri-list': 'file:///runs/a/result.reactionpatternanalysis.pft' });
    assert.deepEqual(plain(messages.filter(message => message.type === 'reactionPatternAnalysis/openResult')), [{ type: 'reactionPatternAnalysis/openResult', path: '/runs/a/result.reactionpatternanalysis.pft' }]);
    await drop(d.getElementById('rpaDatasetDrop'), { 'text/uri-list': 'file:///data/a.msds\nfile:///data/b.msds' });
    assert.deepEqual(plain(messages.filter(message => message.type === 'reactionPatternAnalysis/addSources').at(-1).paths), ['/data/a.msds', '/data/b.msds']);
    const outputZone = d.getElementById('rpaPickOutput');
    await drop(outputZone, { 'text/uri-list': 'file:///runs/out' });
    assert.equal(d.getElementById('rpaOutput').value, '/runs/out');
    assert.equal(statKinds.at(-1), 'folder');
    await drop(d.getElementById('rpaPickParams'), { 'text/uri-list': 'file:///cfg/model.json' });
    assert.deepEqual(plain(messages.filter(message => message.type === 'reactionPatternAnalysis/loadPatterns').at(-1)), { type: 'reactionPatternAnalysis/loadPatterns', path: '/cfg/model.json' }, 'a dropped Fragmenter is applied like a browsed one');

    assert.deepEqual(errors, []);
    console.log('Workbench drag-and-drop integration checks passed.');
  } finally {
    w.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
