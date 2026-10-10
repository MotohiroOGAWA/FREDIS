const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const upstream = require('../src/features/spectrum-viewer/upstream');
const workbench = require('../src/features/spectrum-viewer/workbench');
const pathDrop = require('../src/webview-path-drop');
const root = path.resolve(__dirname, '../../fredis/libs/msentity/vscode-extension');
assert.equal(upstream.configure({}, { extensions: { getExtension: () => { throw new Error('Installed msentity extensions must not be used'); } } }), root);
const messages = [];
const host = new JSDOM(`<!doctype html><html><body>${workbench.html()}<script nonce="test">const vscode=window.fredisMsentityBridge;${pathDrop.script()}${workbench.script()}</script></body></html>`, {
  runScripts: 'dangerously',
  beforeParse(window) {
    window.fredisMsentityBridge = { postMessage: message => messages.push(message) };
  }
});
const frame = host.window.document.getElementById('spectrumViewerFrame');
// jsdom does not execute iframe srcdoc; load the actual generated document.
const embedded = new JSDOM(frame.srcdoc, {
  runScripts: 'dangerously',
  beforeParse(window) {
    Object.defineProperty(window, 'parent', { value: host.window });
    window.URL.createObjectURL = () => 'blob:structure';
    window.URL.revokeObjectURL = () => {};
  }
});
assert.equal(messages.shift().type, 'client-ready');
const doc = embedded.window.document;
assert.equal(doc.querySelector('script').nonce, 'test');
assert.equal(doc.querySelector('style').textContent.includes(fs.readFileSync(path.join(root, 'media/viewer.css'), 'utf8').replace(/\r\n/g, '\n')), true);
const deliver = data => embedded.window.dispatchEvent(new embedded.window.MessageEvent('message', { data }));
deliver({ type: 'backend-ready', dataset: null });
deliver({ type: 'capabilities', chemistry: {} });
assert.match(doc.body.textContent, /No datasets loaded/);
doc.getElementById('add-dataset').click();
assert.equal(messages.pop().type, 'add-dataset');
deliver({ type: 'capabilities', chemistry: { rdkit_available: true, smarts_filter: true, structure_render: true } });
deliver({ type: 'dataset-added', dataset: { id: 'test', name: 'test.msds', path: '/tmp/test.msds', total_rows: 1 } });
deliver({ type: 'dataset-page', value: {
  dataset_id: 'test', dataset_name: 'test.msds', dataset_path: '/tmp/test.msds',
  page: 0, page_size: 20, row_offset: 0, total_rows: 1, total_pages: 1,
  structure_smiles_column: 'SMILES', columns: ['SMILES', 'Name'], all_columns: ['SMILES', 'Name'], row_ids: [0],
  rows: [{ SMILES: 'CCO', Name: 'Ethanol' }],
  spectra: [{ mz: [31], intensity: [100] }], description: '', attributes: {}, tags: []
} });
assert.match(doc.body.textContent, /Ethanol/);
const structure = doc.querySelector('[data-structure-index="0"]');
assert(structure);
structure.click();
assert.equal(messages.pop().type, 'render-structure');
deliver({ type: 'structure-rendered', smiles: 'CCO', svg: '<svg xmlns="http://www.w3.org/2000/svg"><text>Ethanol structure</text></svg>' });
assert.equal(doc.querySelector('.structure-preview img').getAttribute('src'), 'blob:structure');
// Backend responses are confined to the embedded viewer.
const forwarded = [];
frame.contentWindow.postMessage = data => forwarded.push(data);
host.window.dispatchEvent(new host.window.MessageEvent('message', { data: { type: 'workbench/jobs' } }));
assert.equal(forwarded.length, 0);
host.window.dispatchEvent(new host.window.MessageEvent('message', { data: { type: 'spectrumViewer', message: { type: 'capabilities' } } }));
assert.equal(forwarded[0].type, 'capabilities');
// Remote Explorer drags can expose ResourceURLs without CodeFiles/uri-list.
// The Workbench adapter handles this before upstream's document drop handler.
const remoteDrop = new embedded.window.Event('drop', { bubbles: true, cancelable: true });
Object.defineProperty(remoteDrop, 'dataTransfer', { value: {
  types: ['resourceurls'], files: [],
  getData: type => /resourceurls/i.test(type) ? JSON.stringify(['vscode-remote://dev-container+test/workspaces/FREDIS/example.msds']) : ''
} });
embedded.window.document.body.dispatchEvent(remoteDrop);
assert(remoteDrop.defaultPrevented);
// No host stat round trip is needed in this isolated viewer test.
Promise.resolve().then(() => {
  const dropped = messages.find(message => message.type === 'drop-datasets');
  assert(dropped);
  assert.equal(dropped.paths[0], '/workspaces/FREDIS/example.msds');
  embedded.window.close();
  host.window.close();
}).catch(error => { console.error(error); process.exitCode = 1; });
console.log('Spectrum viewer embedding checks passed.');

const vm = require('vm');

// Packaged fallback is built from the submodule, without an external extension.
const os = require('os');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'fredis-viewer-build-'));
try {
  const target = path.join(temporary, 'vendor/msentity-spectrum-viewer');
  require('./build-spectrum-viewer').build(root, target);
  for (const asset of ['extension.js', 'media/viewer.js', 'media/viewer.css', 'python/backend.py']) {
    assert.equal(fs.readFileSync(path.join(target, asset), 'utf8'), fs.readFileSync(path.join(root, asset), 'utf8'));
  }
  const resolver = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/features/spectrum-viewer/upstream.js'), 'utf8'), {
    module: resolver, require, __dirname: path.join(temporary, 'src/features/spectrum-viewer')
  });
  assert.equal(resolver.exports.configure({ extensionPath: temporary }, {
    extensions: { getExtension() { throw new Error('External extensions are not used'); } }
  }), target);
  const manifest = require('../package.json');
  assert(!manifest.extensionDependencies?.includes('motohiroogawa.msentity-spectrum-viewer'));
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
console.log('Submodule viewer build and packaged fallback checks passed.');

// Run the unchanged upstream provider and backend through the Workbench host.
(async () => {
  const { createHost } = require('../src/features/spectrum-viewer/host');
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'fredis-host-check-'));
  const input = path.join(directory, 'example.msp');
  fs.writeFileSync(input, 'Name: Ethanol\nSMILES: CCO\nNum Peaks: 1\n31 100\n\n');
  const uri = fsPath => ({ fsPath, toString: () => fsPath });
  const output = { appendLine() {}, dispose() {} };
  const fakeVscode = {
    Uri: { file: uri, joinPath: (base, ...parts) => uri(path.join(base.fsPath, ...parts)) },
    workspace: { getConfiguration: () => ({ get: (_key, fallback) => fallback }) },
    commands: {},
    window: { showOpenDialog: async () => [uri(input)] }
  };
  // Match VS Code's proposed-API getters: enumerating/copying a namespace
  // must not activate APIs unrelated to the spectrum viewer.
  for (const namespace of [fakeVscode, fakeVscode.workspace, fakeVscode.window, fakeVscode.commands]) {
    Object.defineProperty(namespace, 'agentSessionsWorkspace', {
      enumerable: true,
      get() { throw new Error('CANNOT use API proposal: agentSessionsWorkspace'); }
    });
  }
  const host = createHost(root, fakeVscode, { outputChannel: output, pythonPath: 'python', pageSize: 37 });
  let listener;
  const cleanup = [];
  const responses = [];
  const waiters = [];
  const panel = {
    webview: {
      cspSource: 'test', asWebviewUri: value => value.fsPath,
      postMessage: message => {
        responses.push(message);
        for (const waiter of waiters) waiter();
      },
      onDidReceiveMessage: callback => { listener = callback; return { dispose() {} }; }
    },
    onDidDispose: callback => cleanup.push(callback)
  };
  const waitFor = type => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Timed out waiting for ' + type)), 10000);
    const inspect = () => {
      const message = responses.find(item => item.type === type);
      if (message) { clearTimeout(timer); resolve(message); }
    };
    waiters.push(inspect); inspect();
  });
  try {
    await host.resolveEmbeddedEditor(uri(path.join(directory, 'embedded.msds')), panel);
    assert.equal(responses.length, 0);
    await listener({ type: 'client-ready' });
    const ready = await waitFor('backend-ready');
    assert.equal(ready.dataset, null);
    await listener({ type: 'ready', datasetId: '' });
    await waitFor('dataset-removed');
    await listener({ type: 'add-dataset' });
    const page = await waitFor('dataset-page');
    assert.equal(page.value.rows[0].Name, 'Ethanol');
    assert.equal(page.value.page_size, 37);
    await listener({ type: 'render-structure', smiles: 'CCO' });
    assert.match((await waitFor('structure-rendered')).svg, /<svg/);
    assert(!responses.some(item => item.type === 'error'), JSON.stringify(responses));
    console.log('Unmodified msentity provider integration checks passed.');
  } finally {
    for (const callback of cleanup) callback();
    host.dispose();
    fs.rmSync(directory, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
