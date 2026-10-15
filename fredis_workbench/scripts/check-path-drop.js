const assert = require('assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { JSDOM } = require('jsdom');
const pathDrop = require('../src/webview-path-drop');
const datasetUpload = require('../src/workbench/dataset-upload');
const { pathsFromDroppedValue, pathFromDroppedValue, pathsFromDataTransfer, pathFromDataTransfer, pickedMessageFor, script, css } = pathDrop;

// Values created inside jsdom belong to another realm; compare them as plain data.
const plain = value => JSON.parse(JSON.stringify(value));
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const settle = async (count = 5) => { for (let i = 0; i < count; i++) await tick(); };
const transfer = (data = {}, files = []) => ({
  types: Object.keys(data),
  files,
  getData: type => data[type] || ''
});
// jsdom's File has no arrayBuffer(); the upload client only needs name/size/slice.
const fakeFile = (name, content, extra = {}) => {
  const buffer = Buffer.from(content);
  return {
    name, size: buffer.length, ...extra,
    slice: (start, end) => ({ arrayBuffer: async () => Uint8Array.from(buffer.subarray(start, end)).buffer }),
    text: async () => buffer.toString('utf8')
  };
};

// ---------------------------------------------------------------- URI parsing
assert.equal(pathFromDroppedValue('file:///workspaces/FREDIS/a%20file.msds'), '/workspaces/FREDIS/a file.msds');
assert.equal(pathFromDroppedValue('file:///C:/Data/test.msds'), 'C:/Data/test.msds');
assert.equal(pathFromDroppedValue('file:///c%3A/Data/test.msds'), 'c:/Data/test.msds');
assert.equal(pathFromDroppedValue('vscode-remote://dev-container%2Babc/workspaces/FREDIS/test.msds'), '/workspaces/FREDIS/test.msds');
assert.equal(pathFromDroppedValue('vscode-remote://dev-container+7b22686f737450617468223a22433a5c5c227d/workspaces/FREDIS/test.msds'), '/workspaces/FREDIS/test.msds');
assert.equal(pathFromDroppedValue('/workspaces/FREDIS/result.mssim'), '/workspaces/FREDIS/result.mssim');
assert.equal(pathFromDroppedValue('C:\\Data\\test.msds'), 'C:\\Data\\test.msds');
assert.equal(pathFromDroppedValue('\\\\server\\share\\test.msds'), '\\\\server\\share\\test.msds');
assert.equal(pathFromDroppedValue('file://server/share/test.msds'), '//server/share/test.msds');
assert.equal(pathFromDroppedValue('result.mssim'), '');
assert.equal(pathFromDroppedValue('https://example.com/test.msds'), '');
assert.equal(pathFromDroppedValue(JSON.stringify({ resourceUri: 'vscode-remote://dev-container%2Babc/workspaces/FREDIS/config.pft' })), '/workspaces/FREDIS/config.pft');
assert.deepEqual(pathsFromDroppedValue(JSON.stringify(['file:///tmp/first.pft', 'vscode-remote://dev-container%2Babc/workspaces/FREDIS/second.pft'])), ['/tmp/first.pft', '/workspaces/FREDIS/second.pft']);
assert.equal(pathFromDroppedValue(JSON.stringify([{ resource: { $mid: 1, fsPath: '/tmp/from-editor.pft', external: 'file:///tmp/from-editor.pft', path: '/tmp/from-editor.pft', scheme: 'file' } }])), '/tmp/from-editor.pft');

assert.equal(pathFromDataTransfer(transfer({ 'application/vnd.code.uri-list': 'file:///tmp/code-uri.msds' })), '/tmp/code-uri.msds');
assert.equal(pathFromDataTransfer(transfer({ 'text/uri-list': '# files\nfile:///tmp/first.mssim\r\nfile:///tmp/second.mssim' })), '/tmp/first.mssim');
assert.deepEqual(pathsFromDataTransfer(transfer({ ResourceURLs: JSON.stringify(['file:///tmp/explorer-one.pft', 'file:///tmp/explorer-two.pft']) })), ['/tmp/explorer-one.pft', '/tmp/explorer-two.pft']);
assert.equal(pathFromDataTransfer(transfer({ CodeFiles: JSON.stringify(['/tmp/from-codefiles.pft']) })), '/tmp/from-codefiles.pft');
assert.equal(pathFromDataTransfer({ files: [], getData: type => type === 'codefiles' ? JSON.stringify(['/tmp/lower.pft']) : '' }), '/tmp/lower.pft');
assert.equal(pathFromDataTransfer(transfer({ CodeEditors: JSON.stringify([{ resource: { fsPath: '/tmp/open-editor.msds', scheme: 'file' } }]) })), '/tmp/open-editor.msds');
assert.equal(pathFromDataTransfer(transfer({ 'text/x-moz-url': 'file:///tmp/from-moz-url.pft\nConfiguration' })), '/tmp/from-moz-url.pft');
assert.equal(pathFromDataTransfer(transfer({ 'text/uri-list': 'vscode-remote://dev-container%2Babc/workspaces/FREDIS/remote.msds' })), '/workspaces/FREDIS/remote.msds');
assert.equal(pathFromDataTransfer(transfer({ 'text/uri-list': 'file:///C:/Data/windows.msds' })), 'C:/Data/windows.msds');
assert.equal(pathFromDataTransfer(transfer({}, [{ name: 'from-os.msds', path: '/tmp/from-os.msds' }])), '/tmp/from-os.msds');
// URI formats keep the original path, so they win over Electron's File.path.
assert.equal(pathFromDataTransfer(transfer({ 'text/uri-list': 'file:///tmp/uri.msds' }, [{ path: '/tmp/file-path.msds' }])), '/tmp/uri.msds');
// A Webview File without path cannot produce a path; the drop zone uploads it instead.
assert.deepEqual(pathsFromDataTransfer(transfer({}, [{ name: 'webview.msds' }])), []);
assert.deepEqual(pathsFromDataTransfer(null), []);
assert.deepEqual(pathsFromDataTransfer({ files: [], getData: () => { throw new Error('protected'); } }), []);

// Multiple files: two files give two paths, the same file twice gives one.
assert.deepEqual(pathsFromDataTransfer(transfer({ 'text/uri-list': 'file:///tmp/a.smi\nfile:///tmp/b.smi' })), ['/tmp/a.smi', '/tmp/b.smi']);
assert.deepEqual(pathsFromDataTransfer(transfer({ 'text/uri-list': 'file:///tmp/a.smi\nfile:///tmp/a.smi' })), ['/tmp/a.smi']);
assert.deepEqual(pathsFromDataTransfer(transfer({
  'text/uri-list': 'file:///tmp/a.smi',
  ResourceURLs: JSON.stringify(['file:///tmp/a.smi']),
  CodeFiles: JSON.stringify(['/tmp/a.smi'])
}, [{ path: '/tmp/a.smi' }])), ['/tmp/a.smi']);

assert.deepEqual(pickedMessageFor(null, { dataset: { pick: 'input', form: 'data' } }, '/x'), { type: 'picked', form: 'data', field: 'input', value: '/x' });
assert.deepEqual(pickedMessageFor(null, { dataset: { predictBatchPick: 'input' } }, '/x'), { type: 'predictBatchPicked', field: 'input', value: '/x' });
assert.deepEqual(pickedMessageFor(null, { dataset: { libraryPick: 'lib' } }, '/x'), { type: 'library/picked', target: 'lib', path: '/x' });
assert.equal(pickedMessageFor(null, { dataset: {} }, '/x'), null);

assert.doesNotMatch(script(), /pathDropEnvironment|pathDropEventStatus|pathDrop\/trace/);
assert.match(script(), /data-path-kind/);
assert.match(script(), /ResourceURLs/);
assert.match(css(), /path-drop-active/);

// ------------------------------------------------------------------ DOM events
function page(body, { host = false, before } = {}) {
  const posted = [];
  const dom = new JSDOM(`<!doctype html><body>${body}<script>const vscode=acquireVsCodeApi();${script({ host })}</script></body>`, {
    runScripts: 'dangerously',
    beforeParse(window) {
      window.acquireVsCodeApi = () => ({ postMessage: message => { posted.push(message); window.onHostMessage?.(message); } });
      before?.(window);
    }
  });
  const w = dom.window;
  const fire = (target, type, data, init = {}) => {
    const event = new w.Event(type, { bubbles: true, cancelable: true });
    if (data !== undefined) Object.defineProperty(event, 'dataTransfer', { value: data });
    if ('shiftKey' in init) Object.defineProperty(event, 'shiftKey', { value: init.shiftKey });
    if ('relatedTarget' in init) Object.defineProperty(event, 'relatedTarget', { value: init.relatedTarget });
    target.dispatchEvent(event);
    return event;
  };
  return { dom, w, d: w.document, posted, fire };
}

(async () => {
  // Model the VS Code Webview preload's window-level drag interception.
  // Entering page whitespace must not disable the iframe before reaching
  // a feature-specific zone. getData is protected until the drop itself.
  {
    const { dom, w, d, fire } = page('<h1 id="heading">Workbench</h1><div class="path"><input data-path-kind="file"><button>Browse</button></div>');
    assert.match(d.getElementById('fredisExplorerDropHint').textContent, /hold Shift/);
    assert.equal(d.querySelectorAll('#fredisExplorerDropHint').length, 1);
    let blocked = false, forwarded = 0;
    w.addEventListener('dragenter', event => {
      if (!event.defaultPrevented && !event.shiftKey) blocked = true;
    });
    w.addEventListener('dragover', event => { forwarded++; if (event.shiftKey) blocked = false; });
    const protectedTransfer = {
      types: ['Files', 'resourceurls'], files: [], items: [{ kind: 'file' }],
      getData: () => ''
    };
    fire(d.getElementById('heading'), 'dragenter', protectedTransfer);
    fire(d.getElementById('heading'), 'dragover', protectedTransfer);
    assert(!blocked, 'preload must not disable the iframe when entering whitespace');
    assert.equal(forwarded, 0, 'file drags must not trigger editor overlay forwarding');
    const input = d.querySelector('input');
    fire(d.querySelector('button'), 'drop', transfer({ resourceurls: JSON.stringify(['vscode-remote://dev-container+test/workspaces/FREDIS/data.msds']) }));
    await settle();
    assert.equal(input.value, '/workspaces/FREDIS/data.msds');
    const plainDrag = fire(d.getElementById('heading'), 'dragover', transfer({ 'text/plain': 'internal selection' }));
    assert(!plainDrag.defaultPrevented, 'text and internal feature drags retain their normal behavior');
    assert.equal(forwarded, 1);
    // Shift-held hover must reach the preload, even on a feature drop zone.
    blocked = true;
    const zone = d.querySelector('button');
    fire(zone, 'dragenter', protectedTransfer, { shiftKey: true });
    fire(zone, 'dragover', protectedTransfer, { shiftKey: true });
    assert(!blocked, 'Shift hover reaches the host to dismiss its overlay');
    assert(zone.classList.contains('path-drop-active'));
    assert.equal(forwarded, 2);
    let hostDrops = 0;
    w.addEventListener('drop', () => hostDrops++);
    fire(zone, 'drop', transfer({ ResourceURLs: JSON.stringify(['file:///tmp/shift.msds']) }), { shiftKey: true });
    await settle();
    assert.equal(input.value, '/tmp/shift.msds');
    assert.equal(hostDrops, 0, 'handled drops stay in Workbench');
    dom.window.close();
  }

  // Browse fields are decorated as drop zones without any Shift requirement.
  {
    const { w, d, fire } = page('<label>Dataset<div class="path"><input name="dataset" data-path-kind="file"><button type="button">Browse</button></div></label><div id="outside"></div>');
    const input = d.querySelector('input'), zone = d.querySelector('button');
    assert(zone.classList.contains('file-drop-zone'));
    assert.equal(zone.textContent, 'Drop a file here or click to Browse');
    const events = [];
    input.addEventListener('input', () => events.push('input:' + input.value));
    input.addEventListener('change', () => events.push('change:' + input.value));
    const data = transfer({ 'text/uri-list': 'file:///workspaces/FREDIS/data/train%20set.msds' });

    const enter = fire(zone, 'dragenter', data);
    assert(enter.defaultPrevented);
    assert(zone.classList.contains('path-drop-active'), 'dragenter turns the highlight on');
    const over = fire(zone, 'dragover', data);
    assert(over.defaultPrevented, 'dragover must be cancelled for the drop to be delivered');
    assert.equal(data.dropEffect, 'copy');
    const drop = fire(zone, 'drop', data);
    assert(drop.defaultPrevented);
    assert(!zone.classList.contains('path-drop-active'), 'drop turns the highlight off');
    await settle();
    assert.equal(input.value, '/workspaces/FREDIS/data/train set.msds');
    assert.deepEqual(events, ['input:/workspaces/FREDIS/data/train set.msds', 'change:/workspaces/FREDIS/data/train set.msds']);

    // The input itself accepts drops too.
    fire(input, 'dragover', data);
    assert(input.classList.contains('path-drop-active'));
    fire(input, 'drop', transfer({ ResourceURLs: JSON.stringify(['vscode-remote://dev-container%2Babc/workspaces/FREDIS/other.msds']) }));
    await settle();
    assert.equal(input.value, '/workspaces/FREDIS/other.msds');
    assert(!input.classList.contains('path-drop-active'));

    // dragend always clears the highlight.
    fire(zone, 'dragenter', data);
    fire(zone, 'dragend');
    assert(!zone.classList.contains('path-drop-active'));
    // A drop with nothing usable reports a visible error instead of returning silently.
    fire(zone, 'drop', transfer());
    await settle();
    assert.match(d.querySelector('.path-drop-error').textContent, /Unable to read the dropped file\. Try dropping it from the VS Code Explorer or use Browse\./);
    assert.equal(input.value, '/workspaces/FREDIS/other.msds');
    fire(zone, 'drop', data);
    await settle();
    assert.equal(d.querySelector('.path-drop-error'), null, 'a successful drop clears the previous error');
    assert.equal(w.document.querySelectorAll('.file-drop-zone').length, 1);
  }

  // Nested targets resolve to the parent zone and moving between children does not flicker.
  {
    const { w, d, fire } = page('<button type="button" id="zone"><svg><text>icon</text></svg><span id="label">Drop <small id="hint">files</small> here</span></button><p id="outside">x</p>');
    const zone = d.getElementById('zone'), span = d.getElementById('label'), small = d.getElementById('hint');
    const received = [];
    w.installFileDropZone(zone, { kind: 'file', multiple: true, onDrop: paths => received.push(paths) });
    const data = transfer({ 'text/uri-list': 'file:///tmp/a.smi' });
    fire(span, 'dragenter', data);
    fire(small, 'dragenter', data);
    assert(zone.classList.contains('path-drop-active'));
    fire(span, 'dragleave', data, { relatedTarget: small });
    assert(zone.classList.contains('path-drop-active'), 'moving onto a child keeps the highlight');
    fire(small, 'dragleave', data, { relatedTarget: null });
    assert(zone.classList.contains('path-drop-active'), 'depth counter keeps the highlight when relatedTarget is unavailable');
    fire(span, 'dragleave', data, { relatedTarget: d.getElementById('outside') });
    assert(!zone.classList.contains('path-drop-active'), 'leaving the zone turns the highlight off');
    fire(span, 'dragenter', data);
    fire(span, 'dragleave', data, { relatedTarget: null });
    assert(!zone.classList.contains('path-drop-active'), 'leaving without relatedTarget turns the highlight off');

    fire(span, 'dragover', data);
    assert(zone.classList.contains('path-drop-active'));
    const drop = fire(span, 'drop', transfer({ 'text/uri-list': 'file:///tmp/a.smi\nfile:///tmp/b.smi\nfile:///tmp/a.smi' }));
    assert(drop.defaultPrevented);
    await settle();
    assert.deepEqual(plain(received), [['/tmp/a.smi', '/tmp/b.smi']], 'multiple files are kept and duplicates removed');
    assert(!zone.classList.contains('path-drop-active'));
    fire(d.querySelector('text'), 'drop', transfer({ CodeFiles: JSON.stringify(['/tmp/c.smi']) }));
    await settle();
    assert.deepEqual(plain(received.at(-1)), ['/tmp/c.smi'], 'svg children resolve to the zone as well');

    // Disabled zones ignore drags and drops.
    zone.disabled = true;
    const ignored = fire(span, 'dragover', data);
    assert(ignored.defaultPrevented, "the page guard prevents navigation even on disabled zones");
    assert(!zone.classList.contains('path-drop-active'));
    zone.disabled = false;

    // Reinstalling replaces the previous handlers instead of stacking them.
    const second = [];
    w.installFileDropZone(zone, { kind: 'file', onDrop: paths => second.push(paths) });
    fire(zone, 'drop', transfer({ 'text/uri-list': 'file:///tmp/a.smi\nfile:///tmp/b.smi' }));
    await settle();
    assert.equal(received.length, 2);
    assert.deepEqual(plain(second), [['/tmp/a.smi']], 'single-file zones keep only the first path');
  }

  // Upload fallback: no path + File object -> uploadDataset(); Explorer paths are never uploaded.
  {
    const uploads = [];
    const { w, d, fire } = page('<div class="path"><input name="dataset" data-path-kind="file"><button type="button">Browse</button></div><div class="path"><input name="outputDir" data-path-kind="folder"><button type="button">Browse</button></div><button id="result">Result</button><p id="resultInfo"></p><button id="configZone">Config</button>', {
      before(window) {
        window.uploadDataset = async (file, progress) => { uploads.push(file.name); progress(50); progress(100); return '/storage/datasets/' + file.name; };
      }
    });
    const [datasetInput, folderInput] = d.querySelectorAll('input');
    const [datasetZone, folderZone] = d.querySelectorAll('.file-drop-zone');
    assert.equal(folderZone.textContent, 'Drop a folder here or click to Browse');
    const changes = [];
    datasetInput.addEventListener('change', () => changes.push(datasetInput.value));

    fire(datasetZone, 'drop', transfer({}, [fakeFile('external.msds', 'abc')]));
    await settle();
    assert.deepEqual(uploads, ['external.msds']);
    assert.equal(datasetInput.value, '/storage/datasets/external.msds');
    assert.deepEqual(changes, ['/storage/datasets/external.msds']);

    fire(datasetZone, 'drop', transfer({ 'text/uri-list': 'file:///models/huge-checkpoint.pt' }, [fakeFile('huge-checkpoint.pt', 'x')]));
    await settle();
    assert.deepEqual(uploads, ['external.msds'], 'a path from VS Code is used directly, never uploaded');
    assert.equal(datasetInput.value, '/models/huge-checkpoint.pt');

    // A folder cannot be uploaded; the user is told why.
    fire(folderZone, 'drop', transfer({}, [fakeFile('folder', '')]));
    await settle();
    assert.deepEqual(uploads, ['external.msds']);
    assert.equal(folderInput.value, '');
    assert.match(folderZone.parentElement.querySelector('.path-drop-error').textContent, /A folder is required/);

    // requirePath zones (Reaction Pattern Analysis results) refuse uploads and explain.
    const opened = [];
    w.installFileDropZone(d.getElementById('result'), {
      kind: 'file', requirePath: true,
      requirePathMessage: 'Reaction Pattern Analysis result must be dropped from VS Code Explorer so its companion compounds/ and details/ directories remain accessible.',
      onDrop: paths => opened.push(paths[0]),
      onError: message => { d.getElementById('resultInfo').textContent = message; }
    });
    fire(d.getElementById('result'), 'drop', transfer({}, [fakeFile('result.reactionpatternanalysis.pft', '{}')]));
    await settle();
    assert.deepEqual(uploads, ['external.msds']);
    assert.deepEqual(opened, []);
    assert.match(d.getElementById('resultInfo').textContent, /compounds\/ and details\//);
    fire(d.getElementById('result'), 'drop', transfer({ 'text/uri-list': 'file:///runs/a/result.reactionpatternanalysis.pft' }));
    await settle();
    assert.deepEqual(opened, ['/runs/a/result.reactionpatternanalysis.pft'], 'the original path keeps its sibling directories');

    // Configuration zones read small files directly instead of uploading them.
    const loaded = [];
    w.installFileDropZone(d.getElementById('configZone'), { kind: 'file', upload: false, onDrop: paths => loaded.push(['path', paths[0]]), onFiles: async files => loaded.push(['json', await files[0].text()]) });
    fire(d.getElementById('configZone'), 'drop', transfer({}, [fakeFile('config.pft', '{"a":1}')]));
    await settle();
    fire(d.getElementById('configZone'), 'drop', transfer({ 'text/uri-list': 'file:///cfg/run.pft' }));
    await settle();
    assert.deepEqual(loaded, [['json', '{"a":1}'], ['path', '/cfg/run.pft']]);
    assert.deepEqual(uploads, ['external.msds']);

    // Multiple uploads keep every file.
    const multi = [];
    const zone = d.createElement('div'); d.body.append(zone);
    w.installFileDropZone(zone, { kind: 'file', multiple: true, onDrop: (paths, context) => multi.push([paths, context.uploaded]) });
    fire(zone, 'drop', transfer({}, [fakeFile('train1.smi', 'C'), fakeFile('train2.smi', 'CC')]));
    await settle();
    assert.deepEqual(plain(multi), [[['/storage/datasets/train1.smi', '/storage/datasets/train2.smi'], true]]);

    // Upload failures are surfaced.
    w.uploadDataset = async () => { throw new Error('Dataset upload timed out. Try Browse.'); };
    fire(datasetZone, 'drop', transfer({}, [fakeFile('slow.msds', 'abc')]));
    await settle();
    assert.match(datasetZone.parentElement.querySelector('.path-drop-error').textContent, /timed out/);
    assert.equal(datasetInput.value, '/models/huge-checkpoint.pt');
  }

  // Drops on Workbench fields replay the same message that Browse produces.
  {
    const { w, d, fire } = page('<div class="path"><input name="input" data-path-kind="file"><button type="button" data-pick="input" data-kind="file" data-form="data">Browse</button></div><button id="mol" data-mol-files="trainSmiles">Drop</button>');
    const messages = [];
    w.addEventListener('message', event => messages.push(event.data));
    fire(d.querySelector('button'), 'drop', transfer({ 'text/uri-list': 'file:///data/train.msds' }));
    await settle();
    assert.deepEqual(plain(messages), [{ type: 'picked', form: 'data', field: 'input', value: '/data/train.msds' }]);
    fire(d.querySelector('input'), 'drop', transfer({ 'text/uri-list': 'file:///data/val.msds' }));
    await settle();
    assert.deepEqual(plain(messages.at(-1)), { type: 'picked', form: 'data', field: 'input', value: '/data/val.msds' });
  }

  // Host validation: fs.stat() decides file vs folder, not the UI label.
  {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fredis-path-drop-'));
    const file = path.join(tmp, 'data.msds'), folder = path.join(tmp, 'prepared');
    fs.writeFileSync(file, 'x'); fs.mkdirSync(folder);
    let handler = null;
    const replies = [];
    let bridge = null;
    const panel = { webview: { onDidReceiveMessage: fn => { handler = fn; }, postMessage: message => { replies.push(message); bridge?.(message); return Promise.resolve(true); } } };
    pathDrop.attach(panel);
    const ask = async message => { replies.length = 0; await handler(message); return replies[0]; };
    assert.deepEqual(await ask({ type: 'pathDrop/stat', requestId: 'a', kind: 'file', paths: [file, file] }), { type: 'pathDrop/statResponse', requestId: 'a', paths: [file] });
    assert.deepEqual(await ask({ type: 'pathDrop/stat', requestId: 'b', kind: 'folder', paths: [folder] }), { type: 'pathDrop/statResponse', requestId: 'b', paths: [folder] });
    assert.match((await ask({ type: 'pathDrop/stat', requestId: 'c', kind: 'folder', paths: [file] })).error, /A folder is required, but a file was dropped/);
    assert.match((await ask({ type: 'pathDrop/stat', requestId: 'd', kind: 'file', paths: [folder] })).error, /A file is required, but a folder was dropped/);
    assert.match((await ask({ type: 'pathDrop/stat', requestId: 'e', kind: 'file', paths: [path.join(tmp, 'missing.msds')] })).error, /ENOENT/);
    assert.match((await ask({ type: 'pathDrop/stat', requestId: 'f', kind: 'other', paths: [file] })).error, /Invalid dropped path kind/);
    assert.equal(await ask({ type: 'unrelated' }), undefined);

    // Webview <-> host round trip through the real attach() handler.
    const { w, d, fire, posted } = page('<div class="path"><input name="trainDir" data-path-kind="folder"><button type="button">Browse</button></div><div class="path"><input name="checkpoint" data-path-kind="file"><button type="button">Browse</button></div>', { host: true });
    bridge = reply => w.dispatchEvent(new w.MessageEvent('message', { data: reply }));
    w.onHostMessage = message => { if (message.type === 'pathDrop/stat') handler(message); };
    const [folderInput, fileInput] = d.querySelectorAll('input');
    const [folderZone, fileZone] = d.querySelectorAll('.file-drop-zone');
    const uri = value => transfer({ 'text/uri-list': 'file://' + value.split(path.sep).map(encodeURIComponent).join('/') });
    fire(folderZone, 'drop', uri(folder));
    await settle(10);
    assert.deepEqual(plain(posted.filter(m => m.type === 'pathDrop/stat').map(m => [m.kind, m.paths])), [['folder', [folder]]]);
    assert.equal(folderInput.value, folder);
    fire(folderZone, 'drop', uri(file));
    await settle(10);
    assert.equal(folderInput.value, folder, 'a file dropped on a folder field is rejected');
    assert.match(folderZone.parentElement.querySelector('.path-drop-error').textContent, /A folder is required, but a file was dropped/);
    fire(fileZone, 'drop', uri(folder));
    await settle(10);
    assert.equal(fileInput.value, '');
    assert.match(fileZone.parentElement.querySelector('.path-drop-error').textContent, /A file is required, but a folder was dropped/);
    fire(fileZone, 'drop', uri(file));
    await settle(10);
    assert.equal(fileInput.value, file);
    assert.equal(fileZone.parentElement.querySelector('.path-drop-error'), null);
    fs.rmSync(tmp, { recursive: true, force: true });
  }

  // dataset-upload.js host protocol: uploadStart / uploadChunk / uploadFinish / uploadCancel.
  {
    const storage = fs.mkdtempSync(path.join(os.tmpdir(), 'fredis-upload-'));
    let handler = null, dispose = null;
    const replies = [];
    const panel = {
      onDidDispose: fn => { dispose = fn; },
      webview: { onDidReceiveMessage: fn => { handler = fn; }, postMessage: message => { replies.push(message); bridge?.(message); return Promise.resolve(true); } }
    };
    let bridge = null;
    datasetUpload.attach(panel, { storageUri: { fsPath: storage } });
    const ask = async message => { replies.length = 0; await handler(message); return replies[0]; };
    const content = Buffer.from('SMILES\tPrecursorMZ\nCCO\t47.05\n');
    const start = await ask({ type: 'dataset/uploadStart', requestId: '1', name: '../dropped.TSV', size: content.length });
    assert.equal(start.type, 'dataset/uploadResponse');
    assert.equal(start.requestId, '1');
    assert(start.id);
    const first = await ask({ type: 'dataset/uploadChunk', requestId: '2', id: start.id, data: content.subarray(0, 10).toString('base64') });
    assert.equal(first.offset, 10);
    const incomplete = await ask({ type: 'dataset/uploadFinish', requestId: '3', id: start.id });
    assert.match(incomplete.error, /incomplete/);
    assert.match((await ask({ type: 'dataset/uploadChunk', requestId: '4', id: start.id, data: 'AA==' })).error, /expired/, 'a failed upload is discarded');

    const again = await ask({ type: 'dataset/uploadStart', requestId: '5', name: 'dropped.tsv', size: content.length });
    await ask({ type: 'dataset/uploadChunk', requestId: '6', id: again.id, data: content.subarray(0, 10).toString('base64') });
    await ask({ type: 'dataset/uploadChunk', requestId: '7', id: again.id, data: content.subarray(10).toString('base64') });
    const finished = await ask({ type: 'dataset/uploadFinish', requestId: '8', id: again.id });
    assert(finished.path.startsWith(path.join(storage, 'datasets')));
    assert.equal(path.extname(finished.path), '.tsv');
    assert.deepEqual(fs.readFileSync(finished.path), content);

    const cancelled = await ask({ type: 'dataset/uploadStart', requestId: '9', name: 'cancel.msds', size: 4 });
    const cancelledFile = fs.readdirSync(path.join(storage, 'datasets')).find(name => name.startsWith(cancelled.id));
    assert(cancelledFile);
    assert.deepEqual(await ask({ type: 'dataset/uploadCancel', requestId: '10', id: cancelled.id }), { type: 'dataset/uploadResponse', requestId: '10' });
    assert(!fs.existsSync(path.join(storage, 'datasets', cancelledFile)), 'cancel removes the partial file');

    const tooLarge = await ask({ type: 'dataset/uploadStart', requestId: '11', name: 'big.msds', size: 1 });
    assert.match((await ask({ type: 'dataset/uploadChunk', requestId: '12', id: tooLarge.id, data: Buffer.from('abc').toString('base64') })).error, /exceeds the declared file size/);
    assert.match((await ask({ type: 'dataset/uploadStart', requestId: '13', name: 'x.msds', size: -1 })).error, /Invalid dataset file size/);
    assert.match((await ask({ type: 'dataset/uploadChunk', requestId: '14', id: 'missing', data: '' })).error, /expired/);
    assert.equal(await ask({ type: 'pathDrop/stat' }), undefined, 'unrelated messages are ignored');

    // Webview client -> host round trip: a drop without path uploads the File and uses the stored path.
    const dom = new JSDOM(`<!doctype html><body><div class="path"><input name="dataset" data-path-kind="file"><button type="button">Browse</button></div><script>const vscode=acquireVsCodeApi();${script({ host: true })}${datasetUpload.script()}</script></body>`, {
      runScripts: 'dangerously',
      beforeParse(window) {
        window.acquireVsCodeApi = () => ({
          postMessage: message => { handler(message); }
        });
      }
    });
    const w = dom.window, d = w.document;
    bridge = reply => w.dispatchEvent(new w.MessageEvent('message', { data: reply }));
    assert.equal(typeof w.uploadDataset, 'function');
    const drop = new w.Event('drop', { bubbles: true, cancelable: true });
    const progress = [];
    Object.defineProperty(drop, 'dataTransfer', { value: transfer({}, [fakeFile('from-os.msds', 'MSDS-CONTENT')]) });
    d.querySelector('input').addEventListener('change', event => progress.push(event.target.value));
    d.querySelector('.file-drop-zone').dispatchEvent(drop);
    // Each chunk is written with real file I/O, so wait on wall-clock time.
    for (const deadline = Date.now() + 5000; !progress.length && Date.now() < deadline;) await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(d.querySelector('.path-drop-error'), null);
    assert.equal(progress.length, 1);
    assert(progress[0].startsWith(path.join(storage, 'datasets')));
    assert.equal(fs.readFileSync(progress[0], 'utf8'), 'MSDS-CONTENT');
    dom.window.close();
    dispose();
    fs.rmSync(storage, { recursive: true, force: true });
  }

  console.log('Path drag-and-drop checks passed.');
})().catch(error => { console.error(error); process.exit(1); });
