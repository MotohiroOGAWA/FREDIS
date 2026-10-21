const fs = require('fs');

function pathsFromDroppedValue(value) {
  const paths = [];
  const add = candidate => {
    if (candidate && !paths.includes(candidate)) paths.push(candidate);
  };
  const visit = current => {
    if (current === undefined || current === null) return;
    if (Array.isArray(current)) { current.forEach(visit); return; }
    if (typeof current === 'object') {
      for (const key of ['fsPath', 'external', 'uri', 'resourceUri', 'resource', 'resources', 'uris', 'files', 'path']) {
        if (current[key] !== undefined) visit(current[key]);
      }
      return;
    }
    const candidate = String(current).trim().replace(/^['"]|['"]$/g, '');
    if (!candidate) return;
    if (/^[\[{]/.test(candidate)) {
      try { visit(JSON.parse(candidate)); return; } catch (_) {}
    }
    if (/^(?:file|vscode-remote):\/\//i.test(candidate)) {
      try {
        const uri = new URL(candidate);
        let pathname = decodeURIComponent(uri.pathname);
        if (uri.protocol === 'file:' && uri.host && uri.host !== 'localhost') pathname = `//${uri.host}${pathname}`;
        if (/^\/[A-Za-z]:\//.test(pathname)) pathname = pathname.slice(1);
        add(pathname);
      } catch (_) {}
      return;
    }
    if (/^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(candidate)) add(candidate);
  };
  visit(value);
  return paths;
}

function pathFromDroppedValue(value) {
  return pathsFromDroppedValue(value)[0] || '';
}

function pathsFromDataTransfer(transfer) {
  if (!transfer) return [];
  const paths = [];
  const add = values => values.forEach(value => { if (value && !paths.includes(value)) paths.push(value); });
  const read = type => {
    try { return transfer.getData(type) || ''; } catch (_) { return ''; }
  };
  const addTransferValue = type => {
    const value = read(type);
    for (const line of String(value).split(/\r?\n/)) {
      if (!line.trim() || line.trim().startsWith('#')) continue;
      add(pathsFromDroppedValue(line));
    }
  };

  // VS Code URI formats preserve original local/remote paths, so they take
  // precedence over Electron's File.path and Webview File objects.
  for (const type of [
    'application/vnd.code.uri-list',
    'text/uri-list',
    'ResourceURLs', 'resourceurls',
    'CodeFiles', 'codefiles',
    'CodeEditors', 'codeeditors',
    'text/x-moz-url',
    'text/plain'
  ]) addTransferValue(type);
  for (const type of Array.from(transfer.types || [])) addTransferValue(type);
  for (const file of Array.from(transfer.files || [])) if (file && file.path) add([String(file.path)]);
  return paths;
}

function pathFromDataTransfer(transfer) {
  return pathsFromDataTransfer(transfer)[0] || '';
}

const PATH_DROP_READ_ERROR = 'Unable to read the dropped file. Try dropping it from the VS Code Explorer or use Browse.';

// Browse and Drop share this final step so every downstream listener runs.
function applyPickedPath(input, value) {
  if (!input) return;
  input.setCustomValidity?.('');
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

// Replays the message the extension host would send after Browse.
function dispatchPathPicked(data) {
  window.dispatchEvent(new MessageEvent('message', { data }));
}

function pathDropStatus(element, message) {
  const host = element.closest?.('.path') || element.parentElement;
  if (!host) return;
  let status = host.querySelector(':scope > .path-drop-error');
  if (!message) { status?.remove(); return; }
  if (!status) {
    status = document.createElement('small');
    status.className = 'path-drop-error';
    status.setAttribute('role', 'alert');
    host.append(status);
  }
  status.textContent = message;
}

function pathDropError(element, error, options = {}) {
  const message = error?.message || String(error || '') || PATH_DROP_READ_ERROR;
  if (options.onError) options.onError(message, error);
  else {
    if (options.input?.setCustomValidity) {
      options.input.setCustomValidity(message);
      options.input.reportValidity?.();
    }
    pathDropStatus(element, message);
  }
  element.dispatchEvent(new CustomEvent('pathdrop:error', { bubbles: true, detail: { message } }));
}

// Asks the extension host to fs.stat() dropped paths. Only webviews whose
// host called attach() set window.fredisPathDropHost; window.validateDroppedPaths
// replaces the round trip (used by the jsdom checks).
function checkDroppedPaths(paths, kind) {
  if (!paths.length || !kind) return Promise.resolve(paths);
  if (typeof window.validateDroppedPaths === 'function') return Promise.resolve(window.validateDroppedPaths(paths, kind));
  if (!window.fredisPathDropHost || typeof vscode === 'undefined') return Promise.resolve(paths);
  const state = window.__fredisPathStat || (window.__fredisPathStat = { sequence: 0, pending: new Map() });
  if (!state.listening) {
    state.listening = true;
    window.addEventListener('message', event => {
      const message = event.data;
      if (message?.type !== 'pathDrop/statResponse') return;
      const entry = state.pending.get(message.requestId);
      if (!entry) return;
      state.pending.delete(message.requestId);
      clearTimeout(entry.timer);
      message.error ? entry.reject(new Error(message.error)) : entry.resolve(message.paths || []);
    });
  }
  return new Promise((resolve, reject) => {
    const requestId = 'path-drop-' + (++state.sequence);
    const timer = setTimeout(() => { state.pending.delete(requestId); reject(new Error('Timed out while checking the dropped path. Use Browse.')); }, 15000);
    state.pending.set(requestId, { resolve, reject, timer });
    vscode.postMessage({ type: 'pathDrop/stat', requestId, paths, kind });
  });
}

function installFileDropZone(element, options = {}) {
  if (!element) return () => {};
  const registry = window.__fredisPathDropZones || (window.__fredisPathDropZones = new WeakMap());
  registry.get(element)?.();
  // Lets installPathDrop() skip zones a feature has already configured.
  element.dataset.pathDropInstalled = 'true';
  const kind = options.kind || element.dataset.pathKind || 'file';
  const multiple = options.multiple === true;
  const listeners = [];
  // relatedTarget is null in some Chromium drag events, so a depth counter
  // keeps the highlight steady while the pointer crosses child elements.
  let depth = 0;
  const listen = (type, listener) => { element.addEventListener(type, listener); listeners.push([type, listener]); };
  const highlight = value => {
    if (!value) depth = 0;
    element.classList.toggle('path-drop-active', value);
  };
  const accepts = event => !element.disabled && options.disabled !== true && (!options.requireShift || event.shiftKey);
  listen('dragenter', event => {
    if (!accepts(event)) { highlight(false); return; }
    event.preventDefault();
    if (!event.shiftKey) event.stopPropagation();
    depth++;
    highlight(true);
  });
  listen('dragover', event => {
    if (!accepts(event)) { highlight(false); return; }
    event.preventDefault();
    if (!event.shiftKey) event.stopPropagation();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
    highlight(true);
  });
  listen('dragleave', event => {
    if (!element.classList.contains('path-drop-active')) return;
    event.stopPropagation();
    if (event.relatedTarget && element.contains(event.relatedTarget)) return;
    if (event.relatedTarget || --depth <= 0) highlight(false);
  });
  listen('dragend', () => highlight(false));
  listen('drop', async event => {
    if (!accepts(event)) { highlight(false); return; }
    event.preventDefault();
    event.stopPropagation();
    highlight(false);
    try {
      const transfer = event.dataTransfer;
      let paths = pathsFromDataTransfer(transfer);
      const files = Array.from(transfer?.files || []);
      let uploaded = false;
      if (paths.length) {
        // Explorer paths are used in place: large checkpoints and results
        // with companion directories must never be copied.
        paths = await checkDroppedPaths(paths, kind);
      } else if (files.length && kind === 'folder') {
        throw new Error('A folder is required. Drop it from the VS Code Explorer or use Browse.');
      } else if (files.length && options.requirePath) {
        throw new Error(options.requirePathMessage || 'This file must be dropped from the VS Code Explorer so its companion files remain accessible.');
      } else if (files.length && options.onFiles) {
        await options.onFiles(multiple ? files : files.slice(0, 1), { event, element, kind });
        pathDropStatus(element, '');
        return;
      } else if (files.length && options.upload !== false && typeof window.uploadDataset === 'function') {
        const selected = multiple ? files : files.slice(0, 1);
        paths = [];
        for (let index = 0; index < selected.length; index++) {
          const file = selected[index];
          const uploadedPath = await window.uploadDataset(file, percent => options.onProgress?.(percent, file, index, selected.length));
          if (uploadedPath && !paths.includes(uploadedPath)) paths.push(uploadedPath);
        }
        uploaded = true;
      }
      if (!paths.length) throw new Error(PATH_DROP_READ_ERROR);
      if (!multiple) paths = paths.slice(0, 1);
      pathDropStatus(element, '');
      if (options.onDrop) await options.onDrop(paths, { event, element, files, uploaded, kind });
      else if (options.input) applyPickedPath(options.input, paths[0]);
    } catch (error) {
      pathDropError(element, error, options);
    } finally {
      highlight(false);
      options.onFinally?.();
    }
  });
  const uninstall = () => {
    for (const [type, listener] of listeners) element.removeEventListener(type, listener);
    highlight(false);
    if (registry.get(element) === uninstall) registry.delete(element);
  };
  registry.set(element, uninstall);
  return uninstall;
}

// Builds the message the host sends when the sibling Browse button picks a
// path, so a drop follows exactly the same downstream path.
function pickedMessageFor(input, picker, value) {
  const data = picker?.dataset || {};
  if (data.pick && data.form) return { type: 'picked', form: data.form, field: data.pick, value };
  if (data.predictBatchPick) return { type: 'predictBatchPicked', field: data.predictBatchPick, value };
  if (data.libraryPick) return { type: 'library/picked', target: data.libraryPick, path: value };
  return null;
}

function installPathInputDrop(input, options = {}) {
  if (!input) return () => {};
  const element = options.element || input;
  return installFileDropZone(element, {
    ...options,
    input,
    kind: options.kind || input.dataset.pathKind || 'file',
    multiple: false,
    onDrop: async (paths, context) => {
      const message = pickedMessageFor(input, options.picker, paths[0]);
      if (message) dispatchPathPicked(message);
      else applyPickedPath(input, paths[0]);
      await options.onDrop?.(paths, context);
    }
  });
}

// VS Code's Webview preload watches events on window. An external file may
// first enter a heading or whitespace before reaching a feature's drop zone.
// Claim it at document level so that preload cannot disable the entire iframe.
// Target handlers still run; internal text/chemical drags are left alone.
function installFileDropGuard(doc = document) {
  if (doc.__fredisFileDropGuard) return;
  doc.__fredisFileDropGuard = true;
  const isFile = event => {
    const transfer = event.dataTransfer;
    return Array.from(transfer?.items || []).some(item => item.kind === 'file') ||
      Array.from(transfer?.types || []).some(type => /^(files|text\/uri-list|application\/vnd\.code\.uri-list|resourceurls|codefiles|codeeditors|application\/vnd\.code\.tree\.)/i.test(type));
  };
  for (const type of ['dragenter', 'dragover', 'drop']) {
    doc.addEventListener(type, event => {
      if (!isFile(event)) return;
      event.preventDefault();
      if (type === 'dragover' && event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
    }, true);
    doc.addEventListener(type, event => {
      // Preserve Shift-held hover forwarding to VS Code; handle drops locally.
      if (isFile(event) && (type === 'drop' || !event.shiftKey)) event.stopPropagation();
    });
  }
}

function installPathDrop(options = {}) {
  window.installFileDropGuard = installFileDropGuard;
  installFileDropGuard(document);
  function decorate() {
    if (!window.document?.body) return;
    // Explorer drags are intercepted before entering the webview. Shift is
    // VS Code's bypass; a DOM event guard cannot remove the host's overlay.
    if (!document.getElementById('fredisExplorerDropHint')) {
      const hint = document.createElement('p');
      hint.id = 'fredisExplorerDropHint';
      hint.className = 'path-drop-hint';
      hint.setAttribute('role', 'note');
      hint.textContent = 'VS Code Explorer: hold Shift while dropping a file or folder. Release Shift after the mouse button.';
      (document.querySelector('main') || document.body).prepend(hint);
    }
    for (const button of document.querySelectorAll('button')) {
      if (!/^Browse[.…]*$/i.test(button.textContent.trim())) continue;
      const host = button.closest('.path') || button.parentElement;
      const input = host?.querySelector('input:not([type=checkbox]):not([type=radio]):not([type=number])');
      if (input && !input.dataset.pathKind) input.dataset.pathKind = /dir|folder|run/i.test(input.name || input.id) ? 'folder' : 'file';
    }
    for (const input of document.querySelectorAll('input[data-path-kind]')) {
      const host = input.closest('.path') || input.parentElement;
      const picker = host && [...host.querySelectorAll('button')].find(button => /browse/i.test(button.textContent));
      if (!input.dataset.pathDropInstalled) {
        input.dataset.pathDropInstalled = 'true';
        installPathInputDrop(input, { picker, requireShift: options.compact === true });
      }
      if (!picker || picker.dataset.pathDropInstalled) continue;
      picker.dataset.pathDropInstalled = 'true';
      if (options.compact) {
        picker.title = 'Browse files, or hold Shift and drop a file onto the path field';
        installPathInputDrop(input, { element: picker, picker, requireShift: true });
        continue;
      }
      picker.dataset.pathDropZone = '';
      picker.classList.add('file-drop-zone');
      if (/^Browse[.…]*$/i.test(picker.textContent.trim())) picker.textContent = `Drop ${input.dataset.pathKind === 'folder' ? 'a folder' : 'a file'} here or click to Browse`;
      installPathInputDrop(input, { element: picker, picker });
    }
  }
  window.pathsFromDroppedValue = pathsFromDroppedValue;
  window.pathFromDroppedValue = pathFromDroppedValue;
  window.pathsFromDataTransfer = pathsFromDataTransfer;
  window.pathFromDataTransfer = pathFromDataTransfer;
  window.applyPickedPath = applyPickedPath;
  window.dispatchPathPicked = dispatchPathPicked;
  window.installFileDropZone = installFileDropZone;
  window.installPathInputDrop = installPathInputDrop;
  decorate();
  new MutationObserver(decorate).observe(document.body, { childList: true, subtree: true });
}

function attach(panel) {
  panel.webview.onDidReceiveMessage(async message => {
    if (message?.type !== 'pathDrop/stat') return;
    const reply = data => panel.webview.postMessage({ type: 'pathDrop/statResponse', requestId: message.requestId, ...data });
    try {
      const kind = message.kind;
      if (!['file', 'folder'].includes(kind)) throw new Error('Invalid dropped path kind.');
      if (!Array.isArray(message.paths) || !message.paths.length) throw new Error('No dropped path was provided.');
      const paths = [];
      for (const value of message.paths) {
        const candidate = String(value || '');
        const stat = await fs.promises.stat(candidate);
        if (kind === 'file' && !stat.isFile()) throw new Error(`A file is required, but a folder was dropped: ${candidate}`);
        if (kind === 'folder' && !stat.isDirectory()) throw new Error(`A folder is required, but a file was dropped: ${candidate}`);
        if (!paths.includes(candidate)) paths.push(candidate);
      }
      reply({ paths });
    } catch (error) {
      reply({ error: error.message });
    }
  });
}

function script(options = {}) {
  return [
    pathsFromDroppedValue,
    pathFromDroppedValue,
    pathsFromDataTransfer,
    pathFromDataTransfer,
    applyPickedPath,
    dispatchPathPicked,
    pathDropStatus,
    pathDropError,
    checkDroppedPaths,
    installFileDropZone,
    pickedMessageFor,
    installPathInputDrop,
    installFileDropGuard
  ].map(value => value.toString()).join('') + `const PATH_DROP_READ_ERROR=${JSON.stringify(PATH_DROP_READ_ERROR)};window.fredisPathDropHost=${options.host === true};(${installPathDrop.toString()})(${JSON.stringify({compact: options.compact === true})});`;
}

function css() {
  return '.path-drop-hint{font-size:12px;color:var(--vscode-descriptionForeground);margin:8px 0 12px;line-height:1.5}.file-drop-zone{max-width:100%;min-width:0;white-space:normal;border-style:dashed}.file-drop-zone.path-drop-active{border-color:var(--vscode-focusBorder,#36c5a2)!important;box-shadow:0 0 0 1px var(--vscode-focusBorder,#36c5a2);background:color-mix(in srgb,var(--vscode-focusBorder,#36c5a2) 12%,var(--vscode-button-secondaryBackground))}.path:has(.file-drop-zone){flex-wrap:wrap}.path .file-drop-zone{flex-basis:100%;padding:14px}.path-drop-error{flex-basis:100%;color:var(--vscode-errorForeground,#f48771)}input[data-path-kind]{transition:border-color .12s,box-shadow .12s,background-color .12s}input[data-path-kind].path-drop-active{border-color:var(--vscode-focusBorder,#36c5a2)!important;box-shadow:0 0 0 1px var(--vscode-focusBorder,#36c5a2);background:color-mix(in srgb,var(--vscode-focusBorder,#36c5a2) 12%,var(--vscode-input-background))}';
}

module.exports = {
  installFileDropGuard,
  pathsFromDroppedValue,
  pathFromDroppedValue,
  pathsFromDataTransfer,
  pathFromDataTransfer,
  applyPickedPath,
  pickedMessageFor,
  installFileDropZone,
  installPathInputDrop,
  attach,
  script,
  css
};
