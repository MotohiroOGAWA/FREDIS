/* Extension-host primitives shared by the Workbench reaction-pattern tab and the standalone
   .reaction.pft / .reactionset.pft custom editors, so both talk to the same
   filesystem, RDKit backend and periodic-table picker instead of drifting apart. */
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const periodicTable = require('./periodic-table');
const reactionService = require('./reaction-service');

function isFredisRoot(candidate) {
  return fs.existsSync(path.join(candidate, 'fredis')) && fs.existsSync(path.join(candidate, 'pyproject.toml'));
}

function projectRoot(context) {
  const configured = vscode.workspace.getConfiguration('fredis').get('applicationRoot', '').trim();
  const workspaceFolders = vscode.workspace.workspaceFolders || [];
  if (configured) {
    const base = workspaceFolders[0] ? workspaceFolders[0].uri.fsPath : process.cwd();
    const candidate = path.resolve(base, configured);
    if (isFredisRoot(candidate)) return candidate;
    throw new Error(`fredis.applicationRoot does not point to a FREDIS application: ${configured}`);
  }
  const searchRoots = [
    ...workspaceFolders.map(folder => folder.uri.fsPath),
    context.extensionPath
  ];
  for (const searchRoot of searchRoots) {
    let current = searchRoot;
    while (true) {
      for (const candidate of [current, path.join(current, 'mnt', 'app'), path.join(current, 'app')]) {
        if (isFredisRoot(candidate)) return candidate;
      }
      const parent = path.dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  throw new Error('FREDIS application root was not found. Open the FREDIS workspace or set fredis.applicationRoot to a workspace-relative path.');
}

function runChemistryBackend(context, command, payload, owner = context) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'reaction-pattern-set', 'backend.py');
    let root;
    try { root = projectRoot(context); }
    catch (error) { reject(error); return; }
    if (!isFredisRoot(root)) { reject(new Error('The detected working directory is not a FREDIS application.')); return; }
    if (command === 'reactionPreview' || command === 'reactionPreviewProducts' || command === 'reactionExplore' || command === 'fragmentTree') {
      reactionService.forOwner(owner, { python, script, root }, context).request(command, payload).then(resolve, reject);
      return;
    }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', reject);
    child.on('close', () => {
      try { const response = JSON.parse(stdout); response.ok ? resolve(response.result) : reject(new Error(response.error)); }
      catch (_) { reject(new Error(stderr || stdout || 'The RDKit backend did not return a response.')); }
    });
    child.stdin.end(JSON.stringify({ command, payload }));
  });
}

function showElementPicker(initialElements) {
  return new Promise(resolve => {
    const panel = vscode.window.createWebviewPanel('fredis.periodicTable', 'Select Elements', vscode.ViewColumn.Active, { enableScripts: true, retainContextWhenHidden: false });
    const selected = new Set(Array.isArray(initialElements) ? initialElements : []);
    const rows = periodicTable.rows;
    panel.webview.html = `<!doctype html><html><head><meta charset="UTF-8"><style>
      :root{color-scheme:light dark}*{box-sizing:border-box}body{margin:0;padding:24px;color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);font-family:var(--vscode-font-family)}h1{font-size:22px;margin:0 0 6px}.muted{opacity:.65;margin:0 0 20px}.table{display:grid;grid-template-columns:repeat(18,minmax(36px,1fr));gap:4px;min-width:760px}.element,.gap{aspect-ratio:1}.element{padding:0;border:1px solid var(--vscode-input-border,#777);border-radius:5px;color:inherit;background:var(--vscode-button-secondaryBackground);font-weight:600;cursor:pointer}.element:hover{border-color:var(--vscode-focusBorder)}.element.selected{color:var(--vscode-button-foreground);background:var(--vscode-button-background);outline:2px solid var(--vscode-focusBorder);outline-offset:1px}.series-gap{margin-top:14px}.actions{position:sticky;bottom:0;display:flex;justify-content:flex-end;gap:8px;padding-top:20px;background:var(--vscode-editor-background)}.actions button{padding:8px 16px;border:1px solid var(--vscode-input-border,#777);border-radius:5px;color:inherit;background:var(--vscode-button-secondaryBackground)}.actions .primary{color:var(--vscode-button-foreground);background:var(--vscode-button-background)}</style></head><body><h1>Periodic Table</h1><p class="muted">Choose one or more elements. The current query is preselected.</p><div class="table">${rows.flatMap((row,ri)=>row.map(symbol=>symbol?`<button class="element${selected.has(symbol)?' selected':''}${ri===7?' series-gap':''}" data-element="${symbol}">${symbol}</button>`:`<span class="gap${ri===7?' series-gap':''}"></span>`)).join('')}</div><div class="actions"><button id="cancel">Cancel</button><button id="apply" class="primary">Apply selection</button></div><script>const vscode=acquireVsCodeApi(),chosen=new Set(${JSON.stringify([...selected])});document.querySelector('.table').onclick=e=>{const b=e.target.closest('[data-element]');if(!b)return;chosen.has(b.dataset.element)?chosen.delete(b.dataset.element):chosen.add(b.dataset.element);b.classList.toggle('selected',chosen.has(b.dataset.element))};document.getElementById('apply').onclick=()=>vscode.postMessage({type:'apply',elements:[...chosen]});document.getElementById('cancel').onclick=()=>vscode.postMessage({type:'cancel'});</script></body></html>`;
    let completed = false;
    panel.webview.onDidReceiveMessage(message => {
      if (message.type === 'apply') { completed = true; resolve(message.elements); panel.dispose(); }
      if (message.type === 'cancel') panel.dispose();
    });
    panel.onDidDispose(() => { if (!completed) resolve(null); });
  });
}

async function readReactionPatternImport(context, message) {
  if (typeof message.json === 'string') return { value: JSON.parse(message.json), path: '' };
  let file = typeof message.path === 'string' && message.path.trim()
    ? path.resolve(projectRoot(context), message.path) : undefined;
  if (!file) {
    // Extension-agnostic: the caller (normalizeDocument/normalizeRule/
    // normalizePattern) validates the parsed content's shape, so any
    // filename works here -- unlike an Explorer double-click, which VS Code
    // itself gates by the registered customEditors extensions.
    const picked = await vscode.window.showOpenDialog({ canSelectMany: false });
    if (!picked?.length) return undefined;
    file = picked[0].fsPath;
  }
  return { value: JSON.parse(await fs.promises.readFile(file, 'utf8')), path: file };
}

function safeFileStem(value) { return String(value).trim().replace(/[^A-Za-z0-9_.-]+/g, '_') || 'pattern'; }

function ensureFileSuffix(filePath, suffix, acceptedAliases = []) {
  let result = String(filePath);
  const accepted = [suffix, ...acceptedAliases].map(value => String(value).toLowerCase());
  while (result.toLowerCase().endsWith('.json.json')) result = result.slice(0, -5);
  for (const ending of accepted) {
    while (result.toLowerCase().endsWith(ending + ending)) result = result.slice(0, -ending.length);
  }
  if (accepted.some(ending => result.toLowerCase().endsWith(ending))) return result;
  if (result.toLowerCase().endsWith('.json')) result = result.slice(0, -5);
  return `${result}${suffix}`;
}

function reactionPatternSetSavePath(name, previousPath, fileSuffix) {
  const stem = String(name || '').trim().replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/[. ]+$/g, '') || 'patterns';
  const filename = ensureFileSuffix(stem, fileSuffix);
  return previousPath ? path.join(path.dirname(previousPath), filename) : filename;
}

module.exports = {
  projectRoot, isFredisRoot, runChemistryBackend, showElementPicker, readReactionPatternImport,
  safeFileStem, ensureFileSuffix, reactionPatternSetSavePath
};
