const { filteringHtml, filteringClient } = require('./filtering');
const whiteLightTheme=require('../../light-theme').css;
const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const { groupedArgs, groupedHtml, groupedClient } = require('./grouped-boxplot');
const pathDrop = require('../../webview-path-drop');

const VIEW_TYPE = 'fredis.evaluationConfigEditor';
// .pft (not .json) so VS Code's language-icon fallback actually shows the
// FREDIS icon: virtually every icon theme already claims plain .json, which
// always wins over a contributed language icon, but no theme claims .pft.
const CONFIG_SUFFIXES = { filtering: '.evalfilter.pft', column: '.evalcol.pft', grouped: '.evalgroup.pft' };

function evaluationConfigSuffix(kind) {
  const suffix = CONFIG_SUFFIXES[kind];
  if (!suffix) throw new Error(`Unknown evaluation kind: ${kind}`);
  return suffix;
}

function ensureEvaluationConfigSuffix(filePath, kind) {
  const suffix = evaluationConfigSuffix(kind);
  let stem = filePath;
  // Strip all trailing evaluation suffixes, including names produced by
  // native save dialogs that append their selected extension automatically.
  let existingSuffix;
  while ((existingSuffix = Object.values(CONFIG_SUFFIXES).find(value => stem.toLowerCase().endsWith(value)))) {
    stem = stem.slice(0, -existingSuffix.length);
  }
  if (stem === filePath && stem.toLowerCase().endsWith('.json')) stem = stem.slice(0, -5);
  return stem + suffix;
}

function ensureSvgSuffix(filePath) {
  return filePath.toLowerCase().endsWith('.svg') ? filePath : `${filePath}.svg`;
}

function configFileStem(value, fallback) {
  const stem = String(value || fallback).trim().replace(/[^\p{L}\p{N}._-]+/gu, '-').replace(/^-+|-+$/g, '');
  return stem || fallback;
}

function runCli(context, projectRoot, args, output, onProgress) {
  return new Promise((resolve, reject) => {
    const root = projectRoot(context);
    const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
    const commandArgs = ['-m', 'evaluation', ...args];
    output.appendLine(`$ ${[python, ...commandArgs].join(' ')}`);
    const child = spawn(python, commandArgs, {
      cwd: root,
      env: { ...process.env, PYTHONPATH: ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
    });
    let stdout = '', stderr = '', stderrBuffer = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    const consumeStderrLine = line => {
      const prefix = 'FREDIS_PROGRESS ';
      if (line.startsWith(prefix)) {
        try {
          const progress = JSON.parse(line.slice(prefix.length));
          if (onProgress) onProgress(progress);
          return;
        } catch (_) {}
      }
      stderr += line + '\n';
      output.appendLine(line);
    };
    child.stderr.on('data', chunk => {
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop();
      lines.forEach(consumeStderrLine);
    });
    child.on('error', reject);
    child.on('close', code => {
      if (stderrBuffer) consumeStderrLine(stderrBuffer);
      try {
        const response = JSON.parse(stdout.trim());
        if (code === 0 && response.ok) resolve(response.result);
        else reject(new Error(response.error || stderr || `Evaluation CLI exited with code ${code}.`));
      } catch (_) {
        reject(new Error(stderr || stdout || `Evaluation CLI exited with code ${code}.`));
      }
    });
  });
}

function summaryArgs(request, outputImage) {
  const args = ['summarize', '--input', request.input, '--group-column', request.groupColumn,
    '--mode', request.mode || 'auto',
    '--width', String(request.width || 900), '--height', String(request.height || 520),
    '--color', request.color || '#36c5a2'];
  if (request.metadata) args.push('--metadata', request.metadata);
  if (request.joinColumn) args.push('--join-column', request.joinColumn);
  if (request.metadataJoinColumn) args.push('--metadata-join-column', request.metadataJoinColumn);
  args.push('--transform', request.transform || 'none');
  if (request.precursorMzColumn) args.push('--precursor-mz-column', request.precursorMzColumn);
  if (request.instrumentColumn) args.push('--instrument-column', request.instrumentColumn);
  if (request.smilesColumn) args.push('--smiles-column', request.smilesColumn);
  if (request.chemicalDescriptor) args.push('--chemical-descriptor', request.chemicalDescriptor);
  if (request.transform === 'smarts' && Array.isArray(request.smartsQueries)) args.push('--smarts-json', JSON.stringify(request.smartsQueries));
  args.push('--graph-opacity', String(request.graphOpacity ?? 1));
  args.push('--x-label-size', String(request.xLabelSize ?? 11));
  args.push('--y-label-size', String(request.yLabelSize ?? 11));
  args.push('--x-axis-title-size', String(request.xAxisTitleSize ?? 12));
  args.push('--y-axis-title-size', String(request.yAxisTitleSize ?? 12));
  args.push('--x-axis-title-gap', String(request.xAxisTitleGap ?? 8));
  args.push('--y-axis-title-gap', String(request.yAxisTitleGap ?? 8));
  if (request.showXAxisTitle === false) args.push('--hide-x-axis-title');
  if (request.showYAxisTitle === false) args.push('--hide-y-axis-title');
  args.push('--x-label-rotation', request.xLabelRotation || 'auto');
  args.push('--title-size', String(request.titleSize ?? 17));
  args.push('--title', request.title || '');
  args.push('--plot-type', request.plotType || 'box');
  if (request.mode === 'numeric' && request.bins) args.push('--bins', request.bins);
  if (Array.isArray(request.include)) args.push('--include', JSON.stringify(request.include));
  if (Array.isArray(request.order)) args.push('--order', JSON.stringify(request.order));
  if (!request.transparent) args.push('--opaque');
  if (outputImage) args.push('--output-image', outputImage);
  return args;
}

function evaluationConfigDocument(kind, config) {
  if (!['filtering', 'column', 'grouped'].includes(kind) || !config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('Invalid evaluation configuration.');
  }
  return { schema: 'fredis.evaluation.config', schemaVersion: 1, kind, config };
}

function parseEvaluationConfig(value, expectedKind) {
  if (!value || value.schema !== 'fredis.evaluation.config' || value.schemaVersion !== 1) {
    throw new Error('This is not a supported FREDIS evaluation configuration.');
  }
  if (value.kind !== expectedKind) throw new Error(`Expected a ${expectedKind} evaluation configuration, but found ${value.kind || 'unknown'}.`);
  if (!value.config || typeof value.config !== 'object' || Array.isArray(value.config)) throw new Error('Evaluation configuration payload is missing.');
  return value.config;
}

function open(context, output, projectRoot, kind = null) {
  if (kind !== null) evaluationConfigSuffix(kind);
  const panel = vscode.window.createWebviewPanel('fredis.evaluation', 'FREDIS Evaluation', vscode.ViewColumn.One, {
    enableScripts: true, retainContextWhenHidden: true
  });
  attach(panel, context, output, projectRoot, null, kind);
  return panel;
}

function attach(panel, context, output, projectRoot, initialDocument = null, initialKind = null) {
  panel.webview.options = { ...panel.webview.options, enableScripts: true };
  panel.webview.html = html();
  pathDrop.attach(panel);
  const reportProgress = type => progress => panel.webview.postMessage({ type, ...progress });
  panel.webview.onDidReceiveMessage(async message => {
    try {
      if (message.type === 'evaluationReady') {
        if (initialKind) await panel.webview.postMessage({ type: 'evaluationTabSelected', kind: initialKind });
        if (!initialDocument) return;
        if (initialDocument.error) {
          panel.webview.postMessage({ type: 'error', message: initialDocument.error });
          return;
        }
        await panel.webview.postMessage({ type: 'evaluationConfigOpened', kind: initialDocument.kind });
        await panel.webview.postMessage({
          type: `${initialDocument.kind}ConfigLoaded`, config: initialDocument.config,
          path: initialDocument.path
        });
      } else if (message.type === 'saveEvaluationConfig') {
        const configDocument = evaluationConfigDocument(message.kind, message.config);
        const suffix = evaluationConfigSuffix(message.kind);
        const suggestedName = message.kind === 'column'
          ? configFileStem(message.config.title || message.config.groupColumn, 'evaluation')
          : configFileStem(message.config.title, 'evaluation');
        const selected = await vscode.window.showSaveDialog({
          filters: { 'FREDIS evaluation configuration': ['pft'] },
          defaultUri: initialDocument && initialDocument.kind === message.kind
            ? vscode.Uri.file(initialDocument.path)
            : vscode.Uri.file(ensureEvaluationConfigSuffix(suggestedName, message.kind))
        });
        if (selected) {
          const target = vscode.Uri.file(ensureEvaluationConfigSuffix(selected.fsPath, message.kind));
          await vscode.workspace.fs.writeFile(
            target, Buffer.from(JSON.stringify(configDocument, null, 2) + '\n', 'utf8')
          );
          panel.webview.postMessage({ type: `${message.kind}ConfigSaved`, path: target.fsPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'quickSaveEvaluationConfig') {
        const configDocument = evaluationConfigDocument(message.kind, message.config);
        const suffix = evaluationConfigSuffix(message.kind);
        let targetPath = message.path;
        if (!targetPath) {
          const suggestedName = message.kind === 'column'
            ? configFileStem(message.config.title || message.config.groupColumn, 'evaluation')
            : configFileStem(message.config.title, 'evaluation');
          const selected = await vscode.window.showSaveDialog({
            filters: { 'FREDIS evaluation configuration': ['pft'] },
            defaultUri: initialDocument && initialDocument.kind === message.kind
              ? vscode.Uri.file(initialDocument.path)
              : vscode.Uri.file(ensureEvaluationConfigSuffix(suggestedName, message.kind))
          });
          if (!selected) return;
          targetPath = selected.fsPath;
        }
        const target = vscode.Uri.file(ensureEvaluationConfigSuffix(targetPath, message.kind));
        await vscode.workspace.fs.writeFile(
          target, Buffer.from(JSON.stringify(configDocument, null, 2) + '\n', 'utf8')
        );
        panel.webview.postMessage({ type: `${message.kind}ConfigSaved`, path: target.fsPath });
        vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
      } else if (message.type === 'loadEvaluationConfig') {
        // Extension-agnostic: parseEvaluationConfig validates the content
        // and reports a clear error for the wrong shape, so any filename works.
        const selected = await vscode.window.showOpenDialog({ canSelectMany: false });
        if (selected && selected[0]) {
          const raw = await vscode.workspace.fs.readFile(selected[0]);
          const config = parseEvaluationConfig(
            JSON.parse(Buffer.from(raw).toString('utf8')), message.kind
          );
          panel.webview.postMessage({
            type: `${message.kind}ConfigLoaded`, config, path: selected[0].fsPath
          });
        }
      } else if (message.type === 'filteringPick') {
        const picked = await vscode.window.showOpenDialog({ canSelectMany: false });
        if (picked && picked[0]) panel.webview.postMessage({type:'filteringPicked',metadata:message.metadata,index:message.index,path:picked[0].fsPath});
      } else if (['filteringInspect','filteringRun','filteringExport'].includes(message.type)) {
        let directory;
        if (message.type === 'filteringExport') {
          const picked = await vscode.window.showOpenDialog({canSelectFiles:false,canSelectFolders:true,canSelectMany:false,openLabel:'Export here'});
          if (!picked || !picked[0]) return;
          directory = picked[0].fsPath;
        }
        panel.webview.postMessage({type:'filteringBusy',message:'Auditing SpecID retention…'});
        const args = ['filtering','--request-json',JSON.stringify(message.request)];
        if (message.type === 'filteringInspect') args.push('--inspect');
        if (directory) args.push('--output-dir',directory);
        const result = await runCli(context, projectRoot, args, output);
        panel.webview.postMessage({type:message.type === 'filteringInspect'?'filteringColumns':'filteringResult',result});
      } else if (message.type === 'pickGroupedInput') {
        const picked = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { 'Similarity result': ['mssim'] } });
        if (picked && picked[0]) panel.webview.postMessage({
          type: 'groupedInputPicked', groupId: message.groupId,
          seriesId: message.seriesId, path: picked[0].fsPath
        });
      } else if (message.type === 'groupedSummarize') {
        panel.webview.postMessage({ type: 'groupedBusy', text: 'Starting grouped evaluation…', percent: 2 });
        const result = await runCli(context, projectRoot, groupedArgs(message.request), output, reportProgress('groupedProgress'));
        panel.webview.postMessage({ type: 'groupedSummary', result });
      } else if (message.type === 'saveGroupedImage') {
        const suggestedName = ensureSvgSuffix(configFileStem(message.request.title, 'fredis-grouped-evaluation'));
        const target = await vscode.window.showSaveDialog({
          filters: { 'Scalable Vector Graphics': ['svg'] },
          defaultUri: vscode.Uri.file(suggestedName)
        });
        if (target) {
          const targetPath = ensureSvgSuffix(target.fsPath);
          panel.webview.postMessage({ type: 'groupedBusy', text: 'Preparing SVG save…', percent: 2 });
          const result = await runCli(context, projectRoot, groupedArgs(message.request, targetPath), output, reportProgress('groupedProgress'));
          panel.webview.postMessage({ type: 'groupedSaved', path: result.imagePath || targetPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(targetPath)}`);
        }
      } else if (message.type === 'pickInput' || message.type === 'pickMetadata') {
        const metadata = message.type === 'pickMetadata';
        const picked = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: metadata
            ? { 'Spectrum metadata': ['msds', 'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'parquet'] }
            : { 'Evaluation data': ['mssim', 'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'parquet'] }
        });
        if (picked && picked[0]) panel.webview.postMessage({ type: metadata ? 'metadataPicked' : 'inputPicked', path: picked[0].fsPath });
      } else if (message.type === 'inspect') {
        panel.webview.postMessage({ type: 'busy', text: 'Inspecting columns…', percent: 2 });
        const args = ['inspect', '--input', message.input];
        if (message.metadata) args.push('--metadata', message.metadata);
        if (message.joinColumn) args.push('--join-column', message.joinColumn);
        if (message.metadataJoinColumn) args.push('--metadata-join-column', message.metadataJoinColumn);
        const result = await runCli(context, projectRoot, args, output, reportProgress('progress'));
        panel.webview.postMessage({ type: 'inspected', result });
      } else if (message.type === 'summarize') {
        panel.webview.postMessage({ type: 'busy', text: 'Starting evaluation…', percent: 2 });
        const result = await runCli(context, projectRoot, summaryArgs(message.request), output, reportProgress('progress'));
        panel.webview.postMessage({ type: 'summary', result });
      } else if (message.type === 'saveImage') {
        const suggestedName = ensureSvgSuffix(configFileStem(message.request.title || message.request.groupColumn, 'fredis-evaluation'));
        const target = await vscode.window.showSaveDialog({
          filters: { 'Scalable Vector Graphics': ['svg'] },
          defaultUri: vscode.Uri.file(suggestedName)
        });
        if (target) {
          const targetPath = ensureSvgSuffix(target.fsPath);
          panel.webview.postMessage({ type: 'busy', text: 'Preparing SVG save…', percent: 2 });
          const result = await runCli(context, projectRoot, summaryArgs(message.request, targetPath), output, reportProgress('progress'));
          panel.webview.postMessage({ type: 'saved', path: result.imagePath || targetPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(targetPath)}`);
        }
      }
    } catch (error) {
      const grouped = ['pickGroupedInput', 'groupedSummarize', 'saveGroupedImage'].includes(message.type)
        || (['saveEvaluationConfig', 'loadEvaluationConfig'].includes(message.type)
          && message.kind === 'grouped');
      panel.webview.postMessage({ type: message.type.startsWith('filtering') || message.kind === 'filtering' ? 'filteringError' : grouped ? 'groupedError' : 'error', message: String(error.message || error) });
    }
  });
}

function register(context, output, projectRoot) {
  const provider = {
    async resolveCustomTextEditor(document, panel) {
      let initialDocument;
      try {
        const value = JSON.parse(document.getText());
        const kind = value && value.kind;
        if (!['filtering', 'column', 'grouped'].includes(kind)) throw new Error('Evaluation configuration kind must be column or grouped.');
        initialDocument = {
          kind, config: parseEvaluationConfig(value, kind), path: document.uri.fsPath
        };
      } catch (error) {
        initialDocument = { error: String(error.message || error), path: document.uri.fsPath };
      }
      attach(panel, context, output, projectRoot, initialDocument);
    }
  };
  context.subscriptions.push(vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, {
    webviewOptions: { retainContextWhenHidden: true },
    supportsMultipleEditorsPerDocument: true
  }));
}

function html() {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${whiteLightTheme}
  :root{color-scheme:light dark;--accent:#36c5a2;--border:color-mix(in srgb,var(--vscode-editor-foreground) 18%,transparent);--panel:color-mix(in srgb,var(--vscode-editor-background) 90%,var(--vscode-editor-foreground))}
  *{box-sizing:border-box}body{margin:0;color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);font-family:var(--vscode-font-family)}main{max-width:1400px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:start;margin-bottom:20px}h1{margin:3px 0;font-size:28px}h2{font-size:16px;margin:0 0 14px}.eyebrow{font-size:11px;letter-spacing:.14em;font-weight:700;color:var(--accent)}.muted{opacity:.65;margin:4px 0}.layout{display:grid;grid-template-columns:minmax(430px,500px) minmax(0,1fr);gap:16px}.grouped-layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:16px}.panel{padding:18px;border:1px solid var(--border);border-radius:9px;background:var(--panel);margin-bottom:14px}label{font-size:12px;display:block;margin:11px 0}input,select,textarea{width:100%;display:block;margin-top:5px;padding:8px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,var(--border));border-radius:5px}textarea{min-height:110px;resize:vertical;font-family:var(--vscode-editor-font-family),monospace}.path{display:flex;gap:6px}.path input{flex:1}.path button{margin-top:5px}button{font:inherit;color:inherit;background:var(--vscode-button-secondaryBackground);border:1px solid var(--border);border-radius:6px;padding:7px 11px;cursor:pointer}.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground);font-weight:600}.actions{display:flex;gap:7px;flex-wrap:wrap}.check,.radio-option{display:flex;align-items:center;gap:7px}.check input,.radio-option input{width:auto;margin:0}.transform-options{border:1px solid var(--border);border-radius:6px;margin:12px 0;padding:8px 10px}.transform-options legend{font-size:12px;padding:0 4px}.radio-option{margin:7px 0}.category-tools{display:grid;grid-template-columns:minmax(0,1fr) 150px;gap:7px;margin-bottom:8px}.category-tools input,.category-tools select{margin:0}.category-selection{font-size:11px;margin:7px 0}.category-list{max-height:520px;overflow:auto;border:1px solid var(--border);border-radius:6px}.category-head,.category{display:grid;grid-template-columns:22px minmax(90px,1fr) 46px 52px 92px 27px 27px;align-items:center;gap:6px;padding:6px;border-bottom:1px solid var(--border)}.category-head{position:sticky;top:0;z-index:1;background:var(--panel);font-size:10px;font-weight:700}.category input{width:auto;margin:0}.category button{padding:2px 5px}.category-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.category-stat{text-align:right;font-variant-numeric:tabular-nums}.category-density{width:92px;height:28px}.category-density .area{fill:color-mix(in srgb,var(--accent) 24%,transparent);stroke:var(--accent);stroke-width:1}.category-density .median{stroke:var(--vscode-editor-foreground);stroke-width:1}.chart{overflow:auto;min-height:260px;border:1px dashed var(--border);border-radius:7px;display:grid;place-items:center}.chart svg{max-width:100%;height:auto}.status{font-weight:600}.error{color:var(--vscode-errorForeground)}.eval-progress{width:100%;height:7px;margin:10px 0 2px;accent-color:var(--accent)}table{width:100%;border-collapse:collapse;margin-top:14px;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--border)}.size-row{display:grid;grid-template-columns:1fr 1fr;gap:9px}.eval-tabs{display:flex;gap:8px;margin-bottom:22px;border-bottom:1px solid var(--border);padding-bottom:10px}.eval-tabs button.active{background:var(--vscode-button-background);color:var(--vscode-button-foreground)}.edit-list{display:grid;gap:7px;margin-bottom:10px}.edit-row{display:grid;grid-template-columns:minmax(0,1fr) auto auto auto;gap:5px;align-items:center}.edit-row input{margin:0}.edit-row button{padding:6px 8px}.series-row{grid-template-columns:minmax(0,1fr) 40px auto auto auto}.series-row input[type=color]{height:34px;padding:3px}.result-matrix-scroll{overflow:auto}.result-matrix{min-width:700px}.result-matrix td:nth-child(3){min-width:330px}.series-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}.save-toast{position:fixed;top:14px;right:18px;background:var(--accent);color:#0b1f19;padding:8px 14px;border-radius:7px;font-size:12px;font-weight:700;opacity:0;transform:translateY(-6px);pointer-events:none;transition:opacity .2s ease,transform .2s ease;z-index:1000;box-shadow:0 4px 14px rgba(0,0,0,.28)}.save-toast.show{opacity:1;transform:translateY(0)}${pathDrop.css()}@media(max-width:900px){.layout,.grouped-layout{grid-template-columns:1fr}}
  </style></head><body><div id="saveToast" class="save-toast" role="status" aria-live="polite"></div><script>
  window.showSaveToast=function(text){
    var toast=document.getElementById('saveToast');
    toast.textContent=text||'✓ Saved';
    toast.classList.add('show');
    clearTimeout(window.showSaveToast._t);
    window.showSaveToast._t=setTimeout(function(){toast.classList.remove('show')},1600);
  };
  </script><main><nav class="eval-tabs"><button class="active" data-eval-tab="filtering">Training Data Filtering</button><button data-eval-tab="column">Column Evaluation</button><button data-eval-tab="grouped">Grouped Box Plot</button></nav>${filteringHtml()}<section id="columnEvaluation" hidden><header><div><span class="eyebrow">FREDIS EVALUATION</span><h1>Column Evaluation</h1><p class="muted">Group a result table by categories or numeric ranges.</p></div><div class="actions"><button id="loadColumnConfig">Load Settings</button><button id="saveColumnConfig">Save Settings</button><button id="saveImage" disabled>Save SVG</button></div></header>
  <div class="layout"><aside>
    <section class="panel"><h2>Dataset</h2><label>Similarity result (.mssim)<div class="path"><input id="input" data-path-kind="file" placeholder="results.mssim"><button id="browse">Browse</button></div></label><label>Metadata source (.msds or table)<div class="path"><input id="metadata" data-path-kind="file" placeholder="metadata.msds"><button data-browse-metadata>Browse</button></div></label><div class="size-row"><label>Result join column<input id="joinColumn" placeholder="Auto: SourceSpecID"></label><label>Metadata join column<input id="metadataJoinColumn" placeholder="Auto"></label></div><p class="muted">Column names may differ. Auto prefers SourceSpecID → SourceSpecID when available, then SourceSpecID → SpecID.</p><button id="load" class="primary">Load columns</button><p id="datasetInfo" class="muted"></p></section>
    <section class="panel"><h2>Grouping</h2><label>Column<select id="groupColumn" disabled></select></label><fieldset class="transform-options"><legend>Value conversion</legend><label class="radio-option"><input type="radio" name="transform" value="none" checked>None</label><label class="radio-option"><input type="radio" name="transform" value="collision-energy">Collision Energy parser</label><label class="radio-option"><input type="radio" name="transform" value="chemical">SMILES chemical information</label><label class="radio-option"><input type="radio" name="transform" value="smarts">SMARTS substructure matching</label></fieldset><div id="ceOptions" hidden><label>Precursor m/z column<input id="precursorMzColumn" value="PrecursorMZ"></label><label>Instrument column (optional)<input id="instrumentColumn"></label></div><div id="smilesOptions" hidden><label>SMILES column<input id="smilesColumn" value="SMILES"></label></div><div id="chemicalOptions" hidden><label>Chemical descriptor<select id="chemicalDescriptor"><option>HeavyAtomCount</option><option>ExactMolWt</option><option>TPSA</option><option>MolLogP</option><option>NumHAcceptors</option><option>NumHDonors</option><option>NumRotatableBonds</option><option>RingCount</option><option>NumAromaticRings</option><option>NumAliphaticRings</option><option>FractionCSP3</option><option>NumHeteroatoms</option><option>FormalCharge</option><option>BertzCT</option></select></label></div><div id="smartsOptions" hidden><label>SMARTS queries<textarea id="smartsQueries" spellcheck="false" placeholder="Alcohol = [OX2H]&#10;Aromatic ring = c1ccccc1"></textarea><small class="muted">One query per line. Use “label = SMARTS” (spaces around =), or enter only SMARTS. Groups may overlap.</small></label></div><label>Mode<select id="mode"><option value="auto">Auto</option><option value="categorical">Category</option><option value="numeric">Numeric ranges</option></select></label><label id="binsLabel" hidden>Range boundaries<input id="bins" value="0,10,20" placeholder="0,10,20" disabled><small class="muted">0,10,20 creates [0,10), [10,20), [20,~).</small></label><p class="muted">Changes are applied only when Generate plot is pressed.</p><button id="generate" class="primary" disabled>Generate plot</button></section>
    <section class="panel"><h2>Categories</h2><div class="category-tools"><input id="categorySearch" type="search" aria-label="Search categories" placeholder="Search categories…"><select id="categorySort" aria-label="Sort categories"><option value="custom">Custom order</option><option value="name-asc">Name A → Z</option><option value="name-desc">Name Z → A</option><option value="count-desc">Count high → low</option><option value="count-asc">Count low → high</option><option value="median-desc">Median high → low</option><option value="median-asc">Median low → high</option></select></div><div class="actions"><button id="all">Select all</button><button id="none">Clear all</button><button id="allShown">Select shown</button><button id="noneShown">Clear shown</button></div><p id="categorySelection" class="category-selection muted"></p><p class="muted category-help">n is the valid score count. Distribution spans 0–1; its line is the median. Hover for quartiles and range.</p><div id="categories" class="category-list"><p class="muted" style="padding:8px">Generate once to list categories.</p></div></section>
    <section class="panel"><h2>Image</h2><label>Title<input id="title" placeholder="Cosine similarity by grouping column"></label><label>Plot type<select id="plotType"><option value="box">Box plot</option><option value="violin">Violin plot</option></select></label><div class="size-row"><label>Width<input id="width" type="number" min="240" value="900"></label><label>Height<input id="height" type="number" min="200" value="520"></label></div><label>Graph opacity<input id="graphOpacity" type="number" min="0" max="1" step="0.05" value="1"></label><div class="size-row"><label>X-axis label size<input id="xLabelSize" type="number" min="6" max="96" value="11"></label><label>Y-axis label size<input id="yLabelSize" type="number" min="6" max="96" value="11"></label></div><div class="size-row"><label>X-axis title size<input id="xAxisTitleSize" type="number" min="6" max="96" value="12"></label><label>Y-axis title size<input id="yAxisTitleSize" type="number" min="6" max="96" value="12"></label></div><div class="size-row"><label>X-axis title distance<input id="xAxisTitleGap" type="number" min="0" max="200" value="8"></label><label>Y-axis title distance<input id="yAxisTitleGap" type="number" min="0" max="200" value="8"></label></div><div class="size-row"><label class="check"><input id="showXAxisTitle" type="checkbox" checked>Show X-axis title</label><label class="check"><input id="showYAxisTitle" type="checkbox" checked>Show Y-axis title</label></div><label>X-axis label rotation<select id="xLabelRotation"><option value="auto">Auto</option><option value="0">0°</option><option value="30">30°</option><option value="45">45°</option><option value="60">60°</option><option value="90">90°</option></select></label><label>Chart title size<input id="titleSize" type="number" min="6" max="96" value="17"></label><label>Plot color<input id="color" type="color" value="#36c5a2"></label><label class="check"><input id="transparent" type="checkbox" checked>Transparent background</label></section>
  </aside><section class="panel"><div class="actions"><span id="status" class="status">Choose a dataset.</span></div><progress id="progress" class="eval-progress" max="100" hidden></progress><div id="chart" class="chart"><p class="muted">The plot will appear here.</p></div><table id="table" hidden><thead><tr><th>Category / range</th><th>n</th><th>Matched rows</th><th>Unique compounds</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th></tr></thead><tbody></tbody></table></section></div>
  <script>
  const vscode=acquireVsCodeApi(),input=document.getElementById('input'),group=document.getElementById('groupColumn'),mode=document.getElementById('mode'),binsLabel=document.getElementById('binsLabel'),bins=document.getElementById('bins'),categories=document.getElementById('categories'),status=document.getElementById('status'),progressBar=document.getElementById('progress'),chart=document.getElementById('chart'),table=document.getElementById('table');
  let categoryState=[],pendingColumnConfig=null,lastAppliedRequest=null,pendingAppliedRequest=null,currentPath=null;
  const categorySearch=document.getElementById('categorySearch'),categorySort=document.getElementById('categorySort'),categorySelection=document.getElementById('categorySelection');
  const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const transform=()=>document.querySelector('input[name="transform"]:checked').value;
  function readSmartsQueries(){return document.getElementById('smartsQueries').value.split(/\\r?\\n/).map(line=>line.trim()).filter(Boolean).map(line=>{const tab=line.indexOf(String.fromCharCode(9)),spacedEquals=line.indexOf(' = '),separator=tab>=0?tab:spacedEquals;if(separator<0)return{label:line,smarts:line};const width=tab>=0?1:3;return{label:line.slice(0,separator).trim(),smarts:line.slice(separator+width).trim()}})}
  function setStatus(text,error=false){status.textContent=text;status.className='status'+(error?' error':'');progressBar.hidden=true}
  function setProgress(text,percent){status.textContent=text;status.className='status';progressBar.hidden=false;progressBar.value=Math.max(0,Math.min(100,Number(percent)||0));document.getElementById('generate').disabled=true}
  function request(){const smilesColumn=document.getElementById('smilesColumn').value,valueTransform=transform();return{input:input.value,metadata:document.getElementById('metadata').value,joinColumn:document.getElementById('joinColumn').value,metadataJoinColumn:document.getElementById('metadataJoinColumn').value,groupColumn:['chemical','smarts'].includes(valueTransform)?smilesColumn:group.value,transform:valueTransform,precursorMzColumn:document.getElementById('precursorMzColumn').value,instrumentColumn:document.getElementById('instrumentColumn').value,smilesColumn,chemicalDescriptor:document.getElementById('chemicalDescriptor').value,smartsQueries:readSmartsQueries(),mode:mode.value,bins:mode.value==='numeric'?bins.value:'',title:document.getElementById('title').value,plotType:document.getElementById('plotType').value,width:Number(document.getElementById('width').value),height:Number(document.getElementById('height').value),graphOpacity:Number(document.getElementById('graphOpacity').value),xLabelSize:Number(document.getElementById('xLabelSize').value),yLabelSize:Number(document.getElementById('yLabelSize').value),xAxisTitleSize:Number(document.getElementById('xAxisTitleSize').value),yAxisTitleSize:Number(document.getElementById('yAxisTitleSize').value),xAxisTitleGap:Number(document.getElementById('xAxisTitleGap').value),yAxisTitleGap:Number(document.getElementById('yAxisTitleGap').value),showXAxisTitle:document.getElementById('showXAxisTitle').checked,showYAxisTitle:document.getElementById('showYAxisTitle').checked,xLabelRotation:document.getElementById('xLabelRotation').value,titleSize:Number(document.getElementById('titleSize').value),color:document.getElementById('color').value,transparent:document.getElementById('transparent').checked,include:categoryState.filter(x=>x.checked).map(x=>x.name),order:categoryState.map(x=>x.name)}}
  function markDraft(){if(lastAppliedRequest)setStatus('Unapplied changes · press Generate plot')}
  function resetCategories(){categoryState=[];categorySearch.value='';categorySort.value='custom';categorySelection.textContent='';categories.innerHTML='<p class="muted" style="padding:8px">Press Generate to list categories.</p>';markDraft()}
  function updateMode(){const numeric=mode.value==='numeric';binsLabel.hidden=!numeric;bins.disabled=!numeric;markDraft()}
  function updateTransform(reset=true){const value=transform();document.getElementById('ceOptions').hidden=value!=='collision-energy';document.getElementById('smilesOptions').hidden=!['chemical','smarts'].includes(value);document.getElementById('chemicalOptions').hidden=value!=='chemical';document.getElementById('smartsOptions').hidden=value!=='smarts';if(value==='collision-energy'){const option=[...group.options].find(item=>item.value==='CollisionEnergy'||item.value.endsWith('.CollisionEnergy'));if(option)group.value=option.value}if(['collision-energy','chemical'].includes(value))mode.value='numeric';if(value==='smarts')mode.value='categorical';mode.disabled=value==='smarts';updateMode();if(reset)resetCategories()}
  function generate(preserve=true){if(!input.value||!group.value)return;const value=request();pendingAppliedRequest={...value,include:preserve&&categoryState.length?value.include:undefined,order:preserve&&categoryState.length?value.order:undefined};vscode.postMessage({type:'summarize',request:pendingAppliedRequest})}
  function categoryIndices(){const query=categorySearch.value.trim().toLocaleLowerCase();return categoryState.map((x,i)=>[x,i]).filter(([x])=>!query||x.name.toLocaleLowerCase().includes(query)).map(([,i])=>i)}
  function categorySpark(x){if(!Array.isArray(x.density)||!x.density.length)return '<span class="muted">—</span>';const points=x.density.map(p=>(Math.max(0,Math.min(1,Number(p[0])))*90).toFixed(1)+','+(26-Math.max(0,Math.min(1,Number(p[1])))*23).toFixed(1)).join(' L'),median=Number(x.median),line=Number.isFinite(median)?'<path class="median" d="M'+(median*90).toFixed(1)+' 2V26"/>':'';const tip='min '+formatStat(x.min)+' · Q1 '+formatStat(x.q1)+' · median '+formatStat(x.median)+' · Q3 '+formatStat(x.q3)+' · max '+formatStat(x.max);return '<svg class="category-density" viewBox="0 0 90 28" role="img"><title>'+esc(tip)+'</title><path class="area" d="M0 26 L'+points+' L90 26 Z"/>'+line+'</svg>'}
  function formatStat(value){return value===null||value===undefined||!Number.isFinite(Number(value))?'—':Number(value).toFixed(3)}
  function updateCategorySelection(){categorySelection.textContent=categoryState.filter(x=>x.checked).length+' / '+categoryState.length+' selected · '+categoryIndices().length+' shown'}
  function setCategoryChecks(indices,checked){const selected=new Set(indices);for(const i of selected)categoryState[i].checked=checked;categories.querySelectorAll('[data-i]').forEach(row=>{const index=Number(row.dataset.i);if(selected.has(index))row.querySelector('input[type="checkbox"]').checked=checked});updateCategorySelection();markDraft()}
  function renderCategoryList(){const indices=categoryIndices();updateCategorySelection();const head='<div class="category-head"><span></span><span>Category</span><span>n</span><span>Median</span><span>Distribution</span><span></span><span></span></div>';const rows=indices.map(i=>{const x=categoryState[i];return '<div class="category" data-i="'+i+'"><input type="checkbox" '+(x.checked?'checked':'')+' aria-label="Include '+esc(x.name)+'"><span class="category-name" title="'+esc(x.name)+'">'+esc(x.name)+'</span><span class="category-stat">'+(x.count??'—')+'</span><span class="category-stat">'+formatStat(x.median)+'</span>'+categorySpark(x)+'<button data-move="-1" title="Move up">↑</button><button data-move="1" title="Move down">↓</button></div>'}).join('');categories.innerHTML=categoryState.length?head+(rows||'<p class="muted" style="padding:8px">No matching categories.</p>'):'<p class="muted" style="padding:8px">No categories.</p>'}
  function renderCategories(rows){const old=new Map(categoryState.map(x=>[x.name,x])),incoming=new Map((rows||[]).map(r=>[String(r.category??r.name),r]));if(!categoryState.length)categoryState=(rows||[]).map(r=>({ ...r,name:String(r.category??r.name),checked:old.has(String(r.category??r.name))?old.get(String(r.category??r.name)).checked:true}));else{categoryState=categoryState.map(x=>incoming.has(x.name)?{...x,...incoming.get(x.name),name:x.name}:x);for(const [name,row] of incoming)if(!old.has(name))categoryState.push({...row,name,checked:true})}renderCategoryList()}
  function sortCategories(){const value=categorySort.value;if(value==='custom')return;const [key,direction]=value.split('-'),sign=direction==='asc'?1:-1;categoryState.sort((a,b)=>{if(key==='name')return sign*a.name.localeCompare(b.name,undefined,{numeric:true,sensitivity:'base'});const left=Number(a[key]),right=Number(b[key]),leftOk=Number.isFinite(left),rightOk=Number.isFinite(right);if(leftOk!==rightOk)return leftOk?-1:1;return leftOk&&left!==right?sign*(left-right):a.name.localeCompare(b.name,undefined,{numeric:true,sensitivity:'base'})});renderCategoryList();markDraft()}
  document.getElementById('browse').onclick=()=>vscode.postMessage({type:'pickInput'});
  document.querySelector('[data-browse-metadata]').onclick=()=>vscode.postMessage({type:'pickMetadata'});
  document.getElementById('load').onclick=()=>{if(input.value){const r=request();vscode.postMessage({type:'inspect',input:r.input,metadata:r.metadata,joinColumn:r.joinColumn,metadataJoinColumn:r.metadataJoinColumn})}};
  document.getElementById('saveColumnConfig').onclick=()=>vscode.postMessage({type:'saveEvaluationConfig',kind:'column',config:request()});
  document.getElementById('loadColumnConfig').onclick=()=>vscode.postMessage({type:'loadEvaluationConfig',kind:'column'});
  input.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();document.getElementById('load').click()}};
  mode.onchange=()=>{updateMode();resetCategories()};
  document.getElementById('generate').onclick=()=>generate();
  group.onchange=()=>{mode.value=transform()==='none'?(group.selectedOptions[0].dataset.kind==='numeric'?'numeric':'categorical'):'numeric';updateMode();resetCategories()};
  document.querySelectorAll('input[name="transform"]').forEach(item=>item.onchange=updateTransform);
  bins.onchange=resetCategories;document.getElementById('smilesColumn').onchange=resetCategories;document.getElementById('chemicalDescriptor').onchange=resetCategories;document.getElementById('smartsQueries').onchange=resetCategories;document.getElementById('precursorMzColumn').onchange=resetCategories;document.getElementById('instrumentColumn').onchange=resetCategories;
  categorySearch.oninput=renderCategoryList;categorySort.onchange=sortCategories;
  categories.onchange=e=>{const row=e.target.closest('[data-i]');if(row){categoryState[Number(row.dataset.i)].checked=e.target.checked;updateCategorySelection();markDraft()}};
  categories.onclick=e=>{const button=e.target.closest('[data-move]');if(!button)return;const row=button.closest('[data-i]'),from=Number(row.dataset.i),shown=categoryIndices(),position=shown.indexOf(from),targetPosition=Math.max(0,Math.min(shown.length-1,position+Number(button.dataset.move))),to=shown[targetPosition];if(from===to)return;[categoryState[from],categoryState[to]]=[categoryState[to],categoryState[from]];categorySort.value='custom';renderCategoryList();markDraft()};
  document.getElementById('all').onclick=()=>setCategoryChecks(categoryState.map((_,i)=>i),true);
  document.getElementById('none').onclick=()=>setCategoryChecks(categoryState.map((_,i)=>i),false);
  document.getElementById('allShown').onclick=()=>setCategoryChecks(categoryIndices(),true);
  document.getElementById('noneShown').onclick=()=>setCategoryChecks(categoryIndices(),false);
  document.getElementById('saveImage').onclick=()=>{if(lastAppliedRequest)vscode.postMessage({type:'saveImage',request:lastAppliedRequest})};
  window.evalColumn={request:()=>request(),getPath:()=>currentPath};
  window.addEventListener('message',e=>{
    const m=e.data;
    if(m.type==='inputPicked'){input.value=m.path;return}
    if(m.type==='metadataPicked'){document.getElementById('metadata').value=m.path;return}
    if(m.type==='columnConfigLoaded'){
      const c=m.config;categorySearch.value='';categorySort.value='custom';input.value=c.input||'';document.getElementById('metadata').value=c.metadata||'';document.getElementById('joinColumn').value=c.joinColumn||'';document.getElementById('metadataJoinColumn').value=c.metadataJoinColumn||c.joinColumn||'';
      const radio=document.querySelector('input[name="transform"][value="'+(c.transform||'none')+'"]');if(radio)radio.checked=true;
      document.getElementById('precursorMzColumn').value=c.precursorMzColumn||'PrecursorMZ';document.getElementById('instrumentColumn').value=c.instrumentColumn||'';document.getElementById('smilesColumn').value=c.smilesColumn||'SMILES';document.getElementById('chemicalDescriptor').value=c.chemicalDescriptor||'HeavyAtomCount';document.getElementById('smartsQueries').value=(Array.isArray(c.smartsQueries)?c.smartsQueries:[]).map(item=>{const label=String(item.label??item.name??item.smarts??''),smarts=String(item.smarts??'');return label===smarts?smarts:label+' = '+smarts}).join(String.fromCharCode(10));mode.value=c.mode||'auto';bins.value=c.bins||'';document.getElementById('title').value=c.title||'';document.getElementById('plotType').value=c.plotType||'box';document.getElementById('width').value=c.width||900;document.getElementById('height').value=c.height||520;document.getElementById('graphOpacity').value=c.graphOpacity??1;document.getElementById('xLabelSize').value=c.xLabelSize??11;document.getElementById('yLabelSize').value=c.yLabelSize??11;document.getElementById('xAxisTitleSize').value=c.xAxisTitleSize??c.xLabelSize??12;document.getElementById('yAxisTitleSize').value=c.yAxisTitleSize??c.yLabelSize??12;document.getElementById('xAxisTitleGap').value=c.xAxisTitleGap??8;document.getElementById('yAxisTitleGap').value=c.yAxisTitleGap??8;document.getElementById('showXAxisTitle').checked=c.showXAxisTitle!==false;document.getElementById('showYAxisTitle').checked=c.showYAxisTitle!==false;document.getElementById('xLabelRotation').value=c.xLabelRotation||'auto';document.getElementById('titleSize').value=c.titleSize??17;document.getElementById('color').value=c.color||'#36c5a2';document.getElementById('transparent').checked=c.transparent!==false;pendingColumnConfig=c;updateTransform(false);
      currentPath=m.path;
      if(input.value){setStatus('Loaded '+m.path+' · inspecting columns…');document.getElementById('load').click()}else{pendingColumnConfig=null;setStatus('Loaded settings '+m.path+' · press Generate to apply')}return
    }
    if(m.type==='columnConfigSaved'){currentPath=m.path;setStatus('Saved settings '+m.path);if(window.showSaveToast)window.showSaveToast('✓ Saved '+m.path.split(/[\\/]/).pop());return}
    if(m.type==='busy'||m.type==='progress'){setProgress(m.message||m.text,m.percent??5);return}
    if(m.type==='inspected'){
      const cols=m.result.columns.filter(c=>!['index1','index2','cosine_similarity'].includes(c.name));group.innerHTML=cols.map(c=>'<option value="'+esc(c.name)+'" data-kind="'+c.kind+'">'+esc(c.name)+' ('+c.kind+')</option>').join('');group.disabled=document.getElementById('generate').disabled=!cols.length;document.getElementById('datasetInfo').textContent=m.result.rows+' pairs · '+cols.length+' grouping columns';
      if(cols.length){const saved=pendingColumnConfig;pendingColumnConfig=null;if(saved&&cols.some(c=>c.name===saved.groupColumn)){group.value=saved.groupColumn;mode.value=saved.mode||'auto';bins.value=saved.bins||'';const names=Array.isArray(saved.order)?saved.order:[];const included=new Set(Array.isArray(saved.include)?saved.include:names);categoryState=names.map(name=>({name,checked:included.has(name)}));if(categoryState.length)renderCategories(categoryState.map(x=>({category:x.name})));updateTransform(false)}else{mode.value=transform()==='none'?(group.selectedOptions[0].dataset.kind==='numeric'?'numeric':'categorical'):'numeric';updateTransform(false);categoryState=[]}setStatus('Columns loaded · press Generate plot')}return
    }
    if(m.type==='summary'){
      lastAppliedRequest=pendingAppliedRequest;pendingAppliedRequest=null;renderCategories(m.result.categoryRows||m.result.rows);chart.innerHTML=m.result.svg;table.hidden=false;const f=v=>v===null?'—':Number(v).toFixed(4),n=v=>v===null||v===undefined?'—':v;table.querySelector('tbody').innerHTML=m.result.rows.map(r=>'<tr><td>'+esc(r.category)+'</td><td>'+r.count+'</td><td>'+n(r.matchedRows)+'</td><td>'+n(r.uniqueCompounds)+'</td><td>'+f(r.min)+'</td><td>'+f(r.q1)+'</td><td>'+f(r.median)+'</td><td>'+f(r.q3)+'</td><td>'+f(r.max)+'</td></tr>').join('');document.getElementById('saveImage').disabled=false;document.getElementById('generate').disabled=!group.value;const structures=m.result.structureSummary;setStatus(m.result.rows.length+' groups · '+m.result.sourceRows+' similarity pairs'+(structures?' · '+structures.uniqueMatchedCompounds+'/'+structures.uniqueValidCompounds+' compounds matched':''));return
    }
    if(m.type==='saved'){document.getElementById('generate').disabled=!group.value;setStatus('Saved '+m.path);return}
    if(m.type==='error'){pendingColumnConfig=null;pendingAppliedRequest=null;document.getElementById('generate').disabled=!group.value;setStatus(m.message,true)}
  });
  </script></section>${groupedHtml()}<script>
  document.querySelectorAll('[data-eval-tab]').forEach(button=>button.onclick=()=>{const kind=button.dataset.evalTab;for(const k of ['filtering','column','grouped'])document.getElementById(k+'Evaluation').hidden=k!==kind;document.querySelectorAll('[data-eval-tab]').forEach(item=>item.classList.toggle('active',item===button))});
  (${filteringClient.toString()})();
  (${groupedClient.toString()})();
  window.addEventListener('message',event=>{if(['evaluationConfigOpened','evaluationTabSelected'].includes(event.data.type)){const tab=document.querySelector('[data-eval-tab="'+event.data.kind+'"]');if(tab)tab.click()}});
  document.addEventListener('keydown',e=>{
    const key=(e.key||'').toLowerCase();
    if(!(e.ctrlKey||e.metaKey)||e.shiftKey||e.altKey||key!=='s')return;
    e.preventDefault();
    const grouped=!document.getElementById('groupedEvaluation').hidden;
    const filtering=!document.getElementById('filteringEvaluation').hidden;
    const api=filtering?window.evalFiltering:grouped?window.evalGrouped:window.evalColumn;
    if(!api)return;
    vscode.postMessage({type:'quickSaveEvaluationConfig',kind:filtering?'filtering':grouped?'grouped':'column',config:api.request(),path:api.getPath()});
  });
  ${pathDrop.script({host:true,compact:true})}
  vscode.postMessage({type:'evaluationReady'});
  </script></main></body></html>`;
}

module.exports = {
  open, register, html, summaryArgs, groupedArgs, evaluationConfigDocument,
  parseEvaluationConfig, evaluationConfigSuffix, ensureEvaluationConfigSuffix, ensureSvgSuffix
};
