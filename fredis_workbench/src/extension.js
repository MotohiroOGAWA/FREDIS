const modelArchitecture=require('./features/model-architecture/editor');
const reactionPatternUi = require('./features/reaction-pattern-set/ui');
const {commonCss,formCss} = require('./features/reaction-pattern-set/styles');
const projects = require('./workbench/project');
const reactionPatternVisualEditor = require('./features/reaction-pattern-set/visual-editor');
const reactionPatternPreview = require('./features/reaction-pattern-set/reaction-preview');
const trainingWorkbench = require('./workbench/training');
const resultView = require('./features/fragment-tree-result/view');
const layout = require('./workbench/layout');
const preparation = require('./workbench/preparation');
const datasetUpload = require('./workbench/dataset-upload');
const workbenchServices = require('./workbench/services');
const parameterEditor = require('./workbench/parameter-editor');
const workbenchDefaults = require('./workbench/defaults');
const parameterService = require('./workbench/parameter-service');
const workbench = require('./workbench/panel');
const trainingMetrics = require('./features/training-metrics/editor');
const trainingReport = require('./features/training-report/view');
const smartsSearch = require('./features/smarts-search/editor');
const evaluation = require('./features/evaluation/editor');
const spectrumPrediction = require('./features/spectrum-prediction/editor');
const spectrumViewer = require('./features/spectrum-viewer/workbench');
const embeddedSpectrumViewer = require('./features/spectrum-viewer/extension');
const fragmentTreeVisualization = require('./features/fragment-tree-visualization/editor');
const reactionPatternAnalysis = require('./features/reaction-pattern-analysis/editor');
const reactionPatternAnalysisResultView = require('./features/reaction-pattern-analysis/results-view');
const pathDrop = require('./webview-path-drop');
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const reactionPatternSetEditor = require('./features/reaction-pattern-set/editor');
const adductUi = require('./features/adduct-rule-set/ui');
const adductRuleSetEditor = require('./features/adduct-rule-set/editor');
const fragmenterEditor = require('./features/fragmenter/editor');
const reactionPatternHost = require('./features/reaction-pattern-set/host');
const { projectRoot, isFredisRoot, runChemistryBackend, showElementPicker, readReactionPatternImport, safeFileStem, ensureFileSuffix } = reactionPatternHost;

const RESULT_NAME = 'fragment-tree.training-data.pft';
const REACTION_PATTERN_ANALYSIS_CONFIG_VIEW_TYPE = 'fredis.reactionPatternAnalysisConfigEditor';
// result.reactionpatternanalysis.pft is binary HDF5; it opens in the Workbench, never as text.
const REACTION_PATTERN_ANALYSIS_RESULT_VIEW_TYPE = 'fredis.reactionPatternAnalysisResultEditor';
let runningProcess, predictionProcess;

function activate(context) {
  const output = vscode.window.createOutputChannel('FREDIS');
  const provider = new ResultEditorProvider(context, output);
  const reactionPatternAnalysisConfigProvider = new ReactionPatternAnalysisConfigEditorProvider(context, output);
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand('fredis.openWorkbench', () => openWorkbench(context, output)),
    vscode.commands.registerCommand('fredis.openResult', openResultPicker),
    vscode.commands.registerCommand('fredis.openEvaluation', () => evaluation.open(context, output, projectRoot)),
    vscode.commands.registerCommand('fredis.openColumnEvaluation', () => evaluation.open(context, output, projectRoot, 'column')),
    vscode.commands.registerCommand('fredis.openGroupedEvaluation', () => evaluation.open(context, output, projectRoot, 'grouped')),
    vscode.window.registerCustomEditorProvider('fredis.resultViewer', provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true
    }),
    vscode.window.registerCustomEditorProvider(REACTION_PATTERN_ANALYSIS_CONFIG_VIEW_TYPE, reactionPatternAnalysisConfigProvider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true
    }),
    vscode.window.registerCustomEditorProvider(REACTION_PATTERN_ANALYSIS_RESULT_VIEW_TYPE, reactionPatternAnalysisConfigProvider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true
    })
  );
  evaluation.register(context, output, projectRoot);
  reactionPatternSetEditor.register(context);
  adductRuleSetEditor.register(context);
  modelArchitecture.register(context);
  fragmenterEditor.register(context);
  workbench.register(context, output, projectRoot, openWorkbench);
  enableDevelopmentReload(context, output);
}

function enableDevelopmentReload(context, output) {
  if (context.extensionMode !== vscode.ExtensionMode.Development) return;
  const pattern = new vscode.RelativePattern(context.extensionPath, '{src,media}/**/*');
  const watcher = vscode.workspace.createFileSystemWatcher(pattern);
  let reloadTimer;
  const reload = uri => {
    if (!/\.(js|css|html|json)$/.test(uri.fsPath)) return;
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => {
      output.appendLine(`[development] ${path.relative(context.extensionPath, uri.fsPath)} changed; reloading Extension Host.`);
      // Restart only the Extension Host: reloading the whole Extension Development
      // Host window can end the debug session in a remote/dev-container setup,
      // leaving no window and a stale session that blocks the next Ctrl+F5.
      vscode.commands.executeCommand('workbench.action.restartExtensionHost');
    }, 250);
  };
  watcher.onDidChange(reload);
  watcher.onDidCreate(reload);
  watcher.onDidDelete(reload);
  context.subscriptions.push(watcher, { dispose: () => clearTimeout(reloadTimer) });
  output.appendLine('[development] Auto reload is enabled.');
}

function defaultConfig(context) {
  const root = projectRoot(context);
  const chemistry = parameterService.chemistryDefaults(root);
  return workbenchDefaults.workflowDefaults('data',{
    application: 'fragment-tree-data-preparation', fragmenterParams: chemistry.fragmenter_params, symbols: chemistry.symbols,
    input: '', smilesColumn:'SMILES', adductTypeColumn:'AdductType', collisionEnergyColumn:'CollisionEnergy', precursorMzColumn:'PrecursorMZ',
    validationSmilesColumn:'', validationAdductTypeColumn:'', validationCollisionEnergyColumn:'', validationPrecursorMzColumn:'',
    validationInput:'', validationRatio:0.1, validationSeed:0, minimumRelativeIntensity:0, overwrite:false, numWorkers:1, chunkSize:1,
    outputDir: ''
  },root);
}

function defaultTrainingConfig(context) {
  const root = projectRoot(context);
  // Architecture defaults not listed here are filled in by the parameter editor.
  const modelConfig = { mol_encoder_params: { node_dim: 16, graph_dim: 64, num_layers: 2, num_heads: 4, dropout: 0.5 }, reaction_model_params:{hidden_dim:128}, spectrum_model_params:{} };
  return workbenchDefaults.workflowDefaults('training',{
    application: 'fragment-tree-training', modelConfig, trainDir: '', valDir: '',
    outputDir: '',
    experimentName: 'exp_main', epochs: 1, batchSize: 4, device: 'cuda', maxSamples:128, seed:42, warmupSteps:100, trainLogIntervalSteps:100, validationIntervalSteps:0, validationFraction:0.1, lrPatience:3, earlyStoppingPatience:10, minLr:0.000001, branchWeight:1, negativeWeight:0.5, branchMilTemperature:0.1,   intensityLossWeight:1, cosineLossWeight:1,   gradientClip:1, lr: 0.0001, dropout: 0.5, resume: '', minimumAssignmentScore:0, minimumAssignmentScoreWithoutPrecursor:0, cacheDatasetInMemory:false,
    fineTuneCheckpoint: '', headOnly: false
  },root);
}

function openWorkbench(context, output, initialPage = "home") {
  const panel = vscode.window.createWebviewPanel('fredis.workbench', 'FREDIS Workbench', vscode.ViewColumn.One, {
    enableScripts: true, retainContextWhenHidden: true, localResourceRoots: []
  });
  return resolveWorkbenchPanel(context, output, panel, initialPage);
}

function reactionPatternAnalysisUiSettings() {
  const settings = vscode.workspace.getConfiguration('fredis');
  return {
    compoundPageSize: settings.get('workbench.reactionPatternAnalysisPageSize', 20),
    peakPageSize: settings.get('workbench.reactionPatternAnalysisPeakPageSize', 20),
    smilesColumnWidth: settings.get('workbench.reactionPatternAnalysisSmilesColumnWidth', 240)
  };
}

function resolveWorkbenchPanel(context, output, panel, initialPage = "home", initialReactionPatternDocument = null) {
  context = projects.createContext(context);
  spectrumViewer.configure(context, vscode);
  panel.webview.options = { enableScripts: true, localResourceRoots: [] };
  projects.attach(panel, context, projectRoot);
  smartsSearch.attach(panel, context, projectRoot);
  spectrumPrediction.attach(panel, context, projectRoot, output);
  workbench.attach(panel, context, projectRoot, output, initialPage);
  parameterService.attach(panel, context, projectRoot);
  workbenchServices.attach(panel, context, projectRoot);
  workbenchDefaults.attach(panel);
  datasetUpload.attach(panel,context);
  pathDrop.attach(panel);
  reactionPatternAnalysis.attach(panel,context,output,projectRoot,initialReactionPatternDocument);
  const spectrumProvider = embeddedSpectrumViewer.createEmbeddedProvider(context, output);
  spectrumProvider.resolveEmbeddedEditor(panel).catch(error => {
    output.appendLine(`[spectrum viewer] ${error.stack || error}`);
    panel.webview.postMessage({ type: 'error', title: 'Could not open Spectrum Viewer', message: error.message || String(error) });
  });
  panel.onDidDispose(() => spectrumProvider.dispose());
  panel.onDidDispose(()=>{if(predictionProcess)workbench.stopChild(predictionProcess);});
  const initialConfig = defaultConfig(context);
  const initialTrainingConfig = defaultTrainingConfig(context);
  modelArchitecture.attachWorkbench(panel);
  panel.webview.html = workbench.secure(workbenchHtml(initialConfig, initialTrainingConfig,workbenchDefaults.workflowDefaults('prediction',{},projectRoot(context)),reactionPatternAnalysisUiSettings(),workbenchDefaults.workflowDefaults('trainable',{},projectRoot(context))), panel.webview);
  const panelContext = context;
  panel.webview.onDidReceiveMessage(async message => {
    if (message?.type === 'spectrumViewer') return;
    const context = projects.capture(panelContext);
    try {
      if (message.type === 'pick') {
        const folders = message.kind === 'folder';
        const picked = await vscode.window.showOpenDialog({ canSelectFiles: !folders, canSelectFolders: folders, canSelectMany: false });
        if (picked && picked[0]) {
          panel.webview.postMessage({ type: 'picked', form: message.form, field: message.field, value: picked[0].fsPath });
        }
      } else if (message.type === 'saveConfig') {
        const target = await vscode.window.showSaveDialog({ filters: { 'FREDIS training dataset': ['training-data.pft'] }, defaultUri: vscode.Uri.file('fragment-tree.training-data.pft') });
        if (target) {await fs.promises.writeFile(target.fsPath, JSON.stringify(normalizeConfig(message.config), null, 2) + '\n');projects.record(context,'data',message.config,{label:path.basename(target.fsPath),reason:'exported',sourcePath:target.fsPath,base:projectRoot(context)});}
      } else if (['loadConfig','loadConfigFile','loadConfigJSON'].includes(message.type)) {
        // Extension-agnostic: this always loads a data-preparation run
        // configuration by content (normalizeConfig), so any filename works.
        const picked = message.type==='loadConfigFile'?[vscode.Uri.file(message.path)]:message.type==='loadConfigJSON'?[{fsPath:message.name||'fragment-tree.training-data.pft'}]:await vscode.window.showOpenDialog({ canSelectMany: false });
        if (picked && picked[0]) {
          const config = normalizeConfig(JSON.parse(message.type==='loadConfigJSON'?message.json:await fs.promises.readFile(picked[0].fsPath, 'utf8')),message.type==='loadConfigJSON'?undefined:path.dirname(picked[0].fsPath));
          panel.webview.postMessage({ type: 'config', path: message.type==='loadConfigJSON'?undefined:picked[0].fsPath, config: normalizeConfig({ ...defaultConfig(context), ...config }) });
        }
      } else if (message.type === 'saveTrainingConfig') {
        const target = await vscode.window.showSaveDialog({ filters: { 'FREDIS training configuration': ['pfttrain.json'] }, defaultUri: vscode.Uri.file('fragment_tree.pfttrain.json') });
        if (target) {const config=normalizeTrainingConfig(message.config);await fs.promises.writeFile(target.fsPath, JSON.stringify(config, null, 2) + '\n');projects.record(context,'training',config,{label:path.basename(target.fsPath),reason:'exported',sourcePath:target.fsPath,base:projectRoot(context)});}
      } else if (['loadTrainingConfig','loadTrainingConfigFile','loadTrainingConfigJSON'].includes(message.type)) {
        if(message.type==='loadTrainingConfigJSON'&&(typeof message.json!=='string'||message.json.length>2000000))throw new Error('Training configuration is too large or invalid.');
        if(message.type==='loadTrainingConfigFile'&&(typeof message.path!=='string'||!message.path.trim()))throw new Error('Select a training configuration file.');
        // Extension-agnostic: this always loads a training run configuration
        // by content (normalizeTrainingConfig), so any filename works.
        const picked = message.type==='loadTrainingConfigFile'?[vscode.Uri.file(message.path)]:message.type==='loadTrainingConfigJSON'?[{fsPath:message.name||'fragment_tree.pfttrain.json'}]:await vscode.window.showOpenDialog({ canSelectMany: false });
        if (picked && picked[0]) {
          const raw=JSON.parse(message.type==='loadTrainingConfigJSON'?message.json:await fs.promises.readFile(picked[0].fsPath, 'utf8'));
          const config=normalizeTrainingConfig(raw);
          const restored={...defaultTrainingConfig(context),...config};
          restored.modelConfig=parameterService.merge(defaultTrainingConfig(context).modelConfig,config.modelConfig||{});
          panel.webview.postMessage({ type: 'trainingConfig', path: message.type==='loadTrainingConfigJSON'?undefined:picked[0].fsPath, config: restored });
        }
      } else if (message.type === 'openResult') {
        await openResultPicker();
      } else if (message.type === 'copyText') {
        await vscode.env.clipboard.writeText(String(message.text || ''));
      } else if (message.type === 'copyCommand') {
        const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
        const command = shellDisplay(python, buildArgs(normalizeConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'status', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'run') {
        const config=normalizeConfig(message.config),root=projectRoot(context),target=path.resolve(root,config.outputDir||'');
        if(!config.overwrite&&config.outputDir&&fs.existsSync(target)){
          const choice=await vscode.window.showWarningMessage(`Output directory already exists: ${target}\nOverwrite it before processing?`,{modal:true},'Overwrite');
          if(choice!=='Overwrite'){panel.webview.postMessage({type:'status',status:'idle',text:'Run cancelled. The output directory was not changed.'});return;}
          config.overwrite=true;
        }
        panel.webview.postMessage({type:'status',status:'running',text:'Validating datasets and configuration…'});
        await workbenchServices.validateRun(context,root,config);
        await runFragmentTree(context, output, config, panel);
      } else if (message.type === 'copyTrainingCommand') {
        const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
        const command = shellDisplay(python, buildTrainingArgs(normalizeTrainingConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'trainingStatus', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'training/pickIntegrationOutput') {
        const selected=await vscode.window.showSaveDialog({filters:{'FREDIS checkpoint':['pt']}});
        if(selected)panel.webview.postMessage({type:'training/integrationOutput',path:selected.fsPath});
      } else if (message.type === 'training/integrate') {
        try {
          const result=await require('./workbench/services').backend(context,projectRoot(context),'training-integrate',message,{timeoutMs:0});
          panel.webview.postMessage({type:'training/integrated',output:message.output,...result});
        } catch(error) { panel.webview.postMessage({type:'training/integrated',error:error.message}); }
      } else if (message.type === 'runTraining') {
        await runTraining(context, output, normalizeTrainingConfig(message.config), panel);
      } else if (message.type === 'stop') {
        if (runningProcess) workbench.stopChild(runningProcess,process.platform!=='win32');
      } else if (message.type === 'loadReactionPatternSet') {
        const source = await readReactionPatternImport(context, message);
        if (source) {
          const value = reactionPatternSetEditor.normalizeDocument(source.value);
          panel.webview.postMessage({ type: 'reactionPatternSet', value, path: source.path });
        }
      } else if (message.type === 'saveReactionPatternSet') {
        const value = reactionPatternSetEditor.normalizeDocument(message.value);
        const defaultUri = vscode.Uri.file(reactionPatternSetSavePath(value.reaction_pattern_set.name, message.path));
        const selected = await vscode.window.showSaveDialog({ filters: { 'FREDIS Reaction Pattern Set': ['pft'] }, defaultUri });
        if (selected) {
          const target = vscode.Uri.file(ensureFileSuffix(selected.fsPath, reactionPatternSetEditor.FILE_SUFFIX));
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(value, null, 2)}\n`);
          panel.webview.postMessage({ type: 'reactionPatternSetSaved', path: target.fsPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'loadReactionPattern') {
        const source = await readReactionPatternImport(context, message);
        if (source) {
          const pattern = normalizePattern(source.value);
          panel.webview.postMessage({ type: 'reactionPatternLoaded', path: source.path, pattern, index: message.index });
        }
      } else if (message.type === 'saveReactionPattern') {
        const pattern = normalizePattern(message.pattern);
        const selected = await vscode.window.showSaveDialog({ filters: { 'FREDIS Reaction Pattern': ['pft'] }, defaultUri: vscode.Uri.file(message.path || `${safeFileStem(pattern.name || 'pattern')}.reaction.pft`) });
        if (selected) {
          const target = vscode.Uri.file(ensureFileSuffix(selected.fsPath, '.reaction.pft'));
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(pattern, null, 2)}\n`);
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'loadAdductRuleSet' || message.type === 'loadAdductRule') {
        const source = await readReactionPatternImport(context, message);
        if (source) {
          if (message.type === 'loadAdductRuleSet') panel.webview.postMessage({type:'adductRuleSet',value:adductRuleSetEditor.normalizeDocument(source.value),path:source.path});
          else panel.webview.postMessage({type:'adductRuleLoaded',rule:adductRuleSetEditor.normalizeRule(source.value),index:message.index,path:source.path});
        }
      } else if (message.type === 'saveAdductRuleSet') {
        const value=adductRuleSetEditor.normalizeDocument(message.value),defaultUri=vscode.Uri.file(reactionPatternHost.reactionPatternSetSavePath(value.fragment_ion_adduct_rule_set.name,message.path,adductRuleSetEditor.FILE_SUFFIX));
        const selected=await vscode.window.showSaveDialog({filters:{'FREDIS Fragment Ion Adduct Rule Set':['pft']},defaultUri});
        if(selected){const target=vscode.Uri.file(ensureFileSuffix(selected.fsPath,adductRuleSetEditor.FILE_SUFFIX));await fs.promises.writeFile(target.fsPath,JSON.stringify(value,null,2)+'\n');panel.webview.postMessage({type:'adductRuleSetSaved',path:target.fsPath});vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);}
      } else if (message.type === 'saveAdductRule') {
        const rule=adductRuleSetEditor.normalizeRule(message.rule),selected=await vscode.window.showSaveDialog({filters:{'FREDIS Fragment Ion Adduct Rule':['pft']},defaultUri:vscode.Uri.file(safeFileStem(rule.name||'adduct_rule')+adductRuleSetEditor.RULE_SUFFIX)});
        if(selected){const target=vscode.Uri.file(ensureFileSuffix(selected.fsPath,adductRuleSetEditor.RULE_SUFFIX));await fs.promises.writeFile(target.fsPath,JSON.stringify(rule,null,2)+'\n');vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);}
      } else if (message.type === 'selectElements') {
        const elements = await showElementPicker(message.elements);
        if (elements) panel.webview.postMessage({ type: 'elementSelectionResult', requestId: message.requestId, elements });
      } else if (message.type === 'chemistry') {
        const result = await runChemistryBackend(context, message.command, message.payload, panel);
        panel.webview.postMessage({ type: 'chemistryResult', requestId: message.requestId, result });
      } else if (message.type === 'predictSpectrum') {
        try {
          const result = await runSpectrumPredictionBackend(context, { command: 'predict', payload: message.payload });
          panel.webview.postMessage({ type: 'predictSpectrumResult', requestId: message.requestId, result });
        } catch (error) {
          panel.webview.postMessage({ type: 'predictSpectrumError', requestId: message.requestId, message: String(error.message || error) });
        }
      } else if (message.type === 'stopSpectrumPrediction') {
        if (predictionProcess) workbench.stopChild(predictionProcess);
      } else if (message.type === 'listPredictAdducts') {
        try {
          const result = await runSpectrumPredictionBackend(context, { command: 'listAdducts', payload: message.payload });
          panel.webview.postMessage({ type: 'listPredictAdductsResult', requestId: message.requestId, result });
        } catch (error) {
          panel.webview.postMessage({ type: 'listPredictAdductsError', requestId: message.requestId, message: String(error.message || error) });
        }
      }
    } catch (error) {
      if (/^(load|save)ReactionPattern/.test(message.type)) panel.webview.postMessage({ type: 'reactionPatternError', message: String(error.message || error) });
      if (/^(load|save)Adduct/.test(message.type)) panel.webview.postMessage({ type: 'adductRuleError', message: String(error.message || error) });
      if (message.type === 'chemistry') panel.webview.postMessage({ type: 'chemistryError', requestId: message.requestId, message: String(error.message || error) });
      const trainingMessage = ['runTraining', 'copyTrainingCommand', 'saveTrainingConfig', 'loadTrainingConfig', 'loadTrainingConfigFile', 'loadTrainingConfigJSON'].includes(message.type);
      panel.webview.postMessage({ type: trainingMessage ? 'trainingStatus' : 'status', status: 'error', text: String(error.message || error) });
      vscode.window.showErrorMessage(`FREDIS: ${error.message || error}`);
    }
  });
  return panel;
}

function normalizePattern(value) { return reactionPatternSetEditor.normalizePattern(value); }

function reactionPatternSetSavePath(name, previousPath) {
  return reactionPatternHost.reactionPatternSetSavePath(name, previousPath, reactionPatternSetEditor.FILE_SUFFIX);
}

function runSpectrumPredictionBackend(context, payload) {
  return new Promise((resolve, reject) => {
    if(payload.command==='predict'&&predictionProcess){reject(new Error('A spectrum prediction is already running.'));return;}
    const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'spectrum-prediction', 'backend.py');
    let root;
    try { root = projectRoot(context); }
    catch (error) { reject(error); return; }
    if (!isFredisRoot(root)) { reject(new Error('The detected working directory is not a FREDIS application.')); return; }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    if(payload.command==='predict')predictionProcess=child;
    const observed=payload.command==='predict'?workbench.observeJob(context,{type:'prediction',name:String(payload.payload?.smiles||'Spectrum prediction').slice(0,72),child,command:[python,script],config:payload.payload,workflow:'prediction',base:projectRoot(context)}):null;
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { const text=chunk.toString();stderr += text;if(observed)observed.log(text); });
    child.on('error', error=>{if(predictionProcess===child)predictionProcess=undefined;if(observed)observed.finish({error:error.message});reject(error);});
    child.on('close', (code,signal) => {
      if(predictionProcess===child)predictionProcess=undefined;
      if(signal){if(observed)observed.finish({code,signal});reject(new Error('Spectrum prediction cancelled.'));return;}
      try { const response = JSON.parse(stdout.trim().split('\n').reverse().find(line=>line.startsWith('{'))||stdout);if(!response.ok)throw new Error(response.error);if(code!==0)throw new Error(stderr||'Prediction backend failed.');if(observed){observed.log('Predicted '+response.result.peaks.length+' peaks.\n');observed.finish({code:0,result:response.result});}resolve(response.result); }
      catch (error) {if(observed){observed.log(stderr||stdout||error.message);observed.finish({code,signal,error:signal?undefined:error.message});}reject(new Error(error.message||stderr||stdout||'The spectrum prediction backend did not return a response.'));}
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

function normalizeConfig(config, baseDirectory) {
  const result={...config,...config.workbench_config};
  delete result.workbench_config;
  const aliases={validation_input:'validationInput',validation_ratio:'validationRatio',validation_seed:'validationSeed',
    output_dir:'outputDir',smiles_column:'smilesColumn',adduct_type_column:'adductTypeColumn',collision_energy_column:'collisionEnergyColumn',
    precursor_mz_column:'precursorMzColumn',validation_smiles_column:'validationSmilesColumn',validation_adduct_type_column:'validationAdductTypeColumn',
    validation_collision_energy_column:'validationCollisionEnergyColumn',validation_precursor_mz_column:'validationPrecursorMzColumn',
    minimum_relative_intensity:'minimumRelativeIntensity',
    num_workers:'numWorkers',chunk_size:'chunkSize',max_unique_fragment_smiles:'maxUniqueFragmentSmiles',max_reaction_combinations:'maxReactionCombinations'};
  for(const [from,to] of Object.entries(aliases)){if(result[from]!==undefined&&result[to]===undefined)result[to]=result[from];delete result[from];}
  const chemistry=result.chemistry_config;delete result.chemistry_config;
  if(chemistry){result.fragmenterParams??=chemistry.fragmenter_params;result.symbols??=chemistry.symbols;result.reactionModels??=chemistry.reaction_models||[];}
  if(baseDirectory&&Array.isArray(result.reactionModels))result.reactionModels=result.reactionModels.map(record=>({...record,...(record.source?{source:path.resolve(baseDirectory,record.source)}:{})}));
  if(result.validationRatio==null)delete result.validationRatio;
  if(result.validationInput==null)result.validationInput='';
  result.input ||= result.trainInput;
  return result;
}

function normalizeTrainingConfig(config) {
  const result=JSON.parse(JSON.stringify(config||{}));
  const model=result.modelConfig;
  if(model&&typeof model==='object'&&!Array.isArray(model)){
    result.modelConfig={};
    for(const key of ['mol_encoder_params','reaction_model_params','spectrum_model_params','generation_params'])if(model[key]&&typeof model[key]==='object'&&!Array.isArray(model[key]))result.modelConfig[key]=model[key];
    if(result.modelConfig.mol_encoder_params)delete result.modelConfig.mol_encoder_params.symbols;
    if(!Object.keys(result.modelConfig).length)delete result.modelConfig;
  }
  for(const key of ['adduct_type_strs','fragmenter_params','mol_encoder_params','symbols'])delete result[key];
  delete result.params;
  delete result.initializeFrom;
  return result;
}

async function runTraining(context, output, config, panel) {
  if (runningProcess) throw new Error('Another FREDIS process is already running.');
  for (const key of ['trainDir', 'valDir', 'outputDir']) {
    if (!config[key]) throw new Error(`${key} is required.`);
  }
  if (config.initializeFrom && (config.resume || config.fineTuneCheckpoint)) throw new Error('Use pretrained weight initialization, resume, or frozen-base expansion separately.');
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
  await workbench.startTraining(context, panel, output, root, python, config, buildTrainingArgs);

}

async function runFragmentTree(context, output, config, panel) {
  if (runningProcess) throw new Error('Another FREDIS process is already running.');
  for (const key of ['input', 'outputDir']) if (!config[key]) throw new Error(`${key} is required.`);
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
  const resultPath = path.join(config.outputDir, 'train_structures', 'fragment-tree.training-data.pft');
  const args = buildArgs(config);
  const command = shellDisplay(python, args);
  output.clear(); output.show(true); output.appendLine(`$ ${command}`);
  panel.webview.postMessage({ type: 'status', status: 'running', text: 'Running CLI…', command });
  const startedAt = new Date().toISOString();
  runningProcess = spawn(python, args, { cwd: root, detached:process.platform!=='win32', env: { ...process.env, PYTHONUNBUFFERED:'1', PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) } });
  const observed=workbench.observeJob(context,{type:'preparation',name:path.basename(config.outputDir),outputDir:config.outputDir,child:runningProcess,detached:process.platform!=='win32',command:[python,...args],config,workflow:'data',base:root});
  const stream=chunk=>{const text=chunk.toString();observed.log(text);output.append(text);panel.webview.postMessage({type:'data/log',text});};
  runningProcess.stdout.on('data', stream);
  runningProcess.stderr.on('data', stream);
  runningProcess.on('error', async error => {
    observed.finish({error:error.message});
    runningProcess = undefined;
    panel.webview.postMessage({ type: 'status', status: 'error', text: error.message });
  });
  runningProcess.on('close', async (code,signal) => {
    runningProcess = undefined;
    observed.finish({code,signal});
    const status = observed.job.status;
    if (code === 0) {
      for(const split of ['train','validation']){
        const target=path.join(config.outputDir,split+'_structures','fragment-tree.training-data.pft');
        if(!fs.existsSync(target))continue;
        const saved=JSON.parse(await fs.promises.readFile(target,'utf8'));
        await fs.promises.writeFile(target,JSON.stringify({...saved,status:'completed',startedAt,finishedAt:new Date().toISOString(),exitCode:code,command:args},null,2)+'\n');
      }
    }
    panel.webview.postMessage({ type: 'status', status, text: signal ? 'Cancelled.' : code === 0 ? 'Completed.' : `Failed with exit code ${code}.`, resultPath });
    if (code === 0) {const validationPath=path.join(config.outputDir,'validation_structures','fragment-tree.training-data.pft');vscode.window.showInformationMessage('FREDIS fragment tree data preparation completed.', 'Open Train Result',...(fs.existsSync(validationPath)?['Open Validation Result']:[])).then(choice => { if (choice) openResult(vscode.Uri.file(choice==='Open Validation Result'?validationPath:resultPath)); });}
  });
}

function buildArgs(c) {
  const args=['-m', 'fredis.cli', 'train', 'create-fragment-tree-data',
    '--input', c.input || c.trainInput, '--output-dir', c.outputDir];
  if(c.fragmenterParams) args.push('--params-json',JSON.stringify(c.fragmenterParams));
  if(c.reactionModels?.length)args.push('--reaction-models-json',JSON.stringify(c.reactionModels));
  if(c.symbols)args.push('--symbols-json',JSON.stringify(c.symbols));
  for(const [key,flag]of Object.entries({maxUniqueFragmentSmiles:'--max-unique-fragment-smiles',maxReactionCombinations:'--max-reaction-combinations'}))if(c[key]!==undefined)args.push(flag,String(c[key]));
  for(const [key,flag] of Object.entries({smilesColumn:'--smiles-column',adductTypeColumn:'--adduct-type-column',collisionEnergyColumn:'--collision-energy-column',precursorMzColumn:'--precursor-mz-column'}))if(c[key])args.push(flag,c[key]);
  // Only emitted when the validation dataset genuinely needs a different column name than training.
  if(c.validationInput)for(const [key,flag,trainKey] of [['validationSmilesColumn','--validation-smiles-column','smilesColumn'],['validationAdductTypeColumn','--validation-adduct-type-column','adductTypeColumn'],['validationCollisionEnergyColumn','--validation-collision-energy-column','collisionEnergyColumn'],['validationPrecursorMzColumn','--validation-precursor-mz-column','precursorMzColumn']])if(c[key]&&c[key]!==c[trainKey])args.push(flag,c[key]);
  if(c.validationInput) args.push('--validation-input',c.validationInput);
  // With --validation-input this caps validation to this fraction of training's
  // unique SMILES after removing molecules shared with training, instead of
  // splitting a single dataset (which is what it does without --validation-input).
  if(c.validationRatio !== undefined && c.validationRatio !== '') args.push('--validation-ratio',String(c.validationRatio));
  for(const [key,flag] of Object.entries({validationSeed:'--validation-seed',minimumRelativeIntensity:'--minimum-relative-intensity',numWorkers:'--num-workers',chunkSize:'--chunk-size'}))if(c[key] !== undefined && c[key] !== '')args.push(flag,String(c[key]));
  for(const [key,flag] of Object.entries({overwrite:'--overwrite'}))if(c[key]!==undefined)args.push(flag,c[key]?'1':'0');
  return args;
}

function buildTrainingArgs(c) {
  if(c.tuning?.enabled){const trials=require('./workbench/tuning').plan(c,buildTrainingArgs);return ['-m','fredis.cli','train','tune-fragment-tree','--plan-json',JSON.stringify({outputDir:c.outputDir,trials})];}
  const a = ['-m', 'fredis.cli', 'train', 'fragment-tree',
    '--train-dir', c.trainDir, '--val-dir', c.valDir, '--output-dir', c.outputDir,
    '--epochs', String(c.epochs || 1), '--batch-size', String(c.batchSize || 4),
    '--device', c.device || 'cuda', '--lr', String(c.lr ?? 0.0001)];
  if(!c.resume&&!c.fineTuneCheckpoint){
    if(c.pretrainedPatterns?.length||c.modelConfig?.reaction_model_params?.pattern_model_params){const model=JSON.parse(JSON.stringify(c.modelConfig||{}));model.reaction_model_params??={};model.reaction_model_params.state_dropout=c.dropout??0;model.spectrum_model_params??={};a.push('--model-architecture-json',JSON.stringify({model_config:model,pretrained_patterns:c.pretrainedPatterns||[]}));}

    const sections={reaction_model_params:{hidden_dim:'reaction-hidden-dim',branch_main_adduct_dim:'reaction-main-adduct-dim',num_heads:'reaction-num-heads',state_num_layers:'reaction-state-layers'},spectrum_model_params:{"adduct_embedding_dim":"adduct-embedding-dim","collision_energy_dim": "collision-energy-feature-dim", "ce_fc_dims": "ce-fc-dims", "ce_center": "ce-center", "ce_scale": "ce-scale", "fragment_fc_dims": "fragment-fc-dims", "fragment_dropout": "fragment-dropout", "state_fc_dims": "state-fc-dims", "state_dropout": "state-dropout"},generation_params:{branch_path_threshold:'branch-path-threshold',max_fragment_nodes:'max-fragment-nodes',peak_intensity_threshold:'peak-intensity-threshold'},mol_encoder_params:{node_dim:'node-dim',graph_dim:'graph-dim',num_layers:'num-layers',num_heads:'num-heads',max_degree:'max-degree',max_spatial_dist:'max-spatial-dist',max_edge_dist:'max-edge-dist',dropout:'dropout'}};
    for(const [section,flags]of Object.entries(sections))for(const [key,flag]of Object.entries(flags)){const value=c.modelConfig?.[section]?.[key];if(value!==undefined&&value!==null)a.push('--'+flag,Array.isArray(value)?value.join(','):String(value));}
    const descriptors=c.modelConfig?.spectrum_model_params?.descriptors;if(Array.isArray(descriptors))a.push('--intensity-descriptors',...descriptors);
  }
  for (const [key, flag] of Object.entries(workbench.trainingFlags)) if (!(c.resume&&key==='dropout')&&c[key] !== undefined && c[key] !== '') a.push(flag, String(c[key]));
  if(c.cacheDatasetInMemory)a.push('--cache-dataset-in-memory');
  if(c.overwrite)a.push('--overwrite');
  if (c.initializeFrom) a.push('--initialize-from', c.initializeFrom);
  if (c.resume) a.push('--resume', c.resume);
  if (c.fineTuneCheckpoint && !c.resume) a.push('--fine-tune-checkpoint', c.fineTuneCheckpoint);
  if(c.headOnly&&c.fineTuneCheckpoint&&!c.resume)a.push('--head-only');
  return a;
}

function shellDisplay(program, args) { return [program, ...args].map(v => /^[A-Za-z0-9_./:=,-]+$/.test(v) ? v : `'${v.replace(/'/g, "'\\''")}'`).join(' '); }
async function openResultPicker() {
  // Extension-agnostic: the result viewer renders whatever JSON structure it
  // finds, so any filename works here too (unlike an Explorer double-click,
  // which VS Code itself gates by the registered customEditors extensions).
  const picked = await vscode.window.showOpenDialog({ canSelectMany: false });
  if (picked && picked[0]) openResult(picked[0]);
}
function openResult(uri) { return vscode.commands.executeCommand('vscode.openWith', uri, 'fredis.resultViewer'); }

class ReactionPatternAnalysisConfigEditorProvider {
  constructor(context, output) { this.context = context; this.output = output; }
  async openCustomDocument(uri) { return { uri, dispose() {} }; }
  async resolveCustomEditor(document, panel) {
    panel.title = 'FREDIS Workbench · Reaction Pattern Analysis';
    return resolveWorkbenchPanel(this.context, this.output, panel, 'reactionPatternAnalysis', document.uri.fsPath);
  }
}

class ResultEditorProvider {
  constructor(context, output) { this.context = context; this.output = output; }
  async openCustomDocument(uri) { return { uri, dispose() {} }; }
  async resolveCustomEditor(document, panel) {
    // The generic *.pft viewer reads JSON; route HDF5 Reaction Pattern Analysis results to their Workbench page.
    if (reactionPatternAnalysisResultView.isResultPath(document.uri.fsPath)) {
      panel.title = 'FREDIS Workbench · Reaction Pattern Analysis';
      return resolveWorkbenchPanel(this.context, this.output, panel, 'reactionPatternAnalysis', document.uri.fsPath);
    }
    panel.webview.options = { enableScripts: true };
    trainingMetrics.attach(panel);
    const refresh = async () => {
      try { panel.webview.html = await resultHtml(document.uri.fsPath); }
      catch (e) { panel.webview.html = errorHtml(e.message); }
    };
    let moleculePanel,selection=0;
    panel.onDidDispose(()=>{selection++;moleculePanel?.dispose();});
    panel.webview.onDidReceiveMessage(async m => {
      if (m.type === 'refresh') refresh();
      if (m.type === 'openFile') {
        const base = path.dirname(document.uri.fsPath), target = path.resolve(base, m.path);
        if (!(target === base || target.startsWith(base + path.sep))) return;
        if (target.endsWith('.preft.pt') || target.endsWith('.preft') || target.endsWith('.pt')) {
          panel.webview.postMessage({ type: 'structureLoading', path: m.path });
          try { panel.webview.postMessage({ type: 'structureDetail', result: await inspectFragmentTreeResult(this.context, target) }); }
          catch (error) { panel.webview.postMessage({ type: 'structureError', message: String(error.message || error) }); }
        } else vscode.commands.executeCommand('vscode.open', vscode.Uri.file(target));
      } else if (m.type === 'inspectMolecule') {
        if(!m.smiles)return;
        const request=++selection;
        try {
          const result=await runChemistryBackend(this.context,'depict',{smiles:m.smiles});
          if(request!==selection)return;
          if(!moleculePanel){moleculePanel=vscode.window.createWebviewPanel('fredis.fragmentMolecule','Fragment Molecule',{viewColumn:vscode.ViewColumn.Beside,preserveFocus:true},{enableScripts:true,retainContextWhenHidden:true});moleculePanel.onDidDispose(()=>{moleculePanel=undefined;});}
          moleculePanel.title='Fragment Node '+m.node;
          moleculePanel.webview.html=resultView.moleculeHtml(result,m.smiles,m.node,m.theme);
        }catch(error){panel.webview.postMessage({type:'structureError',message:String(error.message||error)});}
      } else if(m.type==='resultTheme') {
        moleculePanel?.webview.postMessage({type:'theme',theme:m.theme});
      } else if(m.type==='shell/document') {
        await workbenchServices.openDocumentation(this.context,projectRoot(this.context));
      } else if(m.type==='shell/github') {
        await vscode.env.openExternal(vscode.Uri.parse('https://github.com/MotohiroOGAWA/FREDIS'));
      }

    });
    await refresh();
  }
}

async function resultHtml(manifestPath) {
  const manifest = JSON.parse(await fs.promises.readFile(manifestPath, 'utf8'));
  if (manifest.schema === 'fredis.tuning-result') return require('./features/tuning-result/view').html(manifest,commonCss());
  if (manifest.schema === 'fredis.training-report') {
    return trainingReport.html({ manifest, root: path.dirname(manifestPath), reportPath: manifestPath }, commonCss());
  }
  const root = path.dirname(manifestPath);
  const files = await scan(root, root, 3, 500);
  const manifests = await readStructureManifests(root);
  const summary = summarize(files);
  summary.preft = manifests.reduce((total, item) => total + item.rows.filter(row => row.exists).length, 0);
  const assignmentScores=[];
  for(const item of manifests){
    let scoreFile=path.join(root,item.relativeBase,'assignment_scores.tsv');
    if(!fs.existsSync(scoreFile))scoreFile=path.join(root,item.relativeBase,'data','assignment_scores.tsv');
    if(!fs.existsSync(scoreFile))continue;
    const lines=(await fs.promises.readFile(scoreFile,'utf8')).trim().split(/\r?\n/),columns=lines.shift().split('\t'),byFile=new Map();
    for(const line of lines){const cells=line.split('\t'),value=key=>{const raw=cells[columns.indexOf(key)];if(!raw?.trim())return null;const number=Number(raw);return Number.isFinite(number)&&number>=0&&number<=1?number:null;};const entry={score:value('assignment_score'),withoutPrecursor:value('assignment_score_without_precursor')};assignmentScores.push(entry);const file=path.basename(cells[columns.indexOf('structure_file')]||'');if(!byFile.has(file))byFile.set(file,[]);byFile.get(file).push(entry);}
    for(const row of item.rows){const entries=byFile.get(path.basename(row.file||''))||[];for(const [field,key]of [['assignment_score','score'],['assignment_score_without_precursor','withoutPrecursor']])if(row[field]==null||row[field]===''){const values=entries.map(entry=>entry[key]).filter(value=>value!=null);row[field]=values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}}
  }
  const datasetStats = summarizeStructureManifests(manifests);
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}:root{--detected:#ff8a3d}.result-main{max-width:none;width:100%}.dataset-stat-cards{grid-template-columns:repeat(4,minmax(150px,1fr))}.dataset-stat-cards article small{display:block;margin-top:5px}.dataset-breakdown th:not(:first-child),.dataset-breakdown td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.tree-scroll{overflow:auto;max-height:70vh;overscroll-behavior:contain;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background);scrollbar-gutter:stable}.fragment-tree{display:block;width:100%;min-width:700px}.tree-edge{stroke:color-mix(in srgb,var(--vscode-editor-foreground) 35%,transparent);stroke-width:1.5}.tree-edge.detected{stroke:var(--detected);stroke-width:3}.tree-node circle{fill:var(--vscode-editor-background);stroke:var(--vscode-editor-foreground);stroke-width:1.5}.tree-node.detected circle{fill:color-mix(in srgb,var(--detected) 28%,var(--vscode-editor-background));stroke:var(--detected);stroke-width:3}.tree-node text{fill:var(--vscode-editor-foreground);font-size:9px;pointer-events:none}.tree-node{cursor:pointer}.molecule-preview{width:100%;max-height:360px;overflow:hidden;background:#fff;border-radius:6px}.molecule-preview svg{display:block;width:100%;height:auto}.chem-info{width:100%;font-size:11px}.chem-info th,.chem-info td{padding:5px}.used-adduct{background:color-mix(in srgb,var(--accent) 24%,transparent);outline:1px solid var(--accent)}.used-adduct.modified-state{background:color-mix(in srgb,#b06cff 30%,transparent);outline-color:#b06cff}.used-adduct td:first-child::before{content:'✓ ';color:var(--accent);font-weight:700}.used-adduct.modified-state td:first-child::before{color:#b06cff}.manifest-grid{margin-top:18px}.manifest-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}.manifest-controls input{min-width:260px;flex:1;padding:7px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border);border-radius:5px}.manifest-controls label{display:flex;align-items:center;gap:6px}.manifest-controls select{width:auto;margin:0}.manifest-scroll{max-height:none;overflow-x:auto;overflow-y:visible}.column-filters input{width:100%;min-width:80px;padding:4px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border)}.missing-file{display:block;color:var(--vscode-errorForeground)}.tree-controls,.detail-controls{display:flex;justify-content:flex-end;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0}.tree-controls select{width:auto}.tree-controls button{padding:5px 9px}.detail-controls input{width:180px;margin:0}.sortable{cursor:pointer;user-select:none}.sortable:hover{color:var(--accent)}.detail-layout{display:grid;grid-template-columns:minmax(0,1fr) var(--molecule-width,35%);gap:18px;align-items:start}.molecule-side{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto;border:1px solid var(--border);border-radius:8px;padding:14px}.molecule-side[hidden]{display:block;visibility:hidden}.molecule-side code{display:block;overflow-wrap:anywhere;margin:8px 0}.target-diagnostic{font-size:11px;padding:3px 0;border:0}.target-diagnostic summary{font-weight:400}.target-diagnostic code{display:block;white-space:normal;margin:2px 0}@media(max-width:900px){.dataset-stat-cards{grid-template-columns:repeat(2,minmax(140px,1fr))}.detail-layout{grid-template-columns:1fr}.molecule-side{position:static;max-height:none}.molecule-side[hidden]{display:none}}${resultView.css}</style></head><body><div class="result-toolbar"><strong>FREDIS Workbench <small> / Fragment Tree Dataset Results</small></strong><button id="resultDocs">Documentation</button><button id="resultGithub">GitHub</button><button id="resultTheme">☀ / ☾</button><button id="resultHelp">Help</button></div><main class="result-main"><p id="resultHelpText" class="result-help" hidden>Select an output file to inspect its saved fragment trees. Node labels are fragment node IDs; diamonds mark precursors. Click nodes to open the molecule window, or edges to inspect the added reaction. Expand reaction sets and follow Reaction links to the atom-map reference.</p>
    <header><div><span class="eyebrow">FREDIS RESULT</span><h1>Fragment Tree Dataset Results</h1><p class="muted">${escapeHtml(root)}</p></div><button onclick="vscode.postMessage({type:'refresh'})">Refresh</button></header>
    <section class="cards result-overview"><article><b>${summary.preft}</b><span>Generated structures</span></article><article><b>${datasetStats.total.totalSpectra}</b><span>Total records</span></article><article><b>${datasetStats.total.totalCompounds}</b><span>Sources / SMILES</span></article><article><b>${datasetStats.total.invalidSpectra}</b><span>Skipped records</span></article><article><b>${datasetStats.datasets.length}</b><span>Datasets</span></article><article><b>${escapeHtml(manifest.status||'unknown')}</b><span>Job status</span></article></section>
    ${datasetStatisticsHtml(datasetStats)}
    ${resultView.histogramHtml(assignmentScores)}
    <section><details><summary>Run Information</summary><dl><dt>Application</dt><dd>${escapeHtml(manifest.application || '')}</dd><dt>Finished</dt><dd>${escapeHtml(manifest.finishedAt || '—')}</dd><dt>Exit code</dt><dd>${escapeHtml(String(manifest.exitCode ?? '—'))}</dd><dt>Command</dt><dd><code>${escapeHtml((manifest.command || []).join(' '))}</code></dd></dl></details></section>
    <section><h2>Output Files</h2><p class="muted">Select a generated structure to inspect its Source, fragment tree and reaction reference.</p>${manifests.map((manifest,index)=>manifestTableHtml(manifest,index)).join('') || '<p>No structure manifest was found.</p>'}</section>
    <section><details><summary>Supporting Files <small>${files.length} files</small></summary><div class="files">${files.map(f => `<button class="file" data-open-file="${encodeURIComponent(f.relative)}"><span>${escapeHtml(f.relative)}</span><em>${formatBytes(f.size)}</em></button>`).join('')}</div></details></section>
    <section id="detail" hidden></section>
    <script>const vscode=acquireVsCodeApi(),detail=document.getElementById('detail'),sortState=new Map();const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    ${resultView.script()}
    document.addEventListener('click',e=>{const file=e.target.closest('[data-open-file]');if(file)vscode.postMessage({type:'openFile',path:decodeURIComponent(file.dataset.openFile)})});
    const manifestStates=new WeakMap();
    function updateManifest(grid,resetPage=false){const state=manifestStates.get(grid)||{page:0,sortColumn:-1,sortDirection:1};if(resetPage)state.page=0;const search=grid.querySelector('[data-manifest-search]').value.toLowerCase(),filters=[...grid.querySelectorAll('[data-column-filter]')].map(input=>input.value.toLowerCase()),body=grid.querySelector('tbody'),rows=[...body.rows].filter(row=>{const values=[...row.cells].map(cell=>(cell.dataset.value||'').toLowerCase());return(!search||values.some(value=>value.includes(search)))&&filters.every((filter,index)=>!filter||values[index].includes(filter))});if(state.sortColumn>=0)rows.sort((a,b)=>{const left=a.cells[state.sortColumn].dataset.value||'',right=b.cells[state.sortColumn].dataset.value||'',ln=Number(left),rn=Number(right),result=left!==''&&right!==''&&Number.isFinite(ln)&&Number.isFinite(rn)?ln-rn:left.localeCompare(right,undefined,{numeric:true,sensitivity:'base'});return state.sortDirection*result});for(const row of rows)body.appendChild(row);const pageSize=Number(grid.querySelector('[data-page-size]').value),pages=Math.max(1,Math.ceil(rows.length/pageSize));state.page=Math.min(state.page,pages-1);const pageRows=new Set(rows.slice(state.page*pageSize,(state.page+1)*pageSize));for(const row of body.rows)row.hidden=!pageRows.has(row);grid.querySelector('[data-page-label]').textContent=(rows.length?state.page*pageSize+1:0)+'–'+Math.min((state.page+1)*pageSize,rows.length)+' / '+rows.length+' · page '+(state.page+1)+'/'+pages;grid.querySelector('[data-page-prev]').disabled=state.page===0;grid.querySelector('[data-page-next]').disabled=state.page>=pages-1;manifestStates.set(grid,state)}
    document.querySelectorAll('[data-manifest-grid]').forEach(grid=>updateManifest(grid));
    document.addEventListener('input',e=>{const grid=e.target.closest('[data-manifest-grid]');if(grid&&(e.target.matches('[data-manifest-search]')||e.target.matches('[data-column-filter]')))updateManifest(grid,true)});
    document.addEventListener('change',e=>{const grid=e.target.closest('[data-manifest-grid]');if(grid&&e.target.matches('[data-page-size]'))updateManifest(grid,true)});
    document.addEventListener('click',e=>{const grid=e.target.closest('[data-manifest-grid]');if(!grid)return;const state=manifestStates.get(grid);if(e.target.closest('[data-page-prev]'))state.page--;else if(e.target.closest('[data-page-next]'))state.page++;else{const header=e.target.closest('[data-manifest-sort]');if(!header)return;const column=Number(header.dataset.manifestSort);state.sortDirection=state.sortColumn===column?-state.sortDirection:1;state.sortColumn=column;grid.querySelectorAll('[data-manifest-sort]').forEach(item=>item.textContent=item.dataset.label+' ↕');header.textContent=header.dataset.label+(state.sortDirection>0?' ↑':' ↓');state.page=0}updateManifest(grid)});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='structureLoading'){detail.hidden=false;detail.innerHTML='<h2>Loading structure…</h2><p class="muted">'+esc(m.path)+'</p>'}if(m.type==='structureDetail')render(m.result);if(m.type==='structureError')detail.innerHTML='<h2>Unable to inspect structure</h2><pre>'+esc(m.message)+'</pre>'});</script></main></body></html>`;
}

async function readStructureManifests(root) {
  const results = [];
  const localManifest = path.join(root, 'manifest.tsv');
  if (fs.existsSync(localManifest)) {
    const lines = (await fs.promises.readFile(localManifest, 'utf8')).trim().split(/\r?\n/);
    if (lines.length && lines[0]) {
      const columns = lines[0].split('\t');
      const rows = lines.slice(1).map(line => {
        const values = line.split('\t'), row = Object.fromEntries(columns.map((column, index) => [column, values[index] || '']));
        row.relative = row.file ? path.join('data', row.file) : '';
        row.exists = Boolean(row.relative && fs.existsSync(path.join(root, row.relative)));
        return row;
      });
      results.push({ directory: path.basename(root), relativeBase: '', rows });
    }
  }
  for (const directory of ['train_structures', 'validation_structures']) {
    const file = path.join(root, directory, 'manifest.tsv');
    if (!fs.existsSync(file)) continue;
    const lines = (await fs.promises.readFile(file, 'utf8')).trim().split(/\r?\n/);
    if (!lines.length || !lines[0]) continue;
    const columns = lines[0].split('\t');
    const rows = lines.slice(1).map(line => {
      const values = line.split('\t');
      const row = Object.fromEntries(columns.map((column, index) => [column, values[index] || '']));
      row.relative = row.file ? path.join(directory, 'data', row.file) : '';
      row.exists = Boolean(row.relative && fs.existsSync(path.join(root, row.relative)));
      return row;
    });
    results.push({ directory, relativeBase: directory, rows });
  }
  // Older preparation outputs did not write manifest.tsv. Inspect their saved metadata.
  for(const directory of ['', 'train_structures','validation_structures']){
    if(results.some(item=>item.relativeBase===directory))continue;
    const folder=path.join(root,directory);if(!fs.existsSync(folder))continue;
    const locations=[folder,path.join(folder,'data')];let hasStructures=false;
    for(const location of locations)if(fs.existsSync(location)&&(await fs.promises.readdir(location)).some(name=>name.endsWith('.preft.pt')))hasStructures=true;
    if(!hasStructures)continue;
    const context={extensionPath:path.resolve(__dirname,'..')};
    const rows=await workbenchServices.backend(context,projectRoot(context),'structure-manifest',{directory:folder});
    for(const row of rows){row.relative=path.join(directory,row.file);row.exists=fs.existsSync(path.join(root,row.relative));}
    results.push({directory:directory||path.basename(root),relativeBase:directory,rows});
  }
  for(const manifest of results){
    const missing=manifest.rows.filter(row=>row.exists&&['num_nodes','num_edges'].some(key=>row[key]===undefined||row[key]===''));
    if(missing.length){
      const context={extensionPath:path.resolve(__dirname,'..')};
      const summaries=await workbenchServices.backend(context,projectRoot(context),'structure-manifest',{directory:path.join(root,manifest.relativeBase)});
      const byFile=new Map(summaries.map(row=>[path.basename(row.file),row]));
      for(const row of missing){const summary=byFile.get(path.basename(row.file));if(summary)for(const [key,value] of Object.entries(summary))if(row[key]===undefined||row[key]==='')row[key]=value;}
    }
    for(const row of manifest.rows){
      if(row.rejected_sample_count==null||row.rejected_sample_count==='')row.rejected_sample_count=Math.max(0,manifestCount(row,'num_input_records','record_indexes')-manifestCount(row,'num_valid_samples','sample_indexes',true));
      if(!row.rejection_log&&Number(row.rejected_sample_count)>0&&fs.existsSync(path.join(root,manifest.relativeBase,'skipped_sources.json')))row.rejection_log='skipped_sources.json';
      if(!row.exists&&row.status==='skipped'){for(const key of ['num_nodes','num_edges'])if(row[key]==null||row[key]==='')row[key]=0;}
    }
  }
  return results;
}

function manifestListLength(value, validOnly = false) {
  if (!value) return 0;
  try {
    const items = JSON.parse(value);
    if (!Array.isArray(items)) return 0;
    return validOnly ? items.filter(item => Number(item) >= 0).length : items.length;
  } catch (_) { return 0; }
}

function manifestCount(row, key, fallbackList, validOnly = false) {
  const raw = row[key], value = Number(raw);
  return raw !== '' && raw !== null && raw !== undefined && Number.isFinite(value) && value >= 0
    ? value : manifestListLength(row[fallbackList], validOnly);
}

function summarizeStructureManifests(manifests) {
  const summarizeRows = (dataset, rows) => {
    let validCompounds = 0, totalSpectra = 0, validSpectra = 0;
    const validSpectraPerCompound = [];
    for (const row of rows) {
      const inputCount = manifestCount(row, 'num_input_records', 'record_indexes');
      const validCount = manifestCount(row, 'num_valid_samples', 'sample_indexes', true);
      const status = String(row.status || '').toLowerCase();
      const validCompound = validCount > 0 && !['error', 'failed', 'no_valid_samples'].includes(status);
      totalSpectra += inputCount;
      validSpectra += validCount;
      if (validCompound) {
        validCompounds++;
        validSpectraPerCompound.push(validCount);
      }
    }
    validSpectraPerCompound.sort((a, b) => a - b);
    const middle = Math.floor(validSpectraPerCompound.length / 2);
    const median = validSpectraPerCompound.length
      ? (validSpectraPerCompound.length % 2 ? validSpectraPerCompound[middle] : (validSpectraPerCompound[middle - 1] + validSpectraPerCompound[middle]) / 2)
      : 0;
    return {
      dataset, totalCompounds: rows.length, validCompounds,
      invalidCompounds: rows.length - validCompounds,
      totalSpectra, validSpectra, invalidSpectra: Math.max(0, totalSpectra - validSpectra),
      spectraPerCompoundMedian: median,
      spectraPerCompoundMean: validSpectraPerCompound.length ? validSpectra / validSpectraPerCompound.length : 0
    };
  };
  const datasets = manifests.map(manifest => summarizeRows(manifest.directory, manifest.rows));
  const total = summarizeRows('All datasets', manifests.flatMap(manifest => manifest.rows));
  return { total, datasets };
}

function statsPercent(valid, total) { return total ? `${(valid / total * 100).toFixed(1)}%` : '—'; }
function statsNumber(value) { return Number(value).toLocaleString('en-US', { maximumFractionDigits: 1 }); }
function datasetStatisticsHtml(stats) {
  const total = stats.total;
  const rows = [total, ...stats.datasets].map(item => `<tr><td>${escapeHtml(item.dataset)}</td><td>${statsNumber(item.totalCompounds)}</td><td>${statsNumber(item.validCompounds)}</td><td>${statsNumber(item.invalidCompounds)}</td><td>${statsPercent(item.validCompounds, item.totalCompounds)}</td><td>${statsNumber(item.totalSpectra)}</td><td>${statsNumber(item.validSpectra)}</td><td>${statsNumber(item.invalidSpectra)}</td><td>${statsPercent(item.validSpectra, item.totalSpectra)}</td><td>${statsNumber(item.spectraPerCompoundMedian)} / ${statsNumber(item.spectraPerCompoundMean)}</td></tr>`).join('');
  return `<section class="dataset-statistics"><h2>Dataset Statistics</h2><p class="muted">A valid compound has at least one valid spectrum and no failed status. Spectrum counts are summed from each structure-manifest row.</p><div class="cards dataset-stat-cards"><article><b>${statsNumber(total.totalCompounds)}</b><span>Total compounds</span></article><article><b>${statsNumber(total.validCompounds)}</b><span>Valid compounds</span><small>${statsPercent(total.validCompounds, total.totalCompounds)} of total</small></article><article><b>${statsNumber(total.totalSpectra)}</b><span>Total spectra</span></article><article><b>${statsNumber(total.validSpectra)}</b><span>Valid spectra</span><small>${statsPercent(total.validSpectra, total.totalSpectra)} of total</small></article></div><div class="table-scroll"><table class="dataset-breakdown"><thead><tr><th>Dataset</th><th>Total compounds</th><th>Valid compounds</th><th>Rejected compounds</th><th>Compound validity</th><th>Total spectra</th><th>Valid spectra</th><th>Rejected spectra</th><th>Spectrum validity</th><th>Valid spectra / compound<br><small>median / mean</small></th></tr></thead><tbody>${rows}</tbody></table></div>${stats.datasets.length ? '' : '<p class="muted">No structure manifest was found, so dataset counts are unavailable.</p>'}</section>`;
}

function manifestTableHtml(manifest, index) {
  const columns = ['file', 'status', 'smiles', 'num_input_records', 'num_valid_samples', 'rejected_sample_count', 'rejection_log', 'num_branch_groups', 'num_teacher_nodes', 'num_transition_states', 'num_positive_transitions', 'num_ion_states', 'max_teacher_reaction_count', 'num_nodes', 'num_edges', 'assignment_score', 'assignment_score_without_precursor', 'num_primitive_reactions', 'num_raw_combinations', 'num_compiled_sequences', 'num_rdkit_run_reactants', 'num_generated_fragments', 'num_unique_fragment_smiles', 'skip_category', 'reason'];
  const rows = manifest.rows.map(row => `<tr>${columns.map(column => { const value = row[column] ?? ''; if(column.startsWith('assignment_score')) return `<td data-value="${escapeHtml(value)}" title="Mean per-sample intensity coverage">${value!==''&&Number.isFinite(Number(value))?(Number(value)*100).toFixed(2)+'%':'Unavailable'}</td>`; if(column==='rejection_log'&&!value)return '<td data-value="">—</td>'; if (column === 'file' && value && row.exists) return `<td data-value="${escapeHtml(value)}"><button class="manifest-file" data-open-file="${encodeURIComponent(row.relative)}">${escapeHtml(value)}</button></td>`; if (column === 'file' && value) return `<td data-value="${escapeHtml(value)}"><span>${escapeHtml(value)}</span><small class="missing-file">not generated (${escapeHtml(row.status || 'missing')})</small></td>`; if (column === 'rejection_log' && value) return `<td data-value="${escapeHtml(value)}"><button class="manifest-file" data-open-file="${encodeURIComponent(path.join(manifest.relativeBase || '', value))}">${escapeHtml(value)}</button></td>`; return `<td data-value="${escapeHtml(value)}">${escapeHtml(value)}</td>`; }).join('')}</tr>`).join('');
  return `<div class="manifest-grid" data-manifest-grid="${index}"><h3>${escapeHtml(manifest.directory)}</h3><div class="manifest-controls"><input type="search" data-manifest-search placeholder="Filter all columns…"><label>Rows <select data-page-size><option>20</option><option selected>50</option><option>100</option><option>250</option></select></label><button data-page-prev>Previous</button><span data-page-label></span><button data-page-next>Next</button></div><div class="table-scroll manifest-scroll"><table><thead><tr>${columns.map((column,columnIndex) => `<th class="sortable" data-manifest-sort="${columnIndex}" data-label="${escapeHtml(column)}">${escapeHtml(column)} ↕</th>`).join('')}</tr><tr class="column-filters">${columns.map((column,columnIndex) => `<th><input data-column-filter="${columnIndex}" placeholder="Filter…" aria-label="Filter ${escapeHtml(column)}"></th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function inspectFragmentTreeResult(context, target) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'fragment-tree-result', 'backend.py');
    let root; try { root = projectRoot(context); } catch (error) { reject(error); return; }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', reject);
    child.on('close', () => { try { const response = JSON.parse(stdout); response.ok ? resolve(response.result) : reject(new Error(response.error)); } catch (_) { reject(new Error(stderr || stdout || 'Structure inspector returned no JSON response.')); } });
    child.stdin.end(JSON.stringify({ path: target }));
  });
}

async function scan(root, dir, depth, limit, out = []) {
  if (depth < 0 || out.length >= limit) return out;
  for (const entry of await fs.promises.readdir(dir, { withFileTypes: true })) {
    if (entry.name === RESULT_NAME || entry.name === 'fragment-tree.pft' || entry.name === 'fragment-tree.fredis-result' || entry.name.startsWith('.')) continue;
    if (entry.isDirectory() && entry.name === 'data' && /(?:train|validation)_structures$/.test(path.basename(dir))) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await scan(root, full, depth - 1, limit, out);
    else { const stat = await fs.promises.stat(full); out.push({ relative: path.relative(root, full), size: stat.size }); }
    if (out.length >= limit) break;
  }
  return out.sort((a,b) => a.relative.localeCompare(b.relative));
}
function summarize(files) { return files.reduce((s,f) => { s.bytes += f.size; if (f.relative.endsWith('.preft.pt') || f.relative.endsWith('.preft') || f.relative.endsWith('.pt')) s.preft++; if (f.relative.endsWith('.tsv')) s.tsv++; return s; }, {preft:0,tsv:0,bytes:0}); }
function formatBytes(n) { if (n < 1024) return `${n} B`; if (n < 1048576) return `${(n/1024).toFixed(1)} KB`; return `${(n/1048576).toFixed(1)} MB`; }
function escapeHtml(v) { return String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function errorHtml(message) { return `<!doctype html><style>${commonCss()}</style><main><h1>Unable to display result</h1><pre>${escapeHtml(message)}</pre></main>`; }

const HELP = {
  trainInput: 'Training input MSDataset path. Required.', validationInput: 'Optional validation input MSDataset path.',
  params: 'Fragmenter parameter JSON used for preprocessing. Required.', outputDir: 'Output root directory. Required.',
  symbols: 'Element symbols that define atom-feature columns. Required and immutable for generated data.',
  maxUniqueFragmentSmiles: 'Skip a source molecule once its generated fragments exceed this many distinct canonical SMILES (source not counted). Use -1 for no limit.', maxReactionCombinations: 'Skip a source molecule once its reaction search examines more than this many raw reaction combinations, before RDKit runs them. Use -1 for no limit.',
  smilesColumn: 'Metadata column containing SMILES strings. Default: SMILES.', precursorMzColumn: 'Metadata column containing precursor m/z values. Default: PrecursorMZ.',
  adductTypeColumn: 'Metadata column containing adduct types. Default: AdductType.', collisionEnergyColumn: 'Metadata column containing collision energies. Default: CollisionEnergy.',
  instrumentColumn: 'Optional metadata column containing instrument names. Leave blank when unavailable; for NIST-style data it is commonly InstrumentType.', validationSmilesRatio: 'Target validation SMILES count relative to unique training SMILES. Default: 0.1.',
  tanimotoNumBins: 'Number of similarity intervals used for balanced validation sampling. Default: 10.', tanimotoRadius: 'Morgan fingerprint radius used for validation similarity. Default: 2.',
  tanimotoNBits: 'Morgan fingerprint bit count. Default: 2048.', validationSamplingSeed: 'Random seed for deterministic validation sampling. Default: 0.',
  validationStructuresInputDir: 'Optional existing validation structure directory; mutually exclusive with validation input.',
  numWorkers: 'Number of parallel subprocess workers. Use 1 to disable parallel processing.', chunkSize: 'Number of SMILES groups assigned to each parallel chunk.',
  structureRebuildPolicy: 'When reusing an existing structure directory, decide whether newly added reaction patterns require rebuilding: match any saved fragment (recommended), match only the root molecule, or always rebuild. An existing output is still skipped unless Overwrite structures is enabled.',
  overwrite: 'Overwrite existing .preft.pt structure files.', overwritePreprocessingConfig: 'Overwrite an existing preprocessing configuration without prompting.',
  saveTrainValidRecords: 'Save successfully processed training records as an MSDataset.', saveValidationValidRecords: 'Save successfully processed validation records as an MSDataset.',
  keepParallelTemp: 'Keep temporary parallel-processing files after merging.'
  ,trainDir: 'Directory containing generated training structures and preprocessing_config.pftprep.json.',
  valDir: 'Directory containing generated validation structures with compatible preprocessing settings.',
  initializeFrom: 'Initialize compatible Source-anchored model weights for fine-tuning with a fresh optimizer.',
  dropout: 'Dropout shared by the reaction-state encoder and the spectrum model. Mol Encoder dropout is configured in its architecture section.',
  assignmentScoreThreshold: 'Minimum assignment score accepted for training and the filtered validation view. Reads assignment_scores.tsv from each split. Default: 0.8. Validation also evaluates all samples by combining disjoint above- and below-threshold results without repeating inference.',
  maxSamples: 'Maximum spectra packed into one loaded compound batch.',
  minimumAssignmentScore: 'Drop a prepared sample at training load time if its assignment score (fraction of matched peak intensity, all peaks) is below this value. Data preparation keeps every sample; this only affects what training actually loads. Use 0 to keep every sample.',
  minimumAssignmentScoreWithoutPrecursor: 'Drop a prepared sample at training load time if its assignment score excluding the precursor peak is below this value. Use 0 to keep every sample.',
  cacheDatasetInMemory: 'Keep each prepared tensor in RAM after its first training or validation access. Disabled by default; streaming retains only the current source.',
  experimentName: 'Checkpoint experiment directory name.', ckptId: 'Optional checkpoint identifier to resume.',
  batchSize: 'Number of prepared compound batches per optimizer step.', device: 'PyTorch execution device.',
  epochs: 'Number of complete training epochs.', validationIntervalSteps: 'Run validation after this many optimizer steps; use 0 to disable step validation.',
  stepValidationFraction: 'Fraction of validation compounds (SMILES) used at step intervals: 1 = all, 0.1 = 10%. Uses a fixed subset, rounded up to at least one compound. Epoch-end validation always evaluates all compounds.',
  trainLogIntervalSteps: 'Write training metrics after this many steps.', saveIntervalEpochs: 'Save a checkpoint after this many epochs.',
  saveIntervalSteps: 'Save a checkpoint after this many steps; use 0 to disable.', earlyStoppingPatience: 'Stop after this many non-improving validations; blank disables it.',
  optimizer: 'PyTorch optimizer class name.', lr: 'Optimizer learning rate.', weightDecay: 'Optimizer weight decay.', gradClipNorm: 'Maximum gradient norm; non-positive disables clipping.',
  validateAtStart: 'Run validation before the first training update.', detectAnomaly: 'Enable PyTorch autograd anomaly detection.',
  profilePerformance: 'Write detailed runtime performance profiles.', shuffle: 'Shuffle training structure batches each epoch.'
  ,modelPath: 'Trained fragment-tree model checkpoint (model.pt or a raw state_dict) used for spectrum prediction.'
};

function workbenchHtml(config, trainingConfig, predictionConfig = {}, reactionPatternAnalysisUi = {}, trainableConfig = {}) {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}${formCss()}${trainingCss()}${trainingWorkbench.css()}${pathDrop.css()}${workbench.css()}${parameterEditor.css()}${fragmentTreeVisualization.css()}${spectrumViewer.css()}${reactionPatternAnalysis.css()}</style></head><body><main>
  ${layout.header()}
  ${workbench.html()}<nav id="appNavigation"><button class="tab" data-app="smarts">SMARTS Search</button><button class="tab" data-app="fragmentTree">Fragment Tree</button><button class="tab" data-app="reactionViewer">Reaction Viewer</button><button class="tab" data-app="reactionPatternAnalysis">Reaction Pattern Analysis</button><button class="tab" data-app="reactionPatterns">Reaction Pattern Set</button><button class="tab" data-app="adduct">Adduct Rule Set</button><button class="tab active" data-app="data">Data Preparation</button><button class="tab" data-app="training">Training</button><button class="tab" data-app="predict">Predict Spectrum</button></nav>
  ${spectrumViewer.html()}
  ${smartsSearch.html()}
  ${fragmentTreeVisualization.html()}
  ${reactionPatternAnalysis.html()}
  <div id="reactionViewerApp" data-app-panel="reactionViewer" hidden><section><div class="section-title"><div><h2>Single-SMILES Reaction Viewer</h2><p class="muted">Use the patterns currently loaded in Reaction Patterns. Enumerate valid simultaneous reaction sets or add compatible reactions interactively.</p></div><button type="button" id="reactionViewerLoadPatterns">Load Pattern Set</button></div></section><div id="reactionViewerMount"></div></div>
  <div id="reactionPatternPageMount">${reactionPatternUi.html()}</div>
  <div id="adductPageMount">${adductUi.html()}</div>
  ${preparation.html()}
  ${trainingWorkbench.html({pathField,field,optimization:workbench.trainingHtml(),parameters:parameterEditor.importHtml('training'),defaults:workbenchDefaults.html('training')})}
  <form id="predictForm" data-app-panel="predict" class="preparation-page prediction-page" novalidate hidden>
    ${spectrumPrediction.html()}${workbenchDefaults.html('prediction')}
  </form><div id="helpTooltip" role="tooltip"></div><script>const vscode=acquireVsCodeApi();window.fredisMsentityBridge={postMessage:message=>vscode.postMessage({type:'spectrumViewer',message})}; const initial=${safeJson(config)}; const initialTraining=${safeJson(trainingConfig)};const initialPrediction=${safeJson(predictionConfig)};const initialTrainable=${safeJson(trainableConfig)}; ${pathDrop.script({host:true})}${webviewScript()}${reactionPatternVisualEditor.script()}${reactionPatternPreview.script()}${spectrumPrediction.script()}${workbench.script()}${parameterEditor.script()}${fragmentTreeVisualization.script()}${reactionPatternAnalysis.script(reactionPatternAnalysisUi)}${layout.script()}${datasetUpload.script()}${preparation.script()}${workbenchDefaults.script()}${trainingWorkbench.script()}${projects.script()}${spectrumViewer.script()}</script></main></body></html>`;
}
function pathField(name,label,kind,form='data') { return `<label data-help="${HELP[name] || ''}">${label}<div class="path"><input name="${name}" data-path-kind="${kind}"><button type="button" data-pick="${name}" data-kind="${kind}" data-form="${form}">Browse</button></div>${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</label>`; }
function field(name,label,type,step='1') { return `<label data-help="${HELP[name] || ''}">${label}<input name="${name}" type="${type}"${type==='number'?` step="${step}"`:''}>${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</label>`; }
function check(name,label) { return `<label class="check" data-help="${HELP[name] || ''}"><input name="${name}" type="checkbox"><span>${label}${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</span></label>`; }
function safeJson(value) { return JSON.stringify(value).replace(/</g, '\\u003c'); }
function webviewScript() { return `
    const form=document.getElementById('form'), trainingForm=document.getElementById('trainingForm'), predictForm=document.getElementById('predictForm'), statusEl=document.getElementById('status'), stop=document.getElementById('stop'),trainingStatusEl=document.getElementById('trainingStatus'),trainingStop=document.getElementById('trainingStop');
    ${reactionPatternUi.state()}
    ${adductUi.state()}
    ${smartsSearch.script()}
    const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function setFormConfig(target,c){for(const [k,v] of Object.entries(c)){const el=target.elements[k];if(!el)continue;if(el.type==='checkbox')el.checked=!!v;else el.value=Array.isArray(v)?v.join(target===trainingForm?',':' '):v??'';}}
    function readForm(target,application){const c={application};for(const el of target.elements){if(!el.name)continue;if(el.type==='checkbox')c[el.name]=el.checked;else if(el.type==='number')c[el.name]=el.value===''?'':Number(el.value);else c[el.name]=el.value;}return c;}
    let extraConfig={};
    function setConfig(c){extraConfig={...c};setFormConfig(form,c)}
    function getConfig(){return {...extraConfig,...readForm(form,'fragment-tree-data-preparation')};}
    function getTrainingConfig(){return readForm(trainingForm,'fragment-tree-training')}
    const tooltip=document.getElementById('helpTooltip'); let tooltipTimer;
    document.querySelectorAll('[data-help]').forEach(el=>{el.addEventListener('mouseenter',()=>{tooltipTimer=setTimeout(()=>{const r=el.getBoundingClientRect();tooltip.textContent=el.dataset.help;tooltip.style.left=Math.min(r.left,window.innerWidth-390)+'px';tooltip.style.top=(r.bottom+7)+'px';tooltip.classList.add('visible');},500);});el.addEventListener('mouseleave',()=>{clearTimeout(tooltipTimer);tooltip.classList.remove('visible');});});
    setConfig(initial);setFormConfig(trainingForm,initialTraining);setFormConfig(predictForm,initialPrediction);if(initialPrediction.adductType)predictForm.elements.adductType.dataset.restoreValue=initialPrediction.adductType;setPredictEnabled(false); document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>vscode.postMessage({type:'pick',form:b.dataset.form||'data',field:b.dataset.pick,kind:b.dataset.kind}));
    document.querySelectorAll('[data-app]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-app]').forEach(x=>x.classList.toggle('active',x===button));const app=button.dataset.app;if(app==='reactionPatterns'){document.getElementById('reactionPatternPageMount').append(document.getElementById('reactionPatternApp'));onReactionPatternChange=()=>{};}if(app==='adduct'){document.getElementById('adductPageMount').append(document.getElementById('adductApp'));onAdductChange=()=>{};}document.getElementById('homePage').hidden=true;document.getElementById('modelsPage').hidden=true;document.getElementById('trainablePage').hidden=true;document.getElementById('jobPage').hidden=true;document.getElementById('environmentPage').hidden=true;document.getElementById('spectrumViewerApp').hidden=true;document.getElementById('smartsApp').hidden=app!=='smarts';document.getElementById('fragmentTreeApp').hidden=app!=='fragmentTree';document.getElementById('reactionViewerApp').hidden=app!=='reactionViewer';document.getElementById('reactionPatternAnalysisApp').hidden=app!=='reactionPatternAnalysis';document.getElementById('reactionPatternApp').hidden=app!=='reactionPatterns';document.getElementById('adductApp').hidden=app!=='adduct';form.hidden=app!=='data';trainingForm.hidden=app!=='training';predictForm.hidden=app!=='predict';document.getElementById('dataActions').hidden=app!=='data';document.getElementById('appSubtitle').textContent=app==='smarts'?'SMARTS Search':app==='fragmentTree'?'Fragment Tree Visualization':app==='reactionViewer'?'Reaction Viewer':app==='reactionPatternAnalysis'?'Reaction Pattern Analysis':app==='reactionPatterns'?'Reaction Pattern Set Editor':app==='adduct'?'Fragment Ion Adduct Rule Set Editor':app==='training'?'Fragment Tree Training':app==='predict'?'Predict Spectrum':'Fragment Tree Data Preparation';if(app==='reactionViewer')window.dispatchEvent(new Event('reactionViewer/open'));if(app==='reactionPatternAnalysis')window.dispatchEvent(new Event('reactionPatternAnalysis/open'));});
    document.getElementById('reactionViewerLoadPatterns').onclick=()=>vscode.postMessage({type:'loadReactionPatternSet'});
    document.getElementById('save').onclick=()=>vscode.postMessage({type:'saveConfig',config:getConfig()}); document.getElementById('load').onclick=()=>vscode.postMessage({type:'loadConfig'}); document.getElementById('openResult').onclick=()=>vscode.postMessage({type:'openResult'});
    ${reactionPatternUi.script()}
    ${adductUi.script()}
    const initialAdductSet=initial.fragmenterParams?.fragment_ion_tree_builder?.fragment_ion_adduct_rule_set;if(initialAdductSet)window.dispatchEvent(new MessageEvent('message',{data:{type:'adductRuleSet',value:{fragment_ion_adduct_rule_set:JSON.parse(JSON.stringify(initialAdductSet))},path:''}}));
    form.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'run',config:getConfig()});}; document.getElementById('copyCommand').onclick=()=>vscode.postMessage({type:'copyCommand',config:getConfig()}); stop.onclick=()=>vscode.postMessage({type:'stop'});
    trainingForm.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'runTraining',config:getTrainingConfig()})};document.getElementById('trainingCopyCommand').onclick=()=>vscode.postMessage({type:'copyTrainingCommand',config:getTrainingConfig()});trainingStop.onclick=()=>vscode.postMessage({type:'stop'});document.getElementById('saveTraining').onclick=()=>vscode.postMessage({type:'saveTrainingConfig',config:getTrainingConfig()});document.getElementById('loadTraining').onclick=()=>vscode.postMessage({type:'loadTrainingConfig'});
    let predictSequence=0,predictWaiters=new Map(),predictListSequence=0,predictListWaiters=new Map();
    function predictSpectrumRequest(payload){return new Promise((resolve,reject)=>{const requestId=++predictSequence;predictWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'predictSpectrum',requestId,payload})})}
    function listPredictAdductsRequest(payload){return new Promise((resolve,reject)=>{const requestId=++predictListSequence;predictListWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'listPredictAdducts',requestId,payload})})}
    function setPredictEnabled(enabled){predictForm.elements.smiles.disabled=!enabled;predictForm.elements.ce.disabled=!enabled;predictForm.elements.adductType.disabled=!enabled;document.getElementById('predictPreview').disabled=!enabled;document.getElementById('predictSubmit').disabled=!enabled;}
    function renderAdductOptions(adducts){const select=predictForm.elements.adductType,restore=select.dataset.restoreValue;select.innerHTML=adducts.length?adducts.map(a=>'<option value="'+htmlEscape(a)+'">'+htmlEscape(a)+'</option>').join(''):'<option value="">No adducts available</option>';if(restore&&adducts.includes(restore))select.value=restore;delete select.dataset.restoreValue;}
    function resetPredictModelState(){delete predictForm.dataset.modelAppliedPath;document.getElementById('predictModelStatus').textContent='Model checkpoint changed — apply it again to refresh adduct types.';document.getElementById('predictModelStatus').className='status idle';renderAdductOptions([]);predictForm.elements.adductType.innerHTML='<option value="">Apply a model first…</option>';setPredictEnabled(false);window.dispatchEvent(new CustomEvent('prediction/model'));}
    document.getElementById('predictApplyModel').onclick=async()=>{const modelPath=predictForm.elements.modelPath.value.trim(),device=predictForm.elements.device.value,modelStatusEl=document.getElementById('predictModelStatus');if(!modelPath){modelStatusEl.textContent='Model checkpoint path is required.';modelStatusEl.className='status error';return}modelStatusEl.textContent='Loading model…';modelStatusEl.className='status running';setPredictEnabled(false);try{const result=await listPredictAdductsRequest({modelPath,device});renderAdductOptions(result.adducts);predictForm.dataset.modelAppliedPath=modelPath;modelStatusEl.textContent='Model applied. '+result.adducts.length+' main adduct type(s) available.';modelStatusEl.className='status completed';setPredictEnabled(true);window.dispatchEvent(new CustomEvent('prediction/model',{detail:{modelPath,adducts:result.adducts}}));}catch(error){delete predictForm.dataset.modelAppliedPath;modelStatusEl.textContent=String(error.message||error);modelStatusEl.className='status error';setPredictEnabled(false);window.dispatchEvent(new CustomEvent('prediction/model'));}};
    predictForm.elements.modelPath.addEventListener('input',resetPredictModelState);
    function spectrumSvg(peaks,width=900,height=260){if(!peaks.length)return '<p class="muted">No peaks were predicted.</p>';const maxMz=Math.max(...peaks.map(p=>p.mz))*1.05,maxIntensity=Math.max(...peaks.map(p=>p.intensity))||1,padL=55,padB=30,padT=12,padR=20,x=mz=>padL+(mz/maxMz)*(width-padL-padR),y=intensity=>height-padB-(intensity/maxIntensity)*(height-padT-padB);const bars=peaks.map(p=>'<line class="spectrum-bar" x1="'+x(p.mz)+'" x2="'+x(p.mz)+'" y1="'+y(0)+'" y2="'+y(p.intensity)+'"><title>m/z '+p.mz.toFixed(4)+' · intensity '+p.intensity.toPrecision(4)+(p.formula?(' · '+p.formula):'')+'</title></line>').join('');const ticks=[0,.25,.5,.75,1].map(f=>'<text x="'+x(maxMz*f)+'" y="'+(height-8)+'" text-anchor="middle" class="spectrum-tick">'+(maxMz*f).toFixed(0)+'</text>').join('');return '<svg class="spectrum-chart" viewBox="0 0 '+width+' '+height+'"><line x1="'+padL+'" y1="'+padT+'" x2="'+padL+'" y2="'+y(0)+'" class="spectrum-axis"/><line x1="'+padL+'" y1="'+y(0)+'" x2="'+(width-padR)+'" y2="'+y(0)+'" class="spectrum-axis"/>'+bars+ticks+'</svg>'}
    let lastPredictTree=null,lastPredictionResult=null,predictPeakSort={key:'mz',direction:1};
    const predictReactionButton=id=>id==null?'Seed reaction set':'<button type="button" data-predict-reaction="'+id+'">Reaction '+id+'</button>';
    function predictReactionSets(sets){const rows=sets||[];return '<details><summary>'+rows.length+' reaction set'+(rows.length===1?'':'s')+'</summary>'+(rows.length?rows.map(set=>'<p>{ '+(set.length?set.map(predictReactionButton).join(' '):'Source / empty set')+' }</p>').join(''):'<p>No materialized reaction history.</p>')+'</details>';}
    function peakTable(){return '<div class="prediction-table-scroll"><table id="predictPeakTable"><thead><tr><th>#</th><th><button type="button" class="peak-sort" data-predict-peak-sort="mz">m/z <span class="peak-sort-indicator"></span></button></th><th><button type="button" class="peak-sort" data-predict-peak-sort="intensity">Intensity <span class="peak-sort-indicator"></span></button></th><th>Precursor</th><th>Node ID</th><th>SMILES</th><th>Formula</th><th>Ion adduct</th><th><button type="button" class="peak-sort" data-predict-peak-sort="confidence">Confidence <span class="peak-sort-indicator"></span></button></th></tr></thead><tbody id="predictPeakRows"></tbody></table></div>';}
    function renderPredictionPeakRows(){const body=document.getElementById('predictPeakRows');if(!body||!lastPredictionResult)return;const rows=lastPredictionResult.peaks.map((peak,index)=>({peak,index})).sort((a,b)=>predictPeakSort.direction*(Number(a.peak[predictPeakSort.key]??0)-Number(b.peak[predictPeakSort.key]??0))||a.index-b.index);body.innerHTML=rows.map(({peak,index})=>'<tr><td>'+(index+1)+'</td><td>'+Number(peak.mz).toFixed(6)+'</td><td>'+Number(peak.intensity).toPrecision(5)+'</td><td>'+(peak.precursor?'◇ yes':'—')+'</td><td>'+(peak.nodeId??'—')+'</td><td><code>'+htmlEscape(peak.smiles||'')+'</code></td><td>'+htmlEscape(peak.formula||'')+'</td><td>'+htmlEscape(peak.adduct||'')+'</td><td>'+(Number.isFinite(Number(peak.confidence))?(Number(peak.confidence)*100).toFixed(1)+'%':'—')+'</td></tr>').join('');document.querySelectorAll('[data-predict-peak-sort]').forEach(button=>button.querySelector('.peak-sort-indicator').textContent=button.dataset.predictPeakSort===predictPeakSort.key?(predictPeakSort.direction>0?'▲':'▼'):'');}
    function predictTreeSvg(tree){if(!tree||!tree.nodes.length)return '<p class="muted">No fragment tree was produced.</p>';const detectedNodes=new Set(tree.nodes.filter(n=>n.annotations&&n.annotations.length).map(n=>n.id)),depth=new Map(tree.nodes.map(n=>[n.id,Math.max(0,n.depth)]));const levels={};for(const n of tree.nodes)(levels[depth.get(n.id)]||(levels[depth.get(n.id)]=[])).push(n);for(const nodes of Object.values(levels))nodes.sort((a,b)=>a.id-b.id);const maxDepth=Math.max(...depth.values(),1),width=Math.max(900,(maxDepth+1)*230),maxRows=Math.max(...Object.values(levels).map(x=>x.length)),height=Math.max(260,maxRows*62+80),pos=new Map();for(const [d,nodes] of Object.entries(levels))nodes.forEach((n,i)=>pos.set(n.id,{x:45+Number(d)/maxDepth*(width-90),y:45+(i+1)*(height-90)/(nodes.length+1)}));const lines=tree.edges.map(edge=>{const a=pos.get(edge.source),b=pos.get(edge.target);if(!a||!b)return '';return '<g data-predict-edge="'+edge.id+'"><line class="tree-edge" x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y+'"/><line class="edge-hit" x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y+'"/><title>Node '+edge.source+' → '+edge.target+' · '+(edge.transitions||[]).length+' transition(s)</title></g>';}).join('');const nodes=tree.nodes.map(n=>{const p=pos.get(n.id),shape=n.precursor?'<polygon points="0,-19 19,0 0,19 -19,0"/>':'<circle r="16"/>';return '<g class="tree-node '+(n.precursor?'precursor ':'')+(detectedNodes.has(n.id)?'detected':'')+'" data-predict-node="'+n.id+'" transform="translate('+p.x+' '+p.y+')">'+shape+'<text text-anchor="middle" dominant-baseline="central">'+n.id+'</text><title>Node '+n.id+' · '+htmlEscape(n.smiles)+(n.precursor?' · Precursor':'')+'</title></g>';}).join('');return '<div class="zoom-tools"><button type="button" data-predict-tree-zoom="0.8">−</button><button type="button" data-predict-tree-zoom="1.25">+</button><button type="button" data-predict-tree-zoom="reset">Reset</button><span>◇ Precursor · orange outline = predicted ion</span></div><div class="tree-scroll prediction-tree-scroll"><svg class="fragment-tree prediction-tree" viewBox="0 0 '+width+' '+height+'" data-base-width="'+width+'" data-base-height="'+height+'" style="width:'+width+'px;height:'+height+'px">'+lines+nodes+'</svg></div>';}
    document.getElementById('predictPreview').onclick=async()=>{const smiles=predictForm.elements.smiles.value.trim(),adductType=predictForm.elements.adductType.value.trim(),box=document.getElementById('predictMoleculePreview');if(!smiles){box.hidden=true;return}try{const result=await chemistry('depict',{smiles,adducts:adductType?[adductType]:[],usedAdduct:adductType});box.hidden=false;const row=result.adducts[0];box.innerHTML='<div>'+result.svg+'</div><dl><dt>Formula</dt><dd>'+htmlEscape(result.formula)+'</dd><dt>Exact mass</dt><dd>'+result.exactMass.toFixed(6)+'</dd>'+(row?'<dt>Precursor m/z ('+htmlEscape(row.adduct)+')</dt><dd>'+(row.error?('<span class="warning">'+htmlEscape(row.error)+'</span>'):row.mz.toFixed(6))+'</dd>':'')+'</dl>'}catch(e){box.hidden=false;box.innerHTML='<p class="warning">'+htmlEscape(e.message||String(e))+'</p>'}};
    function renderPredictionResult(result){const resultSection=document.getElementById('predictResultSection'),resultBox=document.getElementById('predictResult'),tree=result.tree||{nodes:[],edges:[]},reactions=result.reactions||[];document.getElementById('predictStatus').textContent='Predicted '+result.peaks.length+' peaks.';document.getElementById('predictResultEmpty').hidden=true;resultSection.hidden=false;lastPredictTree=tree;lastPredictionResult=result;const reactionRows=reactions.map(reaction=>'<tr id="predict-reaction-'+reaction.id+'"><td>'+predictReactionButton(reaction.id)+'</td><td>'+reaction.reactionPatternId+' · '+htmlEscape(reaction.reactionPatternName||'')+'</td><td>'+reaction.reactionRuleId+' · '+htmlEscape(reaction.reactionRuleName||'')+'</td><td><code>'+htmlEscape(reaction.reactantSmarts||'')+'</code></td><td>'+reaction.productMoleculeId+'</td><td>'+htmlEscape((reaction.sourceAtomMaps||[]).join(', '))+'</td><td>'+htmlEscape((reaction.retainedAtomMaps||[]).join(', '))+' / '+htmlEscape((reaction.discardedAtomMaps||[]).join(', '))+'</td><td>'+htmlEscape(JSON.stringify(reaction.cutBondMaps||[]))+' / '+htmlEscape(JSON.stringify(reaction.changedBondMaps||[]))+'</td></tr>').join('');const nodeRows=tree.nodes.map(node=>'<tr><td><button type="button" data-predict-node="'+node.id+'">'+node.id+'</button></td><td><code>'+htmlEscape(node.smiles)+'</code></td><td>'+(node.precursor?'◇ yes':'—')+'</td><td>'+(node.annotations?.length||0)+'</td><td>'+predictReactionSets(node.reactionSets||[])+'</td></tr>').join('');resultBox.innerHTML='<div class="prediction-result-cards"><article><strong>'+result.peaks.length+'</strong><span>Predicted peaks</span></article><article><strong>'+tree.nodes.length+'</strong><span>Fragment nodes</span></article><article><strong>'+tree.edges.length+'</strong><span>Fragment edges</span></article><article><strong>'+reactions.length+'</strong><span>Primitive reactions</span></article></div><div class="prediction-source-spectrum"><section><h3>Source Molecule</h3><div class="prediction-source-molecule">'+result.svg+'</div><code>'+htmlEscape(result.canonicalSmiles||'')+'</code><dl><dt>Formula</dt><dd>'+htmlEscape(result.formula)+'</dd><dt>Exact mass</dt><dd>'+Number(result.exactMass).toFixed(6)+'</dd><dt>Adduct</dt><dd>'+htmlEscape(result.adduct)+'</dd><dt>Precursor m/z</dt><dd>'+Number(result.precursorMz).toFixed(6)+'</dd></dl></section><section><h3>Predicted Spectrum</h3>'+spectrumSvg(result.peaks,900,300)+'</section></div><section><h3>Predicted Peaks</h3><p class="muted">One row per retained physical-ion candidate. Click m/z, Intensity, or Confidence to sort.</p>'+peakTable()+'</section><section><h3>Fragment Tree</h3><p class="muted">Click a node to inspect its molecule and ion candidates. Click an edge to inspect every reaction transition that materialized it.</p>'+predictTreeSvg(tree)+'<div id="predictEdgeDetail" role="status"></div><div id="predictFragmentDetail" hidden></div></section><section><h3>Fragment Nodes</h3><p class="muted">Reaction sets are histories from the original Source molecule.</p><div class="prediction-table-scroll"><table><thead><tr><th>Node ID</th><th>SMILES</th><th>Precursor</th><th>Predicted ions</th><th>Reaction sets</th></tr></thead><tbody>'+nodeRows+'</tbody></table></div></section><section><h3>Reaction Reference · Original Source atom maps</h3><div class="prediction-reaction-layout"><div><div class="zoom-tools"><button type="button" data-predict-source-zoom="0.8">−</button><button type="button" data-predict-source-zoom="1.25">+</button><button type="button" data-predict-source-zoom="reset">Reset</button></div><div class="prediction-source-map">'+(result.sourceSvg||result.svg)+'</div></div><div class="prediction-table-scroll prediction-reaction-reference"><table><thead><tr><th>Reaction</th><th>Reaction pattern</th><th>Reaction</th><th>Reactant SMARTS</th><th>Product</th><th>Source maps</th><th>Retained / discarded</th><th>Cut / changed bonds</th></tr></thead><tbody>'+reactionRows+'</tbody></table></div></div><details><summary>Reaction pattern definitions (SMARTS)</summary><pre>'+htmlEscape(JSON.stringify(result.reactionPatternSet||{},null,2))+'</pre></details></section>';renderPredictionPeakRows();resultSection.scrollIntoView({behavior:'smooth',block:'start'});}
    window.addEventListener('message',event=>{if(event.data.type==='workbench/predictionResult')renderPredictionResult(event.data.result);});
    document.getElementById('predictStop').onclick=()=>vscode.postMessage({type:'stopSpectrumPrediction'});
    predictForm.onsubmit=async e=>{e.preventDefault();const modelPath=predictForm.elements.modelPath.value.trim(),smiles=predictForm.elements.smiles.value.trim(),ce=predictForm.elements.ce.value.trim(),adductType=predictForm.elements.adductType.value.trim(),device=predictForm.elements.device.value,predictStatusEl=document.getElementById('predictStatus'),resultSection=document.getElementById('predictResultSection'),resultBox=document.getElementById('predictResult'),submit=document.getElementById('predictSubmit'),cancel=document.getElementById('predictStop');if(!modelPath||!smiles||!ce||!adductType){predictStatusEl.textContent='Model checkpoint, SMILES, collision energy, and adduct type are required.';predictStatusEl.className='status error';return}predictStatusEl.textContent='Predicting… this can take a while for large models.';predictStatusEl.className='status running';submit.disabled=true;cancel.disabled=false;resultSection.hidden=true;document.getElementById('predictResultEmpty').hidden=false;lastPredictTree=null;lastPredictionResult=null;try{const result=await predictSpectrumRequest({modelPath,smiles,ce,adductType,device});predictStatusEl.textContent='Predicted '+result.peaks.length+' peaks.';predictStatusEl.className='status completed';renderPredictionResult(result)}catch(error){predictStatusEl.textContent=String(error.message||error);predictStatusEl.className='status error';resultSection.hidden=true;document.getElementById('predictResultEmpty').hidden=false;}finally{cancel.disabled=true;submit.disabled=false;}};
    document.getElementById('predictResult').addEventListener('click',async e=>{const sort=e.target.closest('[data-predict-peak-sort]');if(sort){const key=sort.dataset.predictPeakSort;predictPeakSort=predictPeakSort.key===key?{key,direction:-predictPeakSort.direction}:{key,direction:1};renderPredictionPeakRows();return;}const zoom=e.target.closest('[data-predict-tree-zoom],[data-predict-source-zoom]');if(zoom){const svg=zoom.hasAttribute('data-predict-tree-zoom')?document.querySelector('#predictResult .prediction-tree'):document.querySelector('#predictResult .prediction-source-map svg');if(!svg)return;const factor=zoom.dataset.predictTreeZoom||zoom.dataset.predictSourceZoom;if(!svg.dataset.baseWidth){const view=svg.viewBox.baseVal;svg.dataset.baseWidth=view.width;svg.dataset.baseHeight=view.height;}const scale=factor==='reset'?1:Math.min(8,Math.max(.2,Number(svg.dataset.scale||1)*Number(factor)));svg.dataset.scale=scale;svg.style.width=Number(svg.dataset.baseWidth)*scale+'px';svg.style.height=Number(svg.dataset.baseHeight)*scale+'px';return;}const ref=e.target.closest('[data-predict-reaction]');if(ref){document.querySelectorAll('#predictResult .selected-reaction').forEach(row=>row.classList.remove('selected-reaction'));const row=document.getElementById('predict-reaction-'+ref.dataset.predictReaction);if(row){row.classList.add('selected-reaction');row.scrollIntoView({block:'center',behavior:'smooth'});}return;}const edgeEl=e.target.closest('[data-predict-edge]');if(edgeEl&&lastPredictTree){document.querySelectorAll('#predictResult .tree-edge.selected').forEach(line=>line.classList.remove('selected'));edgeEl.querySelector('.tree-edge')?.classList.add('selected');const edge=lastPredictTree.edges.find(item=>item.id===Number(edgeEl.dataset.predictEdge)),detail=document.getElementById('predictEdgeDetail');if(!edge||!detail)return;detail.innerHTML='<h4>Edge '+edge.id+' · Node '+edge.source+' → '+edge.target+'</h4>'+(edge.transitions||[]).map(transition=>'<div class="prediction-transition"><p><strong>Added:</strong> '+predictReactionButton(transition.addedReaction)+'</p>'+predictReactionSets([transition.parentReactions||[]])+' <span aria-hidden="true">→</span> '+predictReactionSets([transition.targetReactions||[]])+'</div>').join('');return;}const nodeEl=e.target.closest('[data-predict-node]');if(!nodeEl||!lastPredictTree)return;const nodeId=Number(nodeEl.dataset.predictNode),node=lastPredictTree.nodes.find(n=>n.id===nodeId);if(!node)return;const detail=document.getElementById('predictFragmentDetail');detail.hidden=false;detail.innerHTML='<p class="muted">Loading fragment node '+nodeId+'…</p>';detail.scrollIntoView({behavior:'smooth',block:'nearest'});try{const adducts=(node.annotations||[]).map(a=>a.adduct);const depicted=await chemistry('depict',{smiles:node.smiles,adducts,usedAdduct:adducts[0]||''});const rows=(node.annotations||[]).map(a=>'<tr><td>'+htmlEscape(a.adduct)+'</td><td>'+htmlEscape(a.ionFormula||'')+'</td><td>'+Number(a.peakMz).toFixed(6)+'</td><td>'+Number(a.peakIntensity).toPrecision(4)+'</td><td>'+(Number(a.probability)*100).toFixed(1)+'%</td></tr>').join('');detail.innerHTML='<h4>Fragment node '+nodeId+'</h4><div class="prediction-fragment-detail"><div class="prediction-fragment-molecule">'+depicted.svg+'</div><div><dl><dt>SMILES</dt><dd>'+htmlEscape(node.smiles)+'</dd><dt>Formula</dt><dd>'+htmlEscape(node.formula)+'</dd><dt>Exact mass</dt><dd>'+Number(node.exactMass).toFixed(6)+'</dd></dl><h5>Reaction histories</h5>'+predictReactionSets(node.reactionSets||[])+(rows?('<div class="prediction-table-scroll"><table><thead><tr><th>Adduct</th><th>Ion formula</th><th>m/z</th><th>Intensity</th><th>Probability</th></tr></thead><tbody>'+rows+'</tbody></table></div>'):'<p class="muted">This fragment did not directly back a predicted peak.</p>')+'</div></div>';}catch(error){detail.innerHTML='<p class="warning">'+htmlEscape(error.message||String(error))+'</p>';}});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='picked'){const target=m.form==='training'?trainingForm:m.form==='predict'?predictForm:form;if(target.elements[m.field])applyPickedPath(target.elements[m.field],m.value);}if(m.type==='config')setConfig(m.config);if(m.type==='trainingConfig')setFormConfig(trainingForm,m.config);if(m.type==='predictSpectrumResult'){const waiter=predictWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);predictWaiters.delete(m.requestId)}}if(m.type==='predictSpectrumError'){const waiter=predictWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));predictWaiters.delete(m.requestId)}}if(m.type==='listPredictAdductsResult'){const waiter=predictListWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);predictListWaiters.delete(m.requestId)}}if(m.type==='listPredictAdductsError'){const waiter=predictListWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));predictListWaiters.delete(m.requestId)}}if(m.type==='status'){statusEl.textContent=m.text;statusEl.className='status '+m.status;stop.disabled=m.status!=='running';if(m.command)document.getElementById('command').textContent=m.command;}if(m.type==='trainingStatus'){trainingStatusEl.textContent=m.text;trainingStatusEl.className='status '+m.status;trainingStop.disabled=m.status!=='running';if(m.command)document.getElementById('trainingCommand').textContent=m.command;}});`;
}

function trainingCss() { return `.model-map{background:color-mix(in srgb,var(--vscode-editor-background) 94%,var(--accent))}.architecture{display:grid;gap:13px;margin-top:15px}.architecture-input,.architecture-merge,.architecture-branches{display:grid;gap:9px}.architecture-branches{grid-template-columns:1fr 1fr 1.35fr}.architecture-modes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mode{display:grid;grid-template-columns:1fr 1fr;gap:8px;border:1px solid var(--border);border-radius:8px;padding:10px}.mode strong{grid-column:1/-1;font-size:10px;letter-spacing:.14em}.training-mode strong{color:#e9b949}.prediction-mode strong{color:#63a8ff}.architecture button{position:relative;text-align:left;min-height:78px;background:var(--vscode-editor-background)}.architecture button b,.architecture button code,.architecture button span{display:block}.architecture button b{color:var(--accent);margin-bottom:5px}.architecture button code{font-size:11px;margin-bottom:5px;white-space:normal}.architecture button span{font-size:11px;opacity:.68}.architecture button.selected{outline:2px solid var(--accent);background:color-mix(in srgb,var(--accent) 14%,var(--vscode-editor-background))}.architecture-input button::after,.architecture-branches button::after,.architecture-merge button::after{content:'↓';position:absolute;left:50%;bottom:-22px;color:var(--accent);z-index:2;font-size:17px}.model-hint{margin:13px 0 0;font-size:12px;opacity:.75}[data-model-block]{scroll-margin-top:12px;transition:border-color .15s,box-shadow .15s}[data-model-block].selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.model-losses{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}.model-losses span:nth-child(odd){padding:8px 11px;border:1px solid var(--border);border-radius:18px;background:var(--vscode-editor-background)}.model-notes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0 16px}.model-notes p{margin:0;padding:10px;border-left:3px solid var(--accent);background:var(--vscode-editor-background);font-size:12px;line-height:1.45}.field-help{display:block;margin-top:5px;line-height:1.35;opacity:.7}.check .field-help{margin-top:2px}.check{align-items:flex-start}@media(max-width:850px){.architecture-branches,.architecture-modes,.mode,.model-notes{grid-template-columns:1fr}.mode strong{grid-column:1}.architecture button::after{display:none}}`; }


function deactivate() {}
module.exports = { ResultEditorProvider, activate, deactivate, buildArgs, normalizeConfig, buildTrainingArgs, normalizeTrainingConfig, resultHtml, workbenchHtml, ensureFileSuffix, reactionPatternSetSavePath, summarizeStructureManifests };
