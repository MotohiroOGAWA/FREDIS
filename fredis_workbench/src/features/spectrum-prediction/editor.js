const projects=require('../../workbench/project');
const jobs=require('../../workbench/panel');
const vscode = require('vscode');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { saveConfiguration, loadConfiguration, parseConfiguration } = require('../../configuration');

const defaults = {
  db: 'unspecified', smilesColumn: 'SMILES',
  precursorMzColumn: 'PrecursorMZ', adductTypeColumn: 'AdductType',
  collisionEnergyColumn: 'CollisionEnergy', instrumentColumn: '',
  specIdColumn: 'SpecID', overwrite: false, maxSamples:128, numWorkers:1, chunkSize:1, keepTemp:false, peakIntensityThreshold:0
};

function buildBatchArgs(config) {
  const threshold = Number(config.peakIntensityThreshold ?? 0);
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Peak intensity threshold must be between 0 and 1.');
  const args = ['-m', 'fredis.ml.specgen.predict_spectrum',
    '--input', config.input, '--output-dir', config.outputDir,
    '--model', config.modelPath,
    '--device', config.device || 'cpu', '--max-samples',String(config.maxSamples??128), '--db', config.db || 'unspecified',
    '--smiles-column', config.smilesColumn || 'SMILES',
    '--precursor-mz-column', config.precursorMzColumn || 'PrecursorMZ',
    '--adduct-type-column', config.adductTypeColumn || 'AdductType',
    '--collision-energy-column', config.collisionEnergyColumn || 'CollisionEnergy',
    '--spec-id-column', config.specIdColumn || 'SpecID',
    '--num-workers', String(config.numWorkers??1), '--chunk-size', String(config.chunkSize??1)];
  if (String(config.instrumentColumn || '').trim()) args.push('--instrument-column', config.instrumentColumn.trim());
  if (config.overwrite) args.push('--overwrite');
  if (config.keepTemp) args.push('--keep-temp');
  args.push('--peak-intensity-threshold', String(config.peakIntensityThreshold ?? 0));
  return args;
}

function shellDisplay(program, args) {
  return [program, ...args].map(value => /^[A-Za-z0-9_./:=,-]+$/.test(value)
    ? value : "'" + value.replace(/'/g, "'\\''") + "'").join(' ');
}

function attach(panel, context, projectRoot, output) {
  let child;
  const panelContext=context;
  panel.onDidDispose(() => { if (child) child.kill('SIGTERM'); });
  panel.webview.onDidReceiveMessage(async message => {
    const context=projects.capture(panelContext);
    let observed;
    if (!['predictBatchPick', 'predictBatchCopy', 'predictBatchRun', 'predictBatchStop', 'predictSaveConfig', 'predictLoadConfig', 'predictLoadConfigFile', 'predictLoadConfigJSON'].includes(message.type)) return;
    const post = data => panel.webview.postMessage({ type: 'predictBatchStatus', ...data });
    try {
      if (message.type === 'predictSaveConfig') {
        const saved = await saveConfiguration('spectrum-prediction', message.config, 'spectrum-prediction.json');
        if (saved) panel.webview.postMessage({ type: 'predictConfigSaved', path: saved });
        return;
      }
      if (message.type === 'predictLoadConfig') {
        const loaded = await loadConfiguration('spectrum-prediction');
        if (loaded) panel.webview.postMessage({ type: 'predictConfigLoaded', ...loaded });
        return;
      }
      if (message.type === 'predictLoadConfigFile' || message.type === 'predictLoadConfigJSON') {
        // predict_spectrum.py's main() writes prediction.pft into every
        // run's --output-dir with these same flat, camelCased field names,
        // so a past run's settings can be restored the same way a data
        // preparation or training .pft already can. Extension-agnostic: any
        // filename with this content works, so no filter is applied here.
        if(message.type==='predictLoadConfigJSON'&&(typeof message.json!=='string'||message.json.length>2000000))throw new Error('Prediction configuration is too large or invalid.');
        const picked = message.path?{fsPath:message.path}:message.type==='predictLoadConfigJSON'?{fsPath:message.name||'prediction.pft'}:(await vscode.window.showOpenDialog({ canSelectMany: false }))?.[0];
        if (!picked) return;
        const raw = JSON.parse(message.type==='predictLoadConfigJSON'?message.json:await require('fs').promises.readFile(picked.fsPath, 'utf8'));
        const loaded = raw?.schema ? parseConfiguration(raw,'spectrum-prediction') : raw;
        panel.webview.postMessage({ type: 'predictConfigLoaded', path: picked.fsPath, config: loaded });
        return;
      }
      if (message.type === 'predictBatchStop') { if (child) child.kill('SIGTERM'); return; }
      if (message.type === 'predictBatchPick') {
        const selected = message.field === 'outputDir'
          ? (await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false }))?.[0]
          : (await vscode.window.showOpenDialog({ filters: { 'MSDataset': ['msds'] }, canSelectMany: false }))?.[0];
        if (selected) panel.webview.postMessage({ type: 'predictBatchPicked', field: message.field, value: selected.fsPath });
        return;
      }
      const config = { ...defaults, ...message.config };
      for (const key of ['input', 'outputDir', 'modelPath']) if (!String(config[key] || '').trim()) throw new Error(`${key} is required.`);
      if(!Number.isInteger(Number(config.maxSamples))||Number(config.maxSamples)<1)throw new Error('Maximum simultaneous samples must be a positive integer.');
      const root = projectRoot(context);
      const resolvedOutput=path.resolve(root,config.outputDir);
      const existing=['predicted.msds','predicted.mssim'].filter(name=>fs.existsSync(path.join(resolvedOutput,name)));
      if(message.type==='predictBatchRun'&&existing.length&&!config.overwrite){const choice=await vscode.window.showWarningMessage(`Prediction output already exists in ${resolvedOutput}. Overwrite it?`,{modal:true},'Overwrite');if(choice!=='Overwrite'){post({running:false,text:'Prediction cancelled. Existing output was not changed.'});return;}config.overwrite=true;}
      const python = vscode.workspace.getConfiguration('fredis').get('pythonPath', 'python');
      const args = buildBatchArgs(config);
      const command = shellDisplay(python, args);
      if (message.type === 'predictBatchCopy') {
        await vscode.env.clipboard.writeText(command);
        post({ text: 'Batch prediction command copied.', command });
        return;
      }
      if (child) throw new Error('Batch prediction is already running.');
      output.show(true); output.appendLine('$ ' + command);
      post({ running: true, clear: true, text: 'Predicting MSDataset…', command });
      child = spawn(python, args, { cwd: root, env: { ...process.env, PYTHONUNBUFFERED: '1',
        PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) } });
      observed=jobs.observeJob(context,{type:'batch-prediction',workflow:'prediction',name:'Batch prediction',outputDir:config.outputDir,child,command:[python,...args],config,base:root});
      child.stdin.on('error', () => {}); child.stdin.end();
      const stream = data => {
        observed.log(data.toString());
        const clean = data.toString().replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
        const lines = clean.split(/[\r\n]+/).map(line => line.trim()).filter(Boolean);
        const isProgress = line => /\d+%.*\|/.test(line) || /^\[\d+\/\d+\]/.test(line);
        const progress = lines.filter(isProgress).at(-1);
        const staticLog = lines.filter(line => !isProgress(line)).join('\n');
        if (staticLog) output.appendLine(staticLog);
        post({ ...(staticLog ? { log: staticLog + '\n' } : {}), ...(progress ? { progress } : {}) });
      };
      child.stdout.on('data', stream); child.stderr.on('data', stream);
      const code = await new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', (exitCode, signal) => signal ? reject(new Error('Batch prediction cancelled.')) : resolve(exitCode));
      });
      observed.finish({code,partial:code===3});
      if (code !== 0 && code !== 3) throw new Error(`Prediction CLI failed (exit ${code}). See output for details.`);
      post({ running: false, text: code === 3 ? 'Prediction completed with some failed records.' : 'MSDataset prediction completed.' });
    } catch (error) {
      if(observed)observed.finish({error:error.message});
      post({ running: false, text: error.message });
    } finally {
      if (message.type === 'predictBatchRun') child = undefined;
    }
  });
}

function html() {
  const steps=[['model','Model'],['input','Input'],['conditions','Conditions'],['output','Output'],['run','Run']];
  const pathField=(name,label,kind,pick)=>`<label>${label}<div class="path"><input name="${name}" data-path-kind="${kind}"><button type="button" ${pick==='modelPath'?`data-pick="modelPath" data-kind="file" data-form="predict"`:`data-predict-batch-pick="${pick}"`}>Browse</button></div></label>`;
  const mapping=[['batchSmilesColumn','SMILES','SMILES'],['batchPrecursorMzColumn','Precursor m/z','PrecursorMZ'],['batchAdductTypeColumn','Adduct','AdductType'],['batchCollisionEnergyColumn','Collision energy','CollisionEnergy'],['batchSpecIdColumn','Spectrum ID','SpecID'],['batchInstrumentColumn','Instrument (optional)','']];
  return `<div id="predictConfiguration" class="preparation-heading"><div><h1>Spectrum Prediction</h1><p>Generate a spectrum for one molecule or predict an entire MSDataset with a trained fragment-tree model.</p></div><div class="prep-badges"><button type="button" id="predictLoadConfig" class="config-drop">Load Configuration<br><small>Drop prediction.pft or click to Browse</small></button><span id="predictEnvironment">Checking environment</span><span id="predictModeLabel">Single prediction</span></div></div>
  <nav class="prep-stepper" id="predictStepper" data-predict-mode="batch" hidden aria-label="Spectrum prediction steps">${steps.map(([key,label],i)=>`<button type="button" data-predict-step="${key}"><span class="step-circle">${i+1}</span><span data-predict-step-label="${key}">${label}</span></button>`).join('')}</nav>
  <div class="prep-step" data-predict-panel="model"><section><div class="prep-section-header"><h2>1. Prediction Model</h2><p>Select the trained fragment-tree checkpoint and inference device.</p></div>${pathField('modelPath','Model checkpoint *','file','modelPath')}<label>Device<select name="device"><option value="cpu">cpu</option><option value="cuda">cuda</option><option value="mps">mps</option></select></label><div class="actions"><button type="button" id="predictApplyModel" class="primary">Apply Model</button><button type="button" id="predictLoadConfigFile">Load From Run (.pft)</button><button type="button" id="predictSaveConfig">Save Configuration</button></div><div id="predictModelStatus" class="status idle">No model applied yet.</div><div class="prep-actions"><button type="button" data-next-predict="input" class="primary">Next: Input →</button></div></section></div>
  <div class="prep-step" data-predict-panel="input" hidden>
    <section id="predictSingleSection" data-predict-mode="single"><div class="prep-section-header"><h2>2. Molecule</h2><p>Enter the molecular structure to predict.</p></div><label>SMILES *<input name="smiles" placeholder="CC(=O)Oc1ccccc1C(=O)O"></label><div class="actions"><button type="button" id="predictPreview">Preview Molecule</button></div><div id="predictMoleculePreview" class="molecule-preview" hidden></div><div class="prep-actions"><button type="button" data-next-predict="conditions" class="primary">Next: Conditions →</button></div></section>
    <div id="predictBatchSection" data-predict-mode="batch" hidden><div class="prep-input-layout"><section><div class="prep-section-header"><h2>2. Input MSDataset</h2><p>Select and inspect the spectra whose molecular metadata will be predicted.</p></div><button type="button" id="predictDatasetDrop" class="dataset-drop">Drop an MSDataset here or click to Browse</button>${pathField('batchInput','Input MSDataset *','file','input')}<p class="muted">The source spectra are retained for similarity comparison; prediction uses their mapped structure and condition columns.</p><div id="predictDatasetSummary" class="dataset-summary-grid"></div><div class="actions"><button type="button" id="predictInspectDataset">Refresh Dataset</button><button type="button" id="predictValidateDataset">Validate Dataset Values</button></div><div class="dataset-validation-progress"><div><span>Records</span><progress id="predictDatasetValidationProgress" max="1" value="0"></progress><small id="predictDatasetValidationProgressText">Not checked</small></div></div><p id="predictDatasetStatus" class="validation-summary">Choose an MSDataset to inspect.</p></section><section><div class="preview-tools"><div><h2>Dataset Preview</h2><p id="predictDatasetPreviewStatus" class="muted">Choose an MSDataset to preview records.</p></div><input id="predictDatasetFilter" type="search" placeholder="Search SMILES or spectrum ID…"></div><div id="predictDatasetPreviewTable" class="dataset-preview-table"><p class="dataset-preview-empty">No dataset loaded.</p></div><div class="preview-detail-grid"><section><h3>Selected Molecule</h3><div id="predictDatasetStructure" class="record-structure"></div><div id="predictDatasetMetadata" class="record-metadata"></div></section><section><h3>Source Spectrum</h3><div id="predictDatasetSpectrum"></div></section></div></section></div><div class="prep-actions"><button type="button" data-next-predict="conditions" class="primary">Next: Columns →</button></div></div>
  </div>
  <div class="prep-step" data-predict-panel="conditions" hidden><section data-predict-mode="single"><div class="prep-section-header"><h2>3. Experimental Conditions</h2><p>Choose collision energy and one of the main adducts supported by the applied model.</p></div><div class="grid"><label>Collision energy (eV) *<input name="ce" type="number" min="0" step="any"></label><label>Adduct type *<select name="adductType"><option value="">Apply a model first…</option></select></label></div><div class="prep-actions"><button type="button" data-next-predict="output" class="primary">Next: Output →</button></div></section><section data-predict-mode="batch" hidden><div class="prep-section-header"><h2>3. Dataset Columns and Prediction Batch</h2><p>Map required metadata columns and bound simultaneous prediction work.</p></div><datalist id="predictDatasetColumns"></datalist>${mapping.map(([name,label,value])=>`<div class="mapping-row"><label>${label}</label><input name="${name}" value="${value}" list="predictDatasetColumns" ${name==='batchInstrumentColumn'?'placeholder="Not used"':''}><span id="predict-mapping-${name}" title="Not checked">—</span></div>`).join('')}<label>Maximum simultaneous samples<input name="batchMaxSamples" type="number" min="1" step="1" value="128"></label><div class="prep-actions"><button type="button" data-next-predict="output" class="primary">Next: Output →</button></div></section></div>
  <div class="prep-step" data-predict-panel="output" hidden><section data-predict-mode="single"><div class="prep-section-header"><h2>4. Predicted Spectrum</h2><p>The spectrum, peak table, fragment tree, and molecular details appear here after prediction.</p></div><p id="predictResultEmpty" class="dataset-preview-empty">Review the settings and run prediction.</p><div id="predictResultSection" hidden><div id="predictResult"></div></div><div class="prep-actions"><button type="button" data-next-predict="run" class="primary">Review and Run →</button></div></section><section data-predict-mode="batch" hidden><div class="prep-section-header"><h2>4. Output Settings</h2><p>Choose the output directory, result labeling, and parallel preparation settings.</p></div>${pathField('batchOutputDir','Output directory *','folder','outputDir')}<div class="grid"><label>DB label<input name="batchDb" value="unspecified"></label><label>Parallel prepare workers<input name="batchNumWorkers" type="number" min="1" step="1" value="1"></label><label>Compounds per worker chunk<input name="batchChunkSize" type="number" min="1" step="1" value="1"></label></div><label class="check"><input name="batchOverwrite" type="checkbox"><span>Overwrite existing output</span></label><label class="check"><input name="batchKeepTemp" type="checkbox"><span>Keep parallel preparation temporary files</span></label><p class="muted">The directory receives predicted.msds, predicted.mssim, prediction.pft, run information, and any retained temporary files.</p><div class="prep-actions"><button type="button" data-next-predict="run" class="primary">Review and Run →</button></div></section></div>
  <div class="prep-step" data-predict-panel="run" hidden><section><div class="prep-section-header"><h2>5. Review and Run</h2><p>Confirm the effective model, input, conditions, and output before starting.</p></div><dl id="predictRunSummary" class="run-summary"></dl><button type="button" id="predictPreflight">Preflight Check</button><p id="predictPreflightStatus" role="status"></p><div class="actions" data-predict-mode="batch" hidden><button type="button" id="predictOpenOutput">Reveal Existing Output</button></div><div id="predictBatchProgressRow" class="dataset-validation-progress" data-predict-mode="batch" hidden><div><span>Predict</span><progress id="predictBatchProgress" max="100"></progress><small id="predictBatchProgressText">Not started</small></div></div><pre id="predictBatchLog" data-predict-mode="batch" hidden></pre></section></div>
  <label>Peak intensity threshold<input name="peakIntensityThreshold" type="number" min="0" max="1" step="any" value="0"></label><p class="muted">Remove peaks at or below this relative intensity (maximum = 1). Default: 0.</p>
  <footer class="prep-bottom prediction-bottom"><div><div id="predictStatus" data-predict-mode="single" class="status idle">Ready</div><div id="predictBatchStatus" data-predict-mode="batch" class="status idle" hidden>Ready</div><p id="predictRunReasons" class="danger" role="status" tabindex="-1"></p><code id="predictBatchCommand" data-predict-mode="batch" hidden></code></div><div class="actions"><button type="button" id="predictionSaveDefaults">Save as Default</button><button type="button" id="predictBatchCopy" data-predict-mode="batch" hidden>Copy CLI Command</button><button type="button" id="predictStop" data-predict-mode="single" disabled>Stop</button><button type="button" id="predictBatchStop" data-predict-mode="batch" hidden disabled>Stop</button><button type="submit" id="predictSubmit" data-predict-mode="single" class="primary">▶ Predict Spectrum</button><button type="button" id="predictBatchRun" data-predict-mode="batch" class="primary" hidden>▶ Predict MSDataset</button></div></footer>`;
}

function script() { return `(${client.toString()})(${JSON.stringify(defaults)});`; }
function client(defaults) {
  const el = id => document.getElementById(id), predictForm = el('predictForm');
  const stepOrder=['model','input','conditions','output','run'],userMapped=new Set();let predictionMode='single',currentByMode={single:'model',batch:'model'},dataset=null,datasetError='',datasetLoading=false,datasetChecked=false,datasetVersion=0,selectedRecord=0,mappingTimer,batchRunning=false,validationProgress={current:0,total:0};
  const value = name => predictForm.elements[name]?.value || '';
  const config = () => ({
    input: value('batchInput'), outputDir: value('batchOutputDir'), db: value('batchDb'),
    maxSamples:Number(value('batchMaxSamples')), specIdColumn: value('batchSpecIdColumn'),
    smilesColumn: value('batchSmilesColumn'), precursorMzColumn: value('batchPrecursorMzColumn'),
    adductTypeColumn: value('batchAdductTypeColumn'), collisionEnergyColumn: value('batchCollisionEnergyColumn'),
    instrumentColumn: value('batchInstrumentColumn'), overwrite: !!predictForm.elements.batchOverwrite?.checked,
    numWorkers:Number(value('batchNumWorkers')||1), chunkSize:Number(value('batchChunkSize')||1),
    keepTemp: !!predictForm.elements.batchKeepTemp?.checked,
    peakIntensityThreshold: Number(value('peakIntensityThreshold')),
    modelPath: value('modelPath'), device: value('device')
  });
  const fullConfig = () => ({ ...config(), predictionMode, smiles: value('smiles'), ce: value('ce'), adductType: value('adductType') });
  const send = type => vscode.postMessage({ type, config: config() });
  function mapping(){return {smilesColumn:value('batchSmilesColumn'),precursorMzColumn:value('batchPrecursorMzColumn'),adductTypeColumn:value('batchAdductTypeColumn'),collisionEnergyColumn:value('batchCollisionEnergyColumn')};}
  function showStep(step){if(predictionMode==='single'){for(const panel of predictForm.querySelectorAll('[data-predict-panel]'))panel.hidden=panel.dataset.predictPanel==='run';for(const button of predictForm.querySelectorAll('[data-predict-step]'))button.classList.remove('active');validate();return;}if(!stepOrder.includes(step))return;currentByMode.batch=step;for(const panel of predictForm.querySelectorAll('[data-predict-panel]'))panel.hidden=panel.dataset.predictPanel!==step;for(const button of predictForm.querySelectorAll('[data-predict-step]'))button.classList.toggle('active',button.dataset.predictStep===step);validate();}
  window.fredisPredictStep=showStep;
  for(const button of predictForm.querySelectorAll('[data-predict-step]'))button.onclick=()=>showStep(button.dataset.predictStep);
  for(const button of predictForm.querySelectorAll('[data-next-predict]'))button.onclick=()=>showStep(button.dataset.nextPredict);
  function reasons(){const items=[];if(value('peakIntensityThreshold')===''||!Number.isFinite(Number(value('peakIntensityThreshold')))||Number(value('peakIntensityThreshold'))<0||Number(value('peakIntensityThreshold'))>1)items.push('Peak intensity threshold must be between 0 and 1.');if(!value('modelPath').trim())items.push('Model checkpoint is required.');if(predictionMode==='single'){if(predictForm.dataset.modelAppliedPath!==value('modelPath').trim())items.push('Apply the selected model.');if(!value('smiles').trim())items.push('SMILES is required.');if(value('ce')===''||!Number.isFinite(Number(value('ce')))||Number(value('ce'))<0)items.push('Collision energy must be a non-negative number.');if(!value('adductType').trim())items.push('Adduct type is required.');}else{if(!value('batchInput').trim())items.push('Input MSDataset is required.');if(!value('batchOutputDir').trim())items.push('Output directory is required.');for(const name of ['batchSmilesColumn','batchPrecursorMzColumn','batchAdductTypeColumn','batchCollisionEnergyColumn','batchSpecIdColumn'])if(!value(name).trim())items.push(name.replace(/^batch/,'')+' is required.');for(const name of ['batchMaxSamples','batchNumWorkers','batchChunkSize'])if(!Number.isInteger(Number(value(name)))||Number(value(name))<1)items.push(name.replace(/^batch/,'')+' must be a positive integer.');if(datasetLoading)items.push('Dataset inspection is still running.');if(datasetError)items.push(datasetError);if(dataset&&missingColumns().length)items.push('Mapped columns are missing: '+missingColumns().join(', ')+'.');}return [...new Set(items)];}
  function missingColumns(){if(!dataset?.columns)return [];const required=['batchSmilesColumn','batchPrecursorMzColumn','batchAdductTypeColumn','batchCollisionEnergyColumn','batchSpecIdColumn'],optional=value('batchInstrumentColumn').trim()?['batchInstrumentColumn']:[];return [...required,...optional].filter(name=>!dataset.columns.includes(value(name))).map(name=>value(name));}
  function summary(){const target=el('predictRunSummary');target.replaceChildren();const values=predictionMode==='single'?[['Mode','Single molecule'],['Model',value('modelPath')||'Not selected'],['Device',value('device')],['SMILES',value('smiles')||'Not entered'],['Collision energy',value('ce')?value('ce')+' eV':'Not entered'],['Adduct',value('adductType')||'Not selected']]:[['Mode','Batch MSDataset'],['Model',value('modelPath')||'Not selected'],['Device',value('device')],['Input',value('batchInput')||'Not selected'],['Records',dataset?.records??'Not inspected'],['Output',value('batchOutputDir')||'Not selected'],['Batch size',value('batchMaxSamples')],['Workers / chunk',value('batchNumWorkers')+' / '+value('batchChunkSize')]];for(const [label,text]of values){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=String(text);target.append(dt,dd);}}
  function validationRows(){for(const name of ['batchSmilesColumn','batchPrecursorMzColumn','batchAdductTypeColumn','batchCollisionEnergyColumn','batchSpecIdColumn','batchInstrumentColumn']){const icon=el('predict-mapping-'+name),column=value(name).trim();if(!icon)continue;const optional=name==='batchInstrumentColumn'&&!column,present=optional||!!dataset?.columns?.includes(column);icon.textContent=optional?'—':present?'✓':'✕';icon.className=optional?'':present?'mapping-ok':'mapping-error';icon.title=optional?'Optional column not used':present?'Column exists':'Column not found or dataset not loaded';}}
  function validate(){const items=reasons();el('predictRunReasons').textContent=items.join(' ');el('predictSubmit').disabled=predictionMode==='single'&&!!items.length;el('predictBatchRun').disabled=batchRunning||(predictionMode==='batch'&&!!items.length);el('predictBatchCopy').disabled=predictionMode==='batch'&&!!items.length;summary();validationRows();for(const button of predictForm.querySelectorAll('[data-predict-step]')){const index=stepOrder.indexOf(button.dataset.predictStep),current=stepOrder.indexOf(currentByMode[predictionMode]);button.classList.toggle('complete',index<current);} }
  function inspect(checkValues=false){datasetChecked=false;datasetError='';dataset=null;validationProgress={current:0,total:0};const file=value('batchInput').trim(),requestId='prediction-'+(++datasetVersion);if(!file){datasetLoading=false;renderDataset();validate();return;}datasetLoading=true;el('predictDatasetStatus').textContent=checkValues?'Validating every record…':'Inspecting dataset…';if(checkValues)vscode.postMessage({type:'data/check',target:'prediction',requestId,path:file,mapping:mapping()});else{vscode.postMessage({type:'data/columns',target:'prediction',requestId,path:file,mapping:mapping()});vscode.postMessage({type:'data/preview',target:'prediction',requestId,path:file,mapping:mapping()});}renderValidationProgress();validate();}
  function renderValidationProgress(){const bar=el('predictDatasetValidationProgress'),text=el('predictDatasetValidationProgressText');if(datasetLoading&&validationProgress.total){bar.max=validationProgress.total;bar.value=Math.min(validationProgress.current,validationProgress.total);text.textContent=validationProgress.current.toLocaleString()+' / '+validationProgress.total.toLocaleString();}else if(datasetLoading){bar.removeAttribute('value');text.textContent='Inspecting…';}else if(datasetChecked){const total=dataset?.records||validationProgress.total||0;bar.max=Math.max(total,1);bar.value=total;text.textContent=total.toLocaleString()+' / '+total.toLocaleString()+' (complete)';}else{bar.max=1;bar.value=0;text.textContent='Not checked';}}
  function applyColumnAliases(){if(!dataset?.columns)return false;const aliases={batchSmilesColumn:['SMILES','CanonicalSMILES','smiles','Structure'],batchPrecursorMzColumn:['PrecursorMZ','PrecursorMz','precursor_mz','PEPMASS'],batchAdductTypeColumn:['AdductType','Adduct','PrecursorType'],batchCollisionEnergyColumn:['CollisionEnergy','CE','collision_energy'],batchSpecIdColumn:['SpecID','SpectrumID','ID']};let changed=false;for(const [name,choices]of Object.entries(aliases)){if(userMapped.has(name)||dataset.columns.includes(value(name)))continue;const found=choices.find(choice=>dataset.columns.some(column=>column.toLowerCase()===choice.toLowerCase()));if(found){predictForm.elements[name].value=dataset.columns.find(column=>column.toLowerCase()===found.toLowerCase());changed=true;}}return changed;}
  async function selectDatasetRecord(index){const row=dataset?.rows?.find(item=>item.index===index)||dataset?.rows?.[0];if(!row)return;selectedRecord=row.index;renderDataset();const target=el('predictDatasetStructure');target.textContent='Loading structure…';try{const result=await chemistry('depict',{smiles:row.smilesColumn});if(selectedRecord===row.index)target.innerHTML=result.svg;}catch(error){if(selectedRecord===row.index)target.textContent=error.message||String(error);}}
  function renderDataset(){const cards=el('predictDatasetSummary');cards.replaceChildren();if(dataset){for(const [label,text]of [['Records',dataset.records?.toLocaleString?.()??dataset.records],['Unique SMILES',dataset.summary?.uniqueSmiles?.toLocaleString?.()??'—'],['Average peaks',dataset.summary?.averagePeaks===undefined?'—':Number(dataset.summary.averagePeaks).toFixed(1)]]){const card=document.createElement('article'),span=document.createElement('span'),strong=document.createElement('strong');span.textContent=label;strong.textContent=String(text);card.append(span,strong);cards.append(card);}}const rows=(dataset?.rows||[]).filter(row=>{const query=el('predictDatasetFilter').value.trim().toLowerCase();return !query||row.id.toLowerCase().includes(query)||row.smilesColumn.toLowerCase().includes(query);});const table=el('predictDatasetPreviewTable');table.replaceChildren();if(!rows.length){const p=document.createElement('p');p.className='dataset-preview-empty';p.textContent=datasetLoading?'Loading preview…':'No preview records.';table.append(p);}else{const node=document.createElement('table'),head=document.createElement('tr');for(const title of ['#','Spectrum ID','SMILES','Adduct','CE','Peaks']){const th=document.createElement('th');th.textContent=title;head.append(th);}node.append(head);for(const row of rows){const tr=document.createElement('tr');tr.dataset.record=row.index;tr.classList.toggle('selected',row.index===selectedRecord);for(const text of [row.index+1,row.id,row.smilesColumn,row.adductTypeColumn,row.collisionEnergyColumn,row.numPeaks]){const td=document.createElement('td');td.textContent=text;tr.append(td);}tr.onclick=()=>selectDatasetRecord(row.index);node.append(tr);}table.append(node);}const row=dataset?.rows?.find(item=>item.index===selectedRecord)||dataset?.rows?.[0];if(row){el('predictDatasetPreviewStatus').textContent='Showing '+(dataset.rows?.length||0)+' of '+dataset.records+' records';el('predictDatasetSpectrum').innerHTML=spectrumSvg(row.peaks,620,240);el('predictDatasetMetadata').textContent=row.id+' · '+row.smilesColumn+' · '+row.adductTypeColumn+' · CE '+row.collisionEnergyColumn;}else{el('predictDatasetPreviewStatus').textContent=datasetLoading?'Loading dataset…':'Choose an MSDataset to preview records.';el('predictDatasetSpectrum').replaceChildren();el('predictDatasetStructure').replaceChildren();el('predictDatasetMetadata').textContent='';}const errors=dataset?.errors||[];el('predictDatasetStatus').textContent=datasetError||errors.join(' ')||(datasetChecked?'All records checked. '+(dataset.eligibleRecords??dataset.records)+' eligible for prediction.':dataset?'Dataset loaded. Validate values before a long batch run.':'Choose an MSDataset to inspect.');el('predictDatasetStatus').className=(datasetError||errors.length)?'validation-summary danger':datasetChecked?'validation-summary mapping-ok':'validation-summary';const list=el('predictDatasetColumns');list.replaceChildren();for(const column of dataset?.columns||[]){const option=document.createElement('option');option.value=column;list.append(option);}renderValidationProgress();validationRows();}
  el('predictBatchRun').onclick = () => {validate();if(reasons().length){el('predictRunReasons').focus();return;}send('predictBatchRun');};
  el('predictBatchCopy').onclick = () => send('predictBatchCopy');
  el('predictBatchStop').onclick = () => vscode.postMessage({ type: 'predictBatchStop' });
  el('predictSaveConfig').onclick = () => vscode.postMessage({ type: 'predictSaveConfig', config: fullConfig() });
  el('predictLoadConfig').onclick = () => vscode.postMessage({ type: 'predictLoadConfig' });
  el('predictLoadConfigFile').onclick = () => vscode.postMessage({ type: 'predictLoadConfigFile' });
  el('predictOpenOutput').onclick=()=>{if(value('batchOutputDir').trim())vscode.postMessage({type:'library/open',path:value('batchOutputDir').trim()});};
  el('predictPreflight').onclick=()=>{validate();const items=reasons();if(predictionMode==='batch'&&value('batchInput').trim()&&!datasetChecked&&!datasetLoading){el('predictPreflightStatus').textContent='Validating every dataset record…';inspect(true);return;}el('predictPreflightStatus').textContent=items.length?items.join(' '):'Model, input, conditions, and output are ready.';el('predictPreflightStatus').className=items.length?'danger':'mapping-ok';};
  el('predictInspectDataset').onclick=()=>inspect(false);el('predictValidateDataset').onclick=()=>inspect(true);el('predictDatasetFilter').oninput=renderDataset;
  const datasetDrop=el('predictDatasetDrop');datasetDrop.onclick=()=>vscode.postMessage({type:'predictBatchPick',field:'input'});installFileDropZone(datasetDrop,{kind:'file',onProgress:percent=>{datasetDrop.textContent='Receiving dataset: '+percent+'%';},onDrop:paths=>dispatchPathPicked({type:'predictBatchPicked',field:'input',value:paths[0]}),onError:message=>{datasetError=message;renderDataset();validate();},onFinally:()=>{datasetDrop.textContent='Drop an MSDataset here or click to Browse';}});
  const configDrop=el('predictLoadConfig');installFileDropZone(configDrop,{kind:'file',upload:false,onDrop:paths=>vscode.postMessage({type:'predictLoadConfigFile',path:paths[0]}),onFiles:async files=>{const file=files[0];if(file.size>2000000)throw new Error('Configuration exceeds 2 MB.');vscode.postMessage({type:'predictLoadConfigJSON',json:await file.text(),name:file.name});},onError:message=>{el('predictRunReasons').textContent=message;}});
  document.querySelectorAll('[data-predict-batch-pick]').forEach(button => button.onclick = () =>
    vscode.postMessage({ type: 'predictBatchPick', field: button.dataset.predictBatchPick }));
  predictForm.elements.batchInput.addEventListener('input',()=>{dataset=null;datasetError='';datasetChecked=false;datasetVersion++;renderDataset();validate();});predictForm.elements.batchInput.addEventListener('change',()=>inspect(false));
  for(const name of ['batchSmilesColumn','batchPrecursorMzColumn','batchAdductTypeColumn','batchCollisionEnergyColumn','batchSpecIdColumn','batchInstrumentColumn'])predictForm.elements[name].addEventListener('input',()=>{userMapped.add(name);datasetChecked=false;clearTimeout(mappingTimer);mappingTimer=setTimeout(()=>inspect(false),450);validate();});
  predictForm.addEventListener('input',validate);predictForm.addEventListener('change',validate);window.addEventListener('prediction/model',validate);
  window.addEventListener('prediction/mode',event=>{predictionMode=event.detail?.mode==='batch'?'batch':'single';predictForm.dataset.predictionMode=predictionMode;el('predictModeLabel').textContent=predictionMode==='batch'?'Batch prediction':'Single prediction';el('predictStepper').querySelector('[data-predict-step-label="conditions"]').textContent='Columns';el('predictStepper').querySelector('[data-predict-step-label="output"]').textContent='Output';showStep(currentByMode[predictionMode]);});
  window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'predictBatchPicked') {
      const field = message.field === 'input' ? 'batchInput' : 'batchOutputDir';
      applyPickedPath(predictForm.elements[field], message.value);
      if(field!=='batchInput')validate();
    }
    if (message.type === 'predictConfigLoaded') {
      const loaded = { ...defaults, ...(message.config || {}) };
      if (!loaded.outputDir && loaded.output) loaded.outputDir = loaded.output.replace(/[\\/][^\\/]+$/, '');
      const fields = { maxSamples:'batchMaxSamples',input: 'batchInput', outputDir: 'batchOutputDir', db: 'batchDb',
        specIdColumn: 'batchSpecIdColumn', smilesColumn: 'batchSmilesColumn', precursorMzColumn: 'batchPrecursorMzColumn',
        adductTypeColumn: 'batchAdductTypeColumn', collisionEnergyColumn: 'batchCollisionEnergyColumn', instrumentColumn: 'batchInstrumentColumn',
        numWorkers: 'batchNumWorkers', chunkSize: 'batchChunkSize' };
      for (const [key, name] of Object.entries(fields)) if (predictForm.elements[name]) predictForm.elements[name].value = loaded[key] ?? '';
      predictForm.elements.batchOverwrite.checked = !!loaded.overwrite;
      predictForm.elements.batchKeepTemp.checked = !!loaded.keepTemp;
      predictForm.elements.peakIntensityThreshold.value = loaded.peakIntensityThreshold ?? 0;
      for (const name of ['modelPath', 'device', 'smiles', 'ce']) if (predictForm.elements[name] && loaded[name] !== undefined) predictForm.elements[name].value = loaded[name];
      if (loaded.adductType) predictForm.elements.adductType.dataset.restoreValue = loaded.adductType;
      el('predictBatchStatus').textContent = 'Loaded configuration ' + message.path;
      if (loaded.modelPath) el('predictApplyModel').click();
      if (loaded.input) inspect(false);else validate();
    }
    if (message.type === 'predictConfigSaved') el('predictBatchStatus').textContent = 'Saved configuration ' + message.path;
    if (message.type !== 'predictBatchStatus') return;
    if (message.running !== undefined) {
      batchRunning=message.running;
      el('predictBatchRun').disabled = message.running;
      el('predictBatchStop').disabled = !message.running;
      el('predictBatchStatus').className='status '+(message.running?'running':/completed/i.test(message.text||'')?'completed':/cancelled|failed|error/i.test(message.text||'')?'error':'idle');
      if(message.running&&message.clear){el('predictBatchProgress').removeAttribute('value');el('predictBatchProgressText').textContent='Starting…';}
    }
    if (message.text) el('predictBatchStatus').textContent = message.text;
    if (message.progress) {el('predictBatchStatus').textContent = message.progress;const match=message.progress.match(/(\d+(?:\.\d+)?)%/);if(match){el('predictBatchProgress').value=Math.min(100,Number(match[1]));el('predictBatchProgressText').textContent=match[1]+'%';}}
    if (message.command) el('predictBatchCommand').textContent = message.command;
    if (message.clear) el('predictBatchLog').textContent = '';
    if (message.log) el('predictBatchLog').textContent = (el('predictBatchLog').textContent + message.log).slice(-30000);
    validate();
  });
  window.addEventListener('message',event=>{const m=event.data;if(m.type==='data/check-progress'&&m.target==='prediction'&&m.requestId==='prediction-'+datasetVersion){validationProgress={current:m.current,total:m.total};renderValidationProgress();}if(['data/columns','data/preview','data/check'].includes(m.type)&&m.target==='prediction'&&m.requestId==='prediction-'+datasetVersion){if(m.type!=='data/columns'||!dataset?.rows)dataset=m.data;else dataset={...m.data,...dataset,columns:m.data.columns,validation:m.data.validation};if(applyColumnAliases()){inspect(false);return;}if(m.type!=='data/columns')datasetLoading=false;datasetChecked=m.type==='data/check'&&m.data.valuesChecked===true;datasetError='';renderDataset();if(m.type!=='data/columns'&&dataset?.rows?.length)void selectDatasetRecord(selectedRecord);validate();if(datasetChecked){const items=reasons();el('predictPreflightStatus').textContent=items.length?items.join(' '):'Dataset values and prediction settings are valid.';el('predictPreflightStatus').className=items.length?'danger':'mapping-ok';}}if(m.type==='data/error'&&m.target==='prediction'&&m.requestId==='prediction-'+datasetVersion){datasetLoading=false;datasetError=m.error;renderDataset();validate();}if(m.type==='workbench/environment')el('predictEnvironment').textContent=m.error?'Environment unavailable':'● Environment ready';if(m.type==='project/restore'&&m.workflow==='prediction')queueMicrotask(()=>{if(value('batchInput'))inspect(false);validate();});});
  predictForm.dataset.predictionMode='single';showStep('model');renderDataset();validate();
}

module.exports = { attach, html, script, buildBatchArgs, shellDisplay, defaults };
