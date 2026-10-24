const fs = require('fs');
const path = require('path');
const readline = require('readline');
const msentityViewer = require('./msentity-viewer');
const {createEpochAxis}=require('./epoch-axis');
const TRAINING_CONFIG_NAME = 'fragment-tree' + require('../../file-formats').format('training-config').suffix;

function safeResolve(root, relative) {
  if (!relative) return null;
  const target = path.resolve(root, relative), prefix = root.endsWith(path.sep) ? root : root + path.sep;
  return target === root || target.startsWith(prefix) ? target : null;
}

async function validationResult(root, manifest = {}, requestedEpoch = null) {
  const directory=safeResolve(root,manifest.artifacts?.validationDirectory||'spectrum_validation');
  if(!directory)throw new Error('Invalid validation directory.');
  let files=[];
  try {files=await fs.promises.readdir(directory);}catch(error){if(error.code!=='ENOENT')throw error;}
  const labels=[...new Set(files.map(file=>file.match(/^((?:epoch|step)_\d+)\.(?:json|summary\.json)$/)?.[1]).filter(Boolean))];
  let epochBoundaries=[];
  if(labels.some(label=>label.startsWith('step_'))) {
    const index=await metricIndex(root,manifest);
    if(index)epochBoundaries=await readEpochBoundaries(root,index);
    else {
      const filename=path.join(root,'metrics.tsv');let headers;
      try {for await(const line of completeMetricLines(filename)){
        const cells=line.split('\t');if(!headers){headers=cells;continue;}if(cells.length!==headers.length)continue;
        const row=Object.fromEntries(headers.map((key,i)=>[key,cells[i]]));
        if((row.event==='epoch_end'||!row.event)&&row.epoch?.trim()&&row.global_step?.trim())epochBoundaries.push({step:Number(row.global_step),epoch:Number(row.epoch)});
      }}catch(error){if(error.code!=='ENOENT')throw error;}
    }
  }
  const axis=createEpochAxis(epochBoundaries);
  const axisMode=labels.some(label=>label.startsWith('step_'))?axis.mode:'epoch';
  const checkpoints=labels.map(key=>{
    const [kind,number]=key.split('_'),step=kind==='step'?Number(number):null;
    return {key,epoch:kind==='epoch'?Number(number):axis.at(step),step,estimated:step!==null&&axis.estimated(step)};
  }).filter(item=>item.epoch!==null).sort((a,b)=>a.epoch-b.epoch||Number(b.key.startsWith('epoch_'))-Number(a.key.startsWith('epoch_')));
  let selected;
  if(requestedEpoch!==null){
    selected=checkpoints.find(item=>typeof requestedEpoch==='string'?item.key===requestedEpoch:Math.abs(item.epoch-requestedEpoch)<1e-9);
    if(!selected)throw new Error('No validation artifact for this epoch.');
  }else{
    const latest=path.basename(manifest.latestValidation||'').replace(/\.(json|msds)$/,'');
    selected=checkpoints.find(item=>item.key===latest)||checkpoints.at(-1);
  }
  if(!selected)return null;
  const target=path.join(directory,selected.key+'.json');
  const value=await require('./validation-reader').readValidation(target);
  return {file:path.relative(root,target),epoch:selected.epoch,key:selected.key,estimated:selected.estimated,
    epochs:[...new Set(checkpoints.map(item=>item.epoch))],checkpoints,epochBoundaries,axisMode,
    msdsFile:fs.existsSync(target.replace(/\.json$/,'.msds'))?path.relative(root,target.replace(/\.json$/,'.msds')):null,
    summary:value.summary||{},representatives:value.representatives||{}};
}

function representativePoints(representatives) {
  const points = new Map();
  for (const [metric, levels] of Object.entries(representatives || {})) {
    for (const [quantile,item] of Object.entries(levels)) {
      if (!item?.spectrum) continue;
      const record=item.spectrum;
      const key=String(item.source_spectrum_index ?? item.spectrum_index ?? JSON.stringify([record.piece_index,record.sample_index]));
      if (!points.has(key)) points.set(key,{key,record:{...record},selections:[]});
      const point=points.get(key);point.selections.push(metric+' · '+(quantile==='median'?'q50':quantile));
      if (point.record[metric]==null && item.value!=null) point.record[metric]=item.value;
    }
  }
  return [...points.values()];
}

function addLegacyLossSeries(series, intensityWeight) {
  const add = (name, points) => {
    if (!points.size) return;
    const target = series.get(name) || new Map();
    for (const [step,value] of points) if (!target.has(step)) target.set(step,value);
    series.set(name, target);
  };
  const combine = (a, b, transform) => new Map([...a].filter(([step]) => b.has(step)).map(([step,value]) => [step,transform(value,b.get(step))]));
  for (const split of ['train','train_window','validation','val']) {
    const prefix = 'metrics/' + split + '_';
    const total = series.get(prefix + 'intensity_loss');
    if (total) add(prefix + 'spectrum_loss', total);
    for (const suffix of ['with_precursor','without_precursor']) {
      const ce = series.get(prefix + (suffix === 'with_precursor' ? 'transition_cross_entropy' : 'transition_cross_entropy_without_precursor'));
      const loss = series.get(prefix + 'spectrum_loss_' + suffix);
      if (!ce || !loss || !Number.isFinite(intensityWeight)) continue;
      const weighted = new Map([...ce].map(([step,value]) => [step, value*intensityWeight]));
      add(prefix + 'cross_entropy_loss_' + suffix, weighted);
      add(prefix + 'cosine_loss_' + suffix, combine(loss, weighted, (a,b) => a-b));
    }
    for (const metric of ['cross_entropy_loss','cosine_loss']) {
      const full = series.get(prefix + metric + '_with_precursor'), excluded = series.get(prefix + metric + '_without_precursor');
      if (full && excluded) add(prefix + metric, combine(full, excluded, (a,b) => a+b));
    }
  }
}

async function metricIndex(directory, report) {
  const filename = safeResolve(directory, report?.artifacts?.metricIndex || 'metrics/index.json');
  if (!filename) throw new Error('Invalid metric index path.');
  let index;
  try { index = JSON.parse(await fs.promises.readFile(filename, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
  if (index.schema !== 'fredis.metric-index' || !Array.isArray(index.series)) throw new Error('Unsupported metric index.');
  // A legacy process can still be appending to the original logs after migration.
  // Fall back to those live logs instead of displaying an outdated snapshot.
  for (const source of index.legacySources || []) {
    const target = safeResolve(directory, source.file);
    if (!target) throw new Error('Invalid legacy metric path.');
    try {
      const stat = await fs.promises.stat(target);
      if (stat.size !== source.size || Math.abs(stat.mtimeMs - source.mtimeMs) > 1) return null;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  for (const entry of index.series) {
    if (typeof entry.name !== 'string' || typeof entry.file !== 'string' || !safeResolve(directory, entry.file)) throw new Error('Invalid metric catalog entry.');
  }
  return index;
}

async function indexedRun(directory, report) {
  const index = await metricIndex(directory, report);
  if (!index) return null;
  let maxDepth = null;
  try {
    const target = safeResolve(directory, report?.artifacts?.trainingConfiguration || TRAINING_CONFIG_NAME);
    const config = JSON.parse(await fs.promises.readFile(target, 'utf8'));
    const value = config.modelConfig?.fragmenter_params?.fragment_ion_tree_builder?.max_reaction_count;
    if (Number.isInteger(value) && value >= 0) maxDepth = value;
  } catch {}
  return {directory,report,maxDepth,lazy:true,files:['metrics/index.json'],
    validation:report ? await validationResult(directory, report) : null,
    series:index.series.map(entry=>({name:entry.name,points:[]}))};
}

async function* completeMetricLines(filename) {
  const input = fs.createReadStream(filename, {encoding:'utf8'});
  let pending = '';
  try {
    for await (const chunk of input) {
      pending += chunk;
      let end;
      while ((end = pending.indexOf('\n')) >= 0) {
        yield pending.slice(0,end).replace(/\r$/, '');
        pending = pending.slice(end+1);
      }
    }
    // An unfinished final row is left for the next refresh.
  } finally {input.destroy();}
}

async function readEpochBoundaries(directory,index,cache) {
  const entry=['metrics/epoch_seconds','metrics/validation_loss','metrics/val_loss','metrics/train_loss'].map(name=>index.series.find(item=>item.name===name)).find(Boolean);
  if(!entry)return [];
  if(cache?.has(entry.file))return cache.get(entry.file);
  const boundaries=new Map();let headers;
  for await(const line of completeMetricLines(safeResolve(directory,entry.file))) {
    const cells=line.split('\t');if(!headers){headers=cells;continue;}if(cells.length!==headers.length)continue;
    const row=Object.fromEntries(headers.map((name,i)=>[name,cells[i]]));
    if(row.event!=='epoch_end'||!row.epoch?.trim()||!row.global_step?.trim())continue;
    const epoch=Number(row.epoch),step=Number(row.global_step);
    if(Number.isFinite(epoch)&&Number.isFinite(step))boundaries.set(step,{step,epoch});
  }
  return [...boundaries.values()].sort((a,b)=>a.step-b.step);
}

async function readCharts(directory, names, report) {
  if (!Array.isArray(names) || names.some(name=>typeof name !== 'string')) throw new Error('Invalid chart selection.');
  const index = await metricIndex(directory, report);
  if (!index) throw new Error('Metrics changed. Refresh the training report to reload the catalog.');
  const catalog = groupSeries(index.series.map(entry=>({name:entry.name,points:[]})));
  const requested = new Set(names);
  const sources = new Set(catalog.filter(chart=>requested.has(chart.name)).flatMap(chart=>chart.sources));
  const requestedSources=new Set(sources);
  for(const name of requestedSources)if(/^metrics\/(train|validation|val|intermediate_validation)_batches$/.test(name))sources.add(name.replace(/_batches$/,'_total_batches'));
  const entries = index.series.filter(entry=>sources.has(entry.name));
  const files = [...new Set(entries.map(entry=>entry.file))];
  const series = new Map(entries.map(entry=>[entry.name,new Map()]));
  const boundaryCache=new Map();
  // Bound open streams while reading only files belonging to selected charts.
  for (const file of files) {
    const boundaries=[];boundaryCache.set(file,boundaries);
    let headers;
    for await (const line of completeMetricLines(safeResolve(directory,file))) {
      const cells = line.split('\t');
      if (!headers) {headers=cells;continue;}
      if (cells.length !== headers.length) continue;
      const row = Object.fromEntries(headers.map((key,i)=>[key,cells[i]]));
      if(row.event==='epoch_end'&&row.epoch?.trim()&&row.global_step?.trim()&&Number.isFinite(Number(row.epoch))&&Number.isFinite(Number(row.global_step)))boundaries.push({step:Number(row.global_step),epoch:Number(row.epoch)});
      if (!series.has(row.series) || !row.value?.trim() || !row.global_step?.trim()) continue;
      const step=Number(row.global_step),value=Number(row.value);
      if (Number.isFinite(step) && Number.isFinite(value)) series.get(row.series).set(step,value);
    }
  }
  for(const name of requestedSources)if(name.endsWith('_batches')&&!name.endsWith('_total_batches')){
    const total=series.get(name.replace(/_batches$/,'_total_batches'));if(total?.size)series.set(name,total);
  }
  const epochBoundaries=await readEpochBoundaries(directory,index,boundaryCache);
  return {directory,files,epochBoundaries,series:[...series].filter(([name])=>requestedSources.has(name)).map(([name,points])=>({name,points:[...points].sort((a,b)=>a[0]-b[0])}))};
}

async function readProgressMetrics(directory) {
  const index = await metricIndex(directory);
  if (!index) return null;
  const entries = index.series.filter(entry=>/^metrics\/(train_window|train|validation|val|intermediate_validation)_loss$/.test(entry.name));
  const wanted = new Set(entries.map(entry=>entry.name)), events = new Map();
  for (const filename of new Set(entries.map(entry=>entry.file))) {
    let headers;
    for await (const line of completeMetricLines(safeResolve(directory,filename))) {
      const cells=line.split('\t');
      if (!headers) {headers=cells;continue;}
      if (cells.length !== headers.length) continue;
      const row=Object.fromEntries(headers.map((key,i)=>[key,cells[i]]));
      if (!wanted.has(row.series) || !row.value?.trim()) continue;
      const value=Number(row.value),step=Number(row.global_step),epoch=Number(row.epoch);
      if (![value,step,epoch].every(Number.isFinite)) continue;
      const key=JSON.stringify([row.event,step,epoch]);
      if (!events.has(key)) events.set(key,{event:row.event,global_step:step,epoch});
      const name=row.series.replace(/^metrics\//,'').replace(/^train_window_/,'train_').replace(/^intermediate_validation_/,'validation_');
      events.get(key)[name]=value;
    }
  }
  const rows=[...events.values()].sort((a,b)=>a.global_step-b.global_step || a.epoch-b.epoch);
  return {rows:rows.filter(row=>row.event!=='intermediate_validation'),intermediateRows:rows.filter(row=>row.event==='intermediate_validation')};
}

async function readRun(directory) {
  let report;
  // Extension-agnostic: whether this path is a training-report file (any
  // name) or a run directory is decided by what it actually is on disk, not
  // by its suffix.
  let stat;
  try { stat = await fs.promises.stat(directory); } catch { stat = null; }
  if (stat?.isFile()) {
    let parsed;
    try { parsed = JSON.parse(await fs.promises.readFile(directory, 'utf8')); }
    catch { throw new Error('This file is not a training report (not valid JSON).'); }
    if (parsed.schema !== 'fredis.training-report') throw new Error('This file is not a training report.');
    report = parsed;
    directory = path.dirname(directory);
  }
  const indexed = await indexedRun(directory, report);
  if (indexed) return indexed;
  const series = new Map();
  const epochBoundaries=[];
  const files = [];
  for (const file of ['metrics.tsv', 'metric_distributions.tsv']) {
    const filename = path.join(directory, file);
    try { await fs.promises.access(filename); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    files.push(file);
    const input = fs.createReadStream(filename);
    const lines = readline.createInterface({ input, crlfDelay: Infinity });
    let headers;
    try {
      for await (const line of lines) {
        const cells = line.split('\t');
        if (!headers) { headers = cells; continue; }
        if (cells.length !== headers.length) continue; // A live writer may leave an incomplete final row.
        const row = Object.fromEntries(headers.map((key, i) => [key, cells[i]]));
        const step = Number(row.global_step);
        if(file==='metrics.tsv'&&(row.event==='epoch_end'||!row.event)&&row.global_step?.trim()&&Number.isFinite(step)&&row.epoch?.trim()&&Number.isFinite(Number(row.epoch)))epochBoundaries.push({step,epoch:Number(row.epoch)});
        if (!row.global_step || !Number.isFinite(step)) continue;
        const add = (key, raw) => {
          const value = Number(raw);
          if (!raw?.trim() || !Number.isFinite(value)) return;
          if (!series.has(key)) series.set(key, new Map());
          series.get(key).set(step, value);
        };
        if (file === 'metric_distributions.tsv') add('distributions/' + row.split + '/' + row.metric, row.value);
        else {
          for (const split of ['train','validation','val']) {
            const count = Number(row[split + '_batches']);
            if (count > 0 && !row[split + '_total_samples'] && row[split + '_samples']) {
              row[split + '_total_samples'] = row[split + '_samples'];
              row[split + '_total_batches'] = row[split + '_batches'];
              row[split + '_samples'] = String(Number(row[split + '_samples']) / count);
              row[split + '_batches'] = String(count);
            }
          }
          for(const split of ['train','validation','val'])if(row[split+'_total_batches'])row[split+'_batches']=row[split+'_total_batches'];
          for (const key of Object.keys(row)) {
          if (['event', 'epoch', 'global_step', 'branch_id'].includes(key)) continue;
          // Interval and epoch rows share train_* columns in the recorded TSV.
          const name = row.event === 'train_step' && key.startsWith('train_') && !key.startsWith('train_window_')
            ? 'train_window_' + key.slice('train_'.length) : key;
          add('metrics/' + name, row[key]);
          }
        }
      }
    } finally { lines.close(); input.destroy(); }
  }
  if (!files.length) throw new Error('No metrics.tsv or metric_distributions.tsv found. Select a training run directory.');
  let maxDepth = null;
  let intensityWeight;
  try {
    const configPath = safeResolve(directory, report?.artifacts?.trainingConfiguration || TRAINING_CONFIG_NAME);
    const config = JSON.parse(await fs.promises.readFile(configPath, 'utf8'));
    intensityWeight = Number(config.intensityLossWeight ?? config.intensity_loss_weight ?? 1);
    const value = config.modelConfig?.fragmenter_params?.fragment_ion_tree_builder?.max_reaction_count;
    if (Number.isInteger(value) && value >= 0) maxDepth = value;
  } catch {}
  addLegacyLossSeries(series, intensityWeight);
  const validation = report ? await validationResult(directory, report) : null;
  return { directory, report, validation, files, maxDepth, epochBoundaries, series: [...series].sort(([a], [b]) => a.localeCompare(b)).map(([name, values]) => ({ name, points: [...values].sort((a, b) => a[0] - b[0]) })) };
}
// Keep grouping independent of the webview so recorded naming conventions are testable.
function groupSeries(series, maxDepth = null) {
  const charts = new Map();
  for (const source of series) {
    let metric, split, statistic = 'value';
    if (source.name.startsWith('distributions/')) {
      const match = source.name.match(/^distributions\/([^/]+)\/(.*)_(min|q1|q10|q25|median|q3|q75|q90|max|mean)$/);
      if (!match) continue;
      [, split, metric, statistic] = match;
      statistic = ({q1:'q25',q3:'q75',min:'q10',max:'q90'})[statistic] || statistic;
    } else {
      metric = source.name.replace(/^metrics\//, '');
      const match = metric.match(/^(intermediate_validation|train_window|train|validation|val)_(.*)$/);
      split = match ? match[1] : 'all';
      if (match) metric = match[2];
    }
    if (split === 'val') split = 'validation';
    if(metric.split('@')[0]==='hydrogen_shift_mae')continue;
    if(metric==='batches'&&(statistic!=='value'||split==='train_window'))continue;
    const facet = metric.match(/^(.*?)@([^:]+):(.+)$/);
    const dimension = facet ? facet[2] : '';
    const category = facet ? facet[3] : 'all';
    let base = (facet ? facet[1] : metric).replace(/(peak_(?:precision|recall)_top)(\d+)/g, (_, prefix, n) => prefix + n.padStart(2, '0'));
    const depth = dimension === 'depth' ? Number(category) : Number(base.match(/(?:^|_)depth_(\d+)$/)?.[1]);
    if (maxDepth !== null && Number.isFinite(depth) && depth > maxDepth) continue;
    const separatePrecursor = /^(spectrum_loss|cross_entropy_loss|cosine_loss|transition_cross_entropy)(_|$)/.test(base);
    const precursor = !separatePrecursor && base.endsWith('_without_precursor') ? 'withoutPrecursor' : 'withPrecursor';
    if (precursor === 'withoutPrecursor') base = base.replace(/_without_precursor$/, '');
    const name = base + (dimension ? ' @' + dimension : '');
    if (!charts.has(name)) charts.set(name, { name, metric: base, dimension, sources: [], variants: { withPrecursor: [], withoutPrecursor: [] } });
    const chart = charts.get(name);
    chart.sources.push(source.name);
    const lanes = chart.variants[precursor];
    let lane = lanes.find(item => item.split === split && item.category === category);
    if (!lane) { lane = { split, category, label: (dimension ? category + ' · ' : '') + split, stats: {} }; lanes.push(lane); }
    lane.stats[statistic] = source.points;
  }
  const energyRank = category => /^low\b/i.test(category) ? 0 : /^high\b/i.test(category) ? 2 : /^(mid\b|[0-9])/i.test(category) ? 1 : 3;
  return [...charts.values()].sort((a,b) => a.name.localeCompare(b.name)).map(chart => {
    for (const lanes of Object.values(chart.variants)) lanes.sort((a,b) => (chart.dimension === 'collision_energy' ? (energyRank(a.category) - energyRank(b.category)) : 0) || a.category.localeCompare(b.category, undefined, {numeric:true}) || a.split.localeCompare(b.split));
    return chart;
  });
}
function attach(panel) {
  const vscode = require('vscode');
  let loadedDirectory = null, loadedRequest = null, loadedReport = null;
  panel.webview.onDidReceiveMessage(async message => {
    if (message.type === 'metricsPick') {
      const picked = await vscode.window.showOpenDialog({ canSelectFiles: true, canSelectFolders: true, canSelectMany: false });
      if (picked?.[0]) panel.webview.postMessage({ type: 'metricsPicked', directory: picked[0].fsPath });
    }
    if (message.type === 'metricsSpectrumAction') {
      try {await msentityViewer.handleAction(vscode,message.action);}
      catch(error) {vscode.window.showErrorMessage(error.message);}
      return;
    }
    if (message.type === 'metricsValidation') {
      try {
        if (!loadedDirectory || loadedRequest !== message.request) throw new Error('Load the current training run first.');
        const data=await validationResult(loadedDirectory,loadedReport,message.validationKey ?? message.epoch);
        panel.webview.postMessage({type:'metricsValidationData',request:message.request,validationRequest:message.validationRequest,data});
      } catch(error) {panel.webview.postMessage({type:'metricsValidationData',request:message.request,validationRequest:message.validationRequest,error:error.message});}
      return;
    }
    if (message.type === 'metricsCharts') {
      try {
        if (!loadedDirectory || loadedRequest !== message.request) throw new Error('Load the current training run first.');
        const data = await readCharts(loadedDirectory, message.names, loadedReport);
        panel.webview.postMessage({type:'metricsChartsData', request:message.request, chartRequest:message.chartRequest, data});
      } catch (error) {panel.webview.postMessage({type:'metricsChartsData', request:message.request, chartRequest:message.chartRequest, error:error.message});}
      return;
    }
    if (message.type !== 'metricsLoad') return;
    try {
      if (typeof message.directory !== 'string' || !path.isAbsolute(message.directory.trim())) throw new Error('Enter an absolute path to the training run or result.training.pft.');
      loadedRequest = message.request; loadedDirectory = null;
      const data = await readRun(message.directory.trim());
      if (loadedRequest !== message.request) return;
      loadedDirectory = data.directory; loadedReport = data.report;
      panel.webview.postMessage({ type: 'metricsData', request: message.request, data });
    } catch (error) { panel.webview.postMessage({ type: 'metricsData', request: message.request, error: error.message }); }
  });
}
function html({ hidden = true, autoRefresh = true } = {}) {
  return `<div id="metricsApp"${hidden ? ' hidden' : ''}><section><h2>Training Report</h2><label>Run directory or result.training.pft<input id="metricsDirectory" type="text" data-path-kind="folder" placeholder="/path/to/run/result.training.pft" style="width:100%"></label><div class="actions"><button id="metricsBrowse">Browse…</button><button id="metricsLoad">Load / Refresh</button>${autoRefresh ? '<label><input id="metricsAuto" type="checkbox"> Auto refresh (15s)</label>' : ''}</div><p id="metricsStatus" role="status"></p><p class="muted">Each metric appears once. Chart data loads when selected. After adding it, switch that card between mean, median, and q10–q90 distribution views, or exclude precursor peaks. Train uses warm colors and validation uses cool colors; a breakdown (collision energy, adduct, depth) colors each category instead.</p><div class="actions"><input id="metricsFilter" type="search" placeholder="Filter charts" aria-label="Filter charts"><select id="metricsChoice" aria-label="Chart metric" style="max-width:80%"></select><button id="metricsAdd" disabled>＋ Add chart</button><button id="metricsAddAll" disabled>Add all charts</button><button id="metricsClear">Clear charts</button><label>Layout <select id="metricsLayout"><option value="grid">Grid</option><option value="vertical">Vertical</option></select></label></div></section><div id="metricsCharts" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px"></div><section id="representativeValidation" style="min-width:0;overflow-anchor:none"><h2>Representative validation spectra</h2><div class="actions"><label><span id="representativeAxisLabel">Epoch</span> <input id="representativeEpoch" type="number" min="0" step="any" style="width:100px"></label><input id="representativeEpochSlider" type="range" min="0" step="any" aria-label="Validation epoch" style="min-width:200px"><label>Validation <select id="representativeCheckpoint" style="max-width:260px"></select></label><label><input id="representativePrecursor" type="checkbox" checked> Use scores with precursor</label></div><p id="representativeStatus" class="muted" style="height:2.6em;margin:6px 0;overflow:auto"></p><p class="muted">The plot contains the union of assignment-score and cosine representatives at q10, q25, q50, q75, and q90, selected both with and without precursor peaks. Changing the checkbox changes coordinates only.</p><div id="representativeScatter"></div><div id="representativeSpectra" style="display:grid;grid-template-columns:minmax(min(100%,760px),1fr);gap:12px"></div></section></div>`;
}
function metricInfo(metric, dimension = '') {
  const groups = [
    ['Spectrum similarity', /cosine_similarity|spectrum_cosine|assignment_score|spectrum_nonempty_fraction|teacher_spectrum/],
    ['Ion assignment', /fragment_ion_adduct|hydrogen_shift|ion_assignment/],
    ['Peak assignment (top N)', /^peak_(recall|precision)_top\d+/],
    ['Loss', /loss|cross_entropy/],
    ['Branch classification', /^branch_(accuracy|precision|recall|f1|balanced_accuracy)$/],
    ['Training supervision', /num_(branch_groups|labelled_edges|positive_edges|weak_negative_edges|training_states)|branch_(positive|negative|recall)/],
    ['Probability flow', /probability/],
    ['Ion states & transitions', /num_ion_states|num_fragment_transitions/],
    ['Fragment generation', /fragment_nodes|path_depth|duplicate_states/],
    ['Optimization', /^(learning_rate|gradient_norm)$/],
    ['Dataset', /total_|validation_fraction|compounds/],
    ['Throughput', /samples|batches|interval_steps/],
    ['Performance & resources', /seconds|memory/],
  ];
  const descriptions = {
    loss: 'Total training loss: branch_weight × branch_loss + intensity_loss. Lower is better.',
    branch_loss: 'Classification loss on reaction edges with supervision labels, using positive and negative examples to learn reaction selection.',
    branch_positive_loss: 'Classification loss on positive reaction edges.',
    branch_negative_loss: 'Classification loss on negative reaction edges.',
    intensity_loss: 'Alias of the total spectrum_loss: the sum of the with-precursor and without-precursor objectives.',
    spectrum_loss: 'intensity_loss_weight × [cross entropy + cosine_loss_weight × (1 − cosine)]. Exclude precursor switches the peaks used in this objective.',
    transition_cross_entropy_without_precursor: 'Cross entropy on fragment peaks after excluding precursor peaks and renormalizing the predicted distribution. This is the unweighted diagnostic value; cross_entropy_loss includes intensity_loss_weight.',
    transition_cross_entropy: 'Cross entropy computed from predicted probabilities and normalized observed intensity weights. Lower is better.',
    full_spectrum_cosine: 'Cosine similarity of intensity distributions on training candidates, including precursor peaks. This uses a different candidate set from free-running spectrum evaluation.',
    spectrum_cosine_std: 'Standard deviation of cosine similarity between freely generated and observed spectra. Measures variation across spectra.',
    spectrum_cosine: 'Cosine similarity of intensity distributions on training candidates.',
    num_branch_groups: 'Number of reaction branch groups in the batch. Groups represent compounds paired with main adducts.',
    num_labelled_edges: 'Number of labelled reaction edges in the batch: positive edges plus weak negative edges. This is an edge count, rather than a spectrum count.',
    num_positive_edges: 'Number of reaction edges with positive labels (1) in the batch.',
    num_weak_negative_edges: 'Number of reaction edges with weak negative labels (0) in the batch.',
    num_ion_states: 'Number of ion states used for intensity computation in the batch. Different adducts and hydrogen shifts count as separate states, before aggregation by molecular formula.',
    num_fragment_transitions: 'Number of reaction transitions between fragments included in the batch computation.',
    retained_probability: 'Total probability finally retained on precursor and fragment nodes, averaged across spectra. Its sum with sink_probability is approximately 1. This does not measure prediction accuracy.',
    sink_probability: 'Mean probability flowing into the terminal sink across spectra. Its sum with retained_probability is approximately 1.',
    probability_mass_error: 'Maximum probability conservation error in the batch: |retained_probability + sink_probability − 1|. Values closer to 0 indicate better numerical conservation.',
    mean_path_depth: 'Mean reaction path depth of generated fragments.',
    duplicate_states_removed: 'Number of duplicate states removed during generation.',
    learning_rate: 'Learning rate configured in the optimizer.',
    gradient_norm: 'Gradient norm recorded during training.',
    samples_per_compound: 'For each batch: number of spectra divided by the number of distinct Source compounds (SMILES) in that batch. Epoch and train-window summaries average these batch ratios; distributions show their variation across batches. Unlike samples, this is per compound.',
    loop_seconds: 'Mean elapsed time per batch iteration, including batch preparation, forward evaluation, metric collection, and backward/optimizer steps during training. Excludes free-running validation and report publishing.',
    branch_accuracy: 'Correct labelled-edge predictions / all labelled edges. Positive when sigmoid(logit) >= 0.5. Counts are summed across batches; ignored labels (-1) are excluded. Negative labels are weak negatives, not confirmed impossible reactions.',
    branch_precision: 'TP / (TP + FP), using summed edge counts and a probability threshold of 0.5. Negative labels are weak negatives. Undefined denominators are omitted.',
    branch_recall: 'TP / (TP + FN), using summed edge counts and a probability threshold of 0.5. Undefined denominators are omitted.',
    branch_f1: '2 TP / (2 TP + FP + FN), using summed edge counts and a probability threshold of 0.5. Negative labels are weak negatives.',
    branch_balanced_accuracy: 'Mean of positive recall and negative recall, using summed edge counts and a probability threshold of 0.5. Omitted unless both classes are present.',
    fragment_ion_adduct_accuracy: 'Conditional accuracy of the highest-intensity ion state for an annotated fragment: its adduct must belong to the allowed annotation set. Non-precursor peaks only; missing fragments are reflected in coverage.',
    hydrogen_shift_accuracy: 'Conditional accuracy of the hydrogen shift for the selected ion state of an annotated fragment. Any annotated shift for that fragment is accepted. Non-precursor peaks only.',
    ion_assignment_joint_accuracy: 'Conditional accuracy requiring the predicted (adduct, hydrogen shift) pair to occur together in the annotation set for the selected fragment. Non-precursor peaks only.',
    ion_assignment_coverage: 'Fraction of eligible annotated non-precursor peaks for which a predicted state of an annotated fragment is available. Report alongside conditional ion-assignment accuracy.',
    interval_steps: 'Number of batch updates in this training log interval.',
    epoch_seconds: 'Elapsed time for the initial evaluation or one epoch, in seconds.',
    interval_seconds: 'Elapsed time for the training log interval, in seconds.',
    cuda_peak_memory_mb: 'Peak CUDA memory usage, in MB.',
  };
  const objective = metric.match(/^(spectrum_loss|cross_entropy_loss|cosine_loss)(?:_(with|without)_precursor)?$/);
  const objectiveDescriptions = {
    spectrum_loss: 'Sum of the weighted cross entropy and cosine loss contributions.',
    cross_entropy_loss: 'intensity_loss_weight × cross entropy. Uses normalized observed intensity weights and predicted probabilities. Lower is better.',
    cosine_loss: 'intensity_loss_weight × cosine_loss_weight × (1 − cosine). Lower is better.',
  };
  let description = objective ? objectiveDescriptions[objective[1]] + (objective[2] ? (objective[2] === 'with' ? ' Includes precursor peaks.' : ' Excludes precursor peaks and renormalizes the fragment distribution.') : ' Sum of the with-precursor and without-precursor objectives.') : descriptions[metric];
  if (!description) {
    if (/cosine_similarity/.test(metric)) description = 'Sum of matched predicted–observed intensity products divided by the product of their vector norms. Peaks are matched one-to-one within the m/z tolerance; unmatched peaks remain in the denominator. Higher is better.';
    else if (/assignment_score/.test(metric)) description = 'Sum of observed intensities matched by predicted m/z values divided by the total observed intensity. Predicted intensities are not used. Higher is better.';
    else if (/^peak_recall_top/.test(metric)) description = 'Number of assigned observed peaks recovered by the top N predicted peaks, ranked by predicted intensity, divided by the number of assigned observed peaks. Higher is better.';
    else if (/^peak_precision_top/.test(metric)) description = 'Number of the top N predicted peaks matching assigned observed peaks, divided by N. Higher is better.';
    else if (/nonempty_fraction/.test(metric)) description = 'Fraction of spectra with at least one generated peak.';
    else if (/fragment_nodes/.test(metric)) description = 'Generated fragment node count. Unique counts exclude duplicates; mean counts are averaged across spectra.';
    else if (/total_/.test(metric)) description = 'Total count for the dataset or evaluation interval, as specified by the metric name.';
    else if (/samples/.test(metric)) description = 'Mean number of spectra processed per batch iteration. Total dataset counts are shown separately.';
    else if (/batches/.test(metric)) description = 'Number of processed batches in the epoch (or in the intermediate validation pass). This count is not a per-batch mean; no train-window series is shown.';
    else if (/validation_fraction/.test(metric)) description = 'Fraction of data used for validation.';
    else if (/seconds/.test(metric)) description = 'Elapsed processing time, in seconds.';
    else if (/memory/.test(metric)) description = 'Memory usage during computation. The metric name specifies the unit.';
    else description = 'Value recorded in the training or validation log for ' + metric + '.';
  }
  const facet = {depth:'Evaluated using peaks at exactly depth N, rather than cumulatively through depth N. Observed peaks use their minimum explainable depth. Spectra with no observed peaks at that depth are excluded from aggregation.', main_adduct:'Aggregated separately for each main adduct.', collision_energy:'Aggregated by collision energy tertiles, ordered low → mid → high.'}[dimension];
  if (/^num_/.test(metric)) description += ' Displayed as a mean per batch iteration.';
  return {group: groups.find(([, pattern]) => pattern.test(metric))?.[0] || 'Other', description: description + (facet ? '\n' + facet : '')};
}

function metricOrder(a, b) {
  const groups = ['Spectrum similarity','Ion assignment','Peak assignment (top N)','Branch classification','Loss','Training supervision','Probability flow','Ion states & transitions','Fragment generation','Optimization','Dataset','Throughput','Performance & resources','Other'];
  const metrics = ['loss','branch_loss','branch_positive_loss','branch_negative_loss','spectrum_loss','spectrum_loss_with_precursor','spectrum_loss_without_precursor','cross_entropy_loss','cross_entropy_loss_with_precursor','cross_entropy_loss_without_precursor','cosine_loss','cosine_loss_with_precursor','cosine_loss_without_precursor','intensity_loss','transition_cross_entropy','transition_cross_entropy_without_precursor'];
  const rank = chart => { const i = metrics.indexOf(chart.metric); return i < 0 ? metrics.length : i; };
  const facet = chart => ({'':0, main_adduct:1, collision_energy:2, depth:3}[chart.dimension || ''] ?? 4);
  const groupDiff = groups.indexOf(metricInfo(a.metric).group) - groups.indexOf(metricInfo(b.metric).group);
  return groupDiff || rank(a) - rank(b) || facet(a) - facet(b) || a.metric.localeCompare(b.metric, undefined, {numeric:true}) || a.name.localeCompare(b.name, undefined, {numeric:true});
}

function availableDisplayModes(lanes) {
  const has=(lane,stat)=>Array.isArray(lane.stats[stat])&&lane.stats[stat].length>0;
  return ['mean','median','distribution'].filter(mode=>lanes.some(lane=>mode==='mean'?(has(lane,'mean')||has(lane,'value')):mode==='median'?has(lane,'median'):['q10','q25','median','q75','q90'].every(stat=>has(lane,stat))));
}

function client(groupSeries, metricInfo, metricOrder, representativePoints, msentityAssets, installMsentityViewer, comparisonPayload, createEpochAxis, availableDisplayModes) {
  const el = id => document.getElementById(id);
  const state = vscode.getState() || {};
  const spectrumViewer=installMsentityViewer(msentityAssets,vscode);
  let selected = state.metricsSelected || [], data = [], report = null, validation = null, request = 0, busy = false;
  let lazy = false, chartRequest = 0, maxDepth = null;
  let epochBoundaries=[],epochAxis=createEpochAxis([]);
  const loadedCharts = new Set(), pendingCharts = new Map(), chartErrors = new Map();
  function requestSelectedCharts() {
    if (!lazy || busy) return;
    const pending = new Set([...pendingCharts.values()].flat());
    const names = selected.filter(name => !loadedCharts.has(name) && !pending.has(name) && data.some(chart=>chart.name===name));
    if (!names.length) return;
    const id = ++chartRequest; pendingCharts.set(id,names);
    vscode.postMessage({type:'metricsCharts',request,chartRequest:id,names});
  }
  const cardSettings = new Map(Object.entries(state.metricsCardSettings || {}));
  const hiddenLanes = new Map();
  const splitColors={train:'#e76f51',train_window:'#f4a261',validation:'#3a86ff',intermediate_validation:'#2a9d8f',all:'#8b8f98'};
  const splitDashes={train:'',train_window:'4 2',validation:'',intermediate_validation:'6 3',all:''};
  const categoryColors=['#e76f51','#2a9d8f','#e9c46a','#264653','#8ab17d','#f4a261','#577590','#b56576','#6d597a','#ee6c4d'];
  const GROUP_ORDER=['Spectrum similarity','Ion assignment','Peak assignment (top N)','Branch classification','Loss','Training supervision','Probability flow','Ion states & transitions','Fragment generation','Optimization','Dataset','Throughput','Performance & resources','Other'];
  const metricGroup=metric=>metricInfo(metric).group;
  const border='var(--border,rgba(128,128,128,.35))', accent='var(--accent,#36c5a2)';
  el('metricsDirectory').value = state.metricsDirectory || '';
  el('metricsLayout').value = state.metricsLayout === 'vertical' ? 'vertical' : 'grid';
  let selectedValidationKey=state.representativeValidationKey ?? null;
  let selectedEpoch=state.representativeEpoch ?? null, selectedSpectrumKey=state.representativePoint ?? null, validationRequest=0, epochTimer=null, epochLoading=false;
  if (typeof state.representativePrecursor === 'boolean') el('representativePrecursor').checked = state.representativePrecursor;
  const applyLayout = () => { el('metricsCharts').style.gridTemplateColumns = el('metricsLayout').value === 'vertical' ? 'minmax(0, 1fr)' : 'repeat(auto-fit,minmax(min(100%,440px),1fr))'; };
  applyLayout();
  const save = () => vscode.setState({ ...(vscode.getState() || {}), metricsSelected: selected, metricsCardSettings:Object.fromEntries(cardSettings), metricsLayout: el('metricsLayout').value, metricsDirectory: el('metricsDirectory').value, representativePrecursor: el('representativePrecursor').checked, representativeEpoch:selectedEpoch, representativeValidationKey:selectedValidationKey, representativePoint:selectedSpectrumKey });
  function svgNode(svg, tag, attributes, text) { const node=document.createElementNS('http://www.w3.org/2000/svg',tag); for(const [key,value] of Object.entries(attributes))node.setAttribute(key,value); if(text!==undefined)node.textContent=text; svg.append(node); return node; }
  function load() {
    if (busy) return;
    busy = true; el('metricsLoad').disabled = true; el('metricsStatus').textContent = 'Loading…'; save();
    vscode.postMessage({ type: 'metricsLoad', directory: el('metricsDirectory').value, request: ++request });
  }
  const matchingCharts = () => {
    const query = el('metricsFilter').value.trim().toLowerCase();
    return data.filter(chart => chart.name.toLowerCase().includes(query));
  };
  function updateChoices() {
    const previous = el('metricsChoice').value;
    const matching = matchingCharts().sort(metricOrder);
    el('metricsChoice').replaceChildren();
    const byGroup = new Map();
    for (const chart of matching) { const group = metricGroup(chart.metric); if (!byGroup.has(group)) byGroup.set(group, []); byGroup.get(group).push(chart); }
    for (const group of GROUP_ORDER) {
      const charts = byGroup.get(group);
      if (!charts?.length) continue;
      const optgroup = document.createElement('optgroup'); optgroup.label = group;
      for (const chart of charts) { const option = document.createElement('option'); option.value = option.textContent = chart.name; optgroup.append(option); }
      el('metricsChoice').append(optgroup);
    }
    if (matching.some(chart => chart.name === previous)) el('metricsChoice').value = previous;
    el('metricsAdd').disabled = !matching.length;
    el('metricsAddAll').disabled = !matching.length;
    el('metricsAddAll').textContent = (el('metricsFilter').value.trim() ? 'Add matching charts' : 'Add all charts') + ' (' + matching.length + ')';
  }
  let renderVersion = 0;
  let chartsByName = new Map();
  const cardElements = new Map();
  function buildCard(name, chart) {
    const card = document.createElement('section'); card.style.cssText = 'display:block;content-visibility:auto;contain-intrinsic-size:auto 380px';
    const title = document.createElement('strong'); title.textContent = name;
    const remove = document.createElement('button'); remove.textContent = '×'; remove.title = 'Remove chart'; remove.style.float = 'right';
    remove.onclick = () => { selected = selected.filter(n => n !== name); cardElements.delete(name); save(); render(); };
    card.append(remove, title);
    const help = document.createElement('details');
    const helpSummary = document.createElement('summary'); helpSummary.textContent = 'About this metric'; helpSummary.style.cursor = 'pointer';
    const helpText = document.createElement('p'); helpText.className = 'muted'; helpText.style.whiteSpace = 'pre-line';
    helpText.textContent = metricInfo(chart.metric, chart.dimension).description;
    help.append(helpSummary, helpText); card.append(help);
    if (lazy && (!loadedCharts.has(name) || chartErrors.has(name))) {
      const status = document.createElement('p');status.className='muted';status.textContent=chartErrors.get(name) || 'Loading chart…';card.append(status);
      if (chartErrors.has(name)) {const retry=document.createElement('button');retry.textContent='Retry';retry.onclick=()=>{loadedCharts.delete(name);chartErrors.delete(name);render();};card.append(retry);}
      return card;
    }
    const availableWith = chart.variants.withPrecursor.length > 0, availableWithout = chart.variants.withoutPrecursor.length > 0;
    const settings = { mode:'mean', withoutPrecursor:!availableWith && availableWithout, ...(cardSettings.get(name) || {}) };
    if (settings.withoutPrecursor && !availableWithout) settings.withoutPrecursor=false;
    if (!settings.withoutPrecursor && !availableWith) settings.withoutPrecursor=true;
    cardSettings.set(name,settings);
    const hidden = hiddenLanes.get(name) || new Set(Object.values(chart.variants).flat().filter(lane=>lane.split==='train_window').map(lane=>lane.label));
    hiddenLanes.set(name, hidden);
    const variantLanes=chart.variants[settings.withoutPrecursor?'withoutPrecursor':'withPrecursor'];
    const shownLanes=variantLanes.filter(lane=>!hidden.has(lane.label));
    const supportedModes=availableDisplayModes(shownLanes.length?shownLanes:variantLanes);
    if(!supportedModes.includes(settings.mode))settings.mode=supportedModes[0]||'mean';
    const controls=document.createElement('div');controls.style.cssText='display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0';
    const modeLabel=document.createElement('label');modeLabel.append(document.createTextNode('Display '));const mode=document.createElement('select');
    for(const [value,labelText] of [['mean','Mean'],['median','Median'],['distribution','Distribution']]){const option=document.createElement('option');option.value=value;option.textContent=labelText;option.disabled=!supportedModes.includes(value);mode.append(option);}mode.value=settings.mode;mode.onchange=()=>{settings.mode=mode.value;save();refreshCard(name);};modeLabel.append(mode);controls.append(modeLabel);
    if (availableWith && availableWithout) {
    const precursorLabel=document.createElement('label'),precursor=document.createElement('input');precursor.type='checkbox';precursor.checked=settings.withoutPrecursor;precursor.onchange=()=>{settings.withoutPrecursor=precursor.checked;save();refreshCard(name);};precursorLabel.append(precursor,document.createTextNode(' Exclude precursor'));controls.append(precursorLabel);
    }
    let memoryUnit = settings.memoryUnit;
    if (/memory_mb$/.test(chart.metric)) {
      const peak = Object.values(chart.variants).reduce((maximum, lanes) => lanes.reduce((m, lane) => Object.values(lane.stats).reduce((n, points) => points.reduce((v, point) => Math.max(v, Math.abs(point[1])), n), m), maximum), 0);
      if (!['MB','GB','TB'].includes(memoryUnit)) memoryUnit = peak >= 1024*1024 ? 'TB' : peak >= 1024 ? 'GB' : 'MB';
      const unitLabel = document.createElement('label'); unitLabel.append(document.createTextNode('Unit '));
      const unit = document.createElement('select');
      for (const value of ['MB','GB','TB']) { const option = document.createElement('option'); option.value = option.textContent = value; unit.append(option); }
      unit.value = memoryUnit; unit.onchange = () => {settings.memoryUnit=unit.value;save();refreshCard(name);};
      unitLabel.append(unit); controls.append(unitLabel);
      title.textContent = name.replace(/_mb$/, '') + ' (' + memoryUnit + ')';
    }
    card.append(controls);

    const memoryScale = memoryUnit ? ({MB:1,GB:1024,TB:1024*1024}[memoryUnit] || 1) : 1;
    const allLanes = chart.variants[settings.withoutPrecursor?'withoutPrecursor':'withPrecursor'].map(lane => ({...lane, stats:Object.fromEntries(Object.entries(lane.stats).map(([stat,points]) => [stat,points.map(([step,value]) => [step,value/memoryScale])]))}));
    const lanes = allLanes.filter(lane => !hidden.has(lane.label));
    const categories=[...new Set(allLanes.map(lane=>lane.category))];
    const faceted = !!chart.dimension && categories.length>1;
    const color = lane => faceted ? categoryColors[categories.indexOf(lane.category)%categoryColors.length] : (splitColors[lane.split] || splitColors.all);
    const dash = lane => faceted ? (splitDashes[lane.split] || '') : '';
    const legend = document.createElement('div'); legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin:12px 0';
    for (const lane of allLanes) {
      const label = document.createElement('label'); label.style.cssText = 'display:inline-flex;align-items:center;gap:6px;cursor:pointer';
      const toggle = document.createElement('input'); toggle.type = 'checkbox'; toggle.checked = !hidden.has(lane.label);
      toggle.onchange = () => { if (toggle.checked) hidden.delete(lane.label); else hidden.add(lane.label); refreshCard(name); };
      const swatch = document.createElementNS('http://www.w3.org/2000/svg','svg'); swatch.setAttribute('viewBox','0 0 28 10'); swatch.style.cssText='width:26px;height:10px;flex:0 0 auto';
      const swatchLine = document.createElementNS('http://www.w3.org/2000/svg','line'); swatchLine.setAttribute('x1',1); swatchLine.setAttribute('x2',27); swatchLine.setAttribute('y1',5); swatchLine.setAttribute('y2',5);
      swatchLine.setAttribute('stroke',color(lane)); swatchLine.setAttribute('stroke-width',3); swatchLine.setAttribute('stroke-dasharray',dash(lane)); swatchLine.setAttribute('stroke-linecap','round'); swatch.append(swatchLine);
      label.append(toggle, swatch, document.createTextNode(lane.label)); legend.append(label);
    }
    card.append(legend);
    if (settings.mode === 'distribution') { const hint = document.createElement('p'); hint.className = 'muted'; hint.textContent = 'Light band: q10–q90 · Dark band: q25–q75 · Line: median'; card.append(hint); }
    const visibleStats=lane=>settings.mode==='distribution'?['q10','q25','median','q75','q90']:settings.mode==='median'?['median']:['mean','value'];
    const points = lanes.flatMap(lane => visibleStats(lane).flatMap(stat=>lane.stats[stat]||[])).filter(point=>epochAxis.at(point[0])!==null);
    if (!points.length) { const p = document.createElement('p'); p.textContent = 'No measurements available to display.'; card.append(p); }
    else {
      const ns = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 600 260'); svg.style.width = '100%'; svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', name);
      const node = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
      let min = Infinity, max = -Infinity, first = Infinity, last = -Infinity;
      for (const [step, value] of points) { min = Math.min(min, value); max = Math.max(max, value); first = Math.min(first, epochAxis.at(step)); last = Math.max(last, epochAxis.at(step)); }
      if (min === max) { min -= Math.abs(min) * .05 || 1; max += Math.abs(max) * .05 || 1; }
      const x = step => 75 + (epochAxis.at(step) - first) / (last - first || 1) * 500;
      const y = value => 215 - (value - min) / (max - min) * 190;
      const coords = values => values.map(p => x(p[0])+','+y(p[1])).join(' ');
      for (let i = 0; i <= 4; i++) {
        const value = min + (max-min)*i/4;
        node('line', { x1:75, x2:575, y1:y(value), y2:y(value), stroke:'currentColor', opacity:'.15' });
        node('text', { x:70, y:y(value)+4, fill:'currentColor', 'font-size':11, 'text-anchor':'end' }, value.toPrecision(4) + (memoryUnit ? ' ' + memoryUnit : ''));
      }
      for (const lane of lanes) {
        const line = (values, opacity = 1, width = 2) => {
          values=values?.filter(point=>epochAxis.at(point[0])!==null);
          if (!values?.length) return;
          node('polyline', { points:coords(values), fill:'none', stroke:color(lane), 'stroke-width':width, 'stroke-dasharray':dash(lane), opacity });
          if (values.length === 1) node('circle', { cx:x(values[0][0]), cy:y(values[0][1]), r:3, fill:color(lane), opacity });
        };
        const band = (lower, upper, opacity) => {
          const upperMap = new Map(upper || []);
          const paired = (lower || []).filter(p => upperMap.has(p[0])&&epochAxis.at(p[0])!==null);
          if (paired.length === 1) node('line', { x1:x(paired[0][0]), x2:x(paired[0][0]), y1:y(paired[0][1]), y2:y(upperMap.get(paired[0][0])), stroke:color(lane), 'stroke-width':8, opacity });
          else if (paired.length) node('polygon', { points:coords(paired)+' '+coords(paired.map(p => [p[0], upperMap.get(p[0])]).reverse()), fill:color(lane), opacity });
        };
        if (settings.mode === 'distribution') {
          band(lane.stats.q10, lane.stats.q90, .10); band(lane.stats.q25, lane.stats.q75, .25);
          for (const stat of ['q10', 'q90', 'q25', 'q75']) line(lane.stats[stat], .45, 1);
          line(lane.stats.median);
        } else line(lane.stats[settings.mode] || (settings.mode==='mean'?lane.stats.value:null));
      }
      node('text', { x:75, y:240, fill:'currentColor', 'font-size':12 }, epochAxis.format(first));
      node('text', { x:575, y:240, fill:'currentColor', 'font-size':12, 'text-anchor':'end' }, epochAxis.format(last));
      node('text',{x:325,y:257,fill:'currentColor','font-size':12,'text-anchor':'middle'},epochAxis.mode==='step'?'Step':'Epoch');
      const info = document.createElement('p'); info.className = 'muted'; info.style.whiteSpace = 'pre-line'; info.textContent = (faceted ? 'Color: category · Dashed: intermediate validation' : 'Color: split') + ' · Hover to inspect' + (points.some(point=>epochAxis.estimated(point[0]))?' · Current-epoch positions are estimated until its end step is recorded.':'');
      svg.onmousemove = event => {
        const rect = svg.getBoundingClientRect(); const epoch = first + ((event.clientX-rect.left)/rect.width*600-75)/500*(last-first);
        info.textContent = lanes.map(lane => {
          const values = visibleStats(lane).filter(stat=>lane.stats[stat]?.some(point=>epochAxis.at(point[0])!==null)).map(stat => { const samples=lane.stats[stat].filter(point=>epochAxis.at(point[0])!==null);
            let closest = samples[0]; for (const p of samples) if (Math.abs(epochAxis.at(p[0])-epoch)<Math.abs(epochAxis.at(closest[0])-epoch)) closest=p;
            return stat+'='+closest[1].toPrecision(5)+(memoryUnit?' '+memoryUnit:'')+' ('+epochAxis.mode+' '+epochAxis.format(epochAxis.at(closest[0]))+(epochAxis.estimated(closest[0])?' estimated':'')+')';
          });
          return lane.label+': '+values.join(' · ');
        }).join('\n');
      };
      card.append(svg, info);
    }
    return card;
  }
  // Re-rendering just the one card that changed (instead of tearing down the whole
  // grid) keeps every other card's DOM node intact, so the page doesn't lose scroll
  // position or jump to the top whenever a per-card control is touched.
  function refreshCard(name) {
    const chart = chartsByName.get(name), old = cardElements.get(name);
    if (!chart || !old) { render(); return; }
    const updated = buildCard(name, chart);
    old.replaceWith(updated);
    cardElements.set(name, updated);
  }
  async function render() {
    requestSelectedCharts();
    const version = ++renderVersion;
    let rendered = 0;
    el('metricsCharts').replaceChildren();
    cardElements.clear();
    chartsByName = new Map(data.map(chart => [chart.name, chart]));
    const orderedSelected = [...selected].sort((a,b) => {
      const ca=chartsByName.get(a), cb=chartsByName.get(b);
      return ca && cb ? metricOrder(ca, cb) : a.localeCompare(b);
    });
    const showGroups = new Set(orderedSelected.filter(name => chartsByName.has(name)).map(name => metricGroup(chartsByName.get(name).metric))).size > 1;
    let lastGroup = null;
    for (const name of orderedSelected) {
      if (rendered > 0 && rendered % 8 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
        if (version !== renderVersion) return;
      }
      const chart = chartsByName.get(name);
      if (!chart) continue;
      const group = metricGroup(chart.metric);
      if (showGroups && group !== lastGroup) {
        const header = document.createElement('div');
        header.style.cssText = 'grid-column:1/-1;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.7;margin:18px 0 -4px;padding-top:14px;border-top:1px solid '+border+(lastGroup===null?';margin-top:0;padding-top:0;border-top:0':'');
        header.textContent = group;
        el('metricsCharts').append(header);
        lastGroup = group;
      }
      const card = buildCard(name, chart);
      el('metricsCharts').append(card);
      cardElements.set(name, card);
      rendered++;
    }
  }
  function peaks(record, kind) { const values=record[kind+'_peaks']; if(values) return values; return (record[kind+'_mz']||[]).map((mz,index)=>({mz,intensity:record[kind+'_intensity'][index],precursor:false})); }
  const validationCheckpoints=()=>validation?.checkpoints || (validation?.epochs||[]).map(epoch=>({key:'epoch_'+epoch,epoch}));
  function requestEpoch(value, immediate=false, key=null) {
    const checkpoints=validationCheckpoints();
    if(!checkpoints.length||String(value).trim()===''||!Number.isFinite(Number(value)))return;
    const point=key?checkpoints.find(item=>item.key===key):checkpoints.reduce((best,current)=>Math.abs(current.epoch-value)<Math.abs(best.epoch-value)?current:best,checkpoints[0]);
    if(!point)return;
    selectedEpoch=point.epoch;selectedValidationKey=point.key;
    el('representativeEpoch').value=point.epoch;el('representativeEpochSlider').value=point.epoch;save();
    clearTimeout(epochTimer);const token=++validationRequest;
    if(point.key===(validation.key||'epoch_'+validation.epoch)){epochLoading=false;renderSpectra();return;}
    epochLoading=true;renderSpectra();
    const send=()=>vscode.postMessage({type:'metricsValidation',request,validationRequest:token,epoch:point.epoch,validationKey:point.key});
    if(immediate)send();else epochTimer=setTimeout(send,150);
  }
  function renderSpectra() {
    const epochs=validation?.epochs || [];
    const epochInput=el('representativeEpoch'),slider=el('representativeEpochSlider');
    for(const control of [epochInput,slider]) {control.disabled=!epochs.length;control.min=epochs[0]??0;control.max=epochs.at(-1)??0;control.step='any';control.value=selectedEpoch??validation?.epoch??epochs.at(-1)??0;}
    el('representativeAxisLabel').textContent=validation?.axisMode==='step'?'Step':'Epoch';
    const checkpointSelect=el('representativeCheckpoint');checkpointSelect.replaceChildren();
    for(const point of validationCheckpoints()){const option=document.createElement('option');option.value=point.key;option.textContent=(validation?.axisMode==='step'?'Step ':'Epoch ')+epochAxis.format(point.epoch)+(point.estimated?' (estimated)':'')+' · '+point.key;checkpointSelect.append(option);}
    checkpointSelect.value=selectedValidationKey||validation?.key||'epoch_'+validation?.epoch;checkpointSelect.disabled=!validationCheckpoints().length;
    const plotBox=el('representativeScatter'),spectrumBox=el('representativeSpectra');
    for (const box of [plotBox,spectrumBox]) {box.setAttribute('aria-busy',String(epochLoading));box.inert=epochLoading;box.style.opacity=epochLoading?'.55':'';}
    if (epochLoading) {el('representativeStatus').textContent='Loading '+(validation?.axisMode==='step'?'step ':'epoch ')+selectedEpoch+'…';return;}
    plotBox.replaceChildren();
    const points=representativePoints(validation?.representatives);
    if (!points.length) {spectrumBox.hidden=true;el('representativeStatus').textContent=report?'No validation spectrum artifact is available yet.':'Load a training report to see representative spectra.';return;}
    spectrumBox.hidden=false;
    if (!points.some(point=>point.key===selectedSpectrumKey)) selectedSpectrumKey=points[0].key;
    const include=el('representativePrecursor').checked,suffix=include?'':'_without_precursor';
    const number=value=>value==null||!Number.isFinite(Number(value))?'unavailable':Number(value).toFixed(4);
    const choose=key=>{selectedSpectrumKey=key;save();renderSpectra();};
    const scatterWidth=Math.max(320,plotBox.clientWidth||680);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 '+scatterWidth+' 300');svg.style.cssText='width:100%;height:300px;display:block';svg.setAttribute('role','img');svg.setAttribute('aria-label','Assignment score versus cosine similarity');
    const x=value=>55+Math.max(0,Math.min(1,Number(value)||0))*(scatterWidth-71),y=value=>250-Math.max(0,Math.min(1,Number(value)||0))*220;
    for(let tick=0;tick<=5;tick++) {const value=tick/5;
      svgNode(svg,'line',{x1:x(value),x2:x(value),y1:30,y2:250,stroke:'currentColor',opacity:'.15'});
      svgNode(svg,'line',{x1:55,x2:scatterWidth-16,y1:y(value),y2:y(value),stroke:'currentColor',opacity:'.15'});
      svgNode(svg,'text',{x:x(value),y:270,'text-anchor':'middle',fill:'currentColor','font-size':12},value.toFixed(1));
      svgNode(svg,'text',{x:45,y:y(value)+4,'text-anchor':'end',fill:'currentColor','font-size':12},value.toFixed(1));
    }
    svgNode(svg,'text',{x:(scatterWidth+39)/2,y:294,'text-anchor':'middle',fill:'currentColor','font-size':14},'Assignment score');
    svgNode(svg,'text',{transform:'translate(16 140) rotate(-90)','text-anchor':'middle',fill:'currentColor','font-size':14},'Cosine similarity');
    for(const point of points) {
      const a=point.record['assignment_score'+suffix],c=point.record['cosine_similarity'+suffix];
      const missing=a==null||c==null||!Number.isFinite(Number(a))||!Number.isFinite(Number(c));
      const label='Assignment '+number(a)+' · Cosine '+number(c)+' · '+point.selections.join(', ');
      const circle=svgNode(svg,'circle',{cx:x(a),cy:y(c),r:point.key===selectedSpectrumKey?4:2.5,fill:missing?'none':point.key===selectedSpectrumKey?'#f0883e':'#58a6ff',stroke:point.key===selectedSpectrumKey?'#f0883e':'#58a6ff','stroke-width':1.25,'stroke-dasharray':missing?'2 2':'',tabindex:0,role:'button','aria-label':label,'data-spectrum-key':point.key});
      circle.style.cursor='pointer';svgNode(circle,'title',{},label+(missing?' · Unavailable coordinates shown at the axis edge.':''));
      circle.onclick=()=>choose(point.key);circle.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose(point.key);}};
    }
    plotBox.append(svg);
    const hint=document.createElement('p');hint.className='muted';hint.textContent='Click a point to show its generated/observed spectrum. All quantile representatives remain in the plot when score mode changes. Hollow points have unavailable scores and are placed at the axis edge.';plotBox.append(hint);
    const pickerLabel=document.createElement('label');pickerLabel.append(document.createTextNode('Spectrum (also resolves overlapping points) '));
    const picker=document.createElement('select');picker.id='representativePoint';picker.style.cssText='width:100%;min-width:0;max-width:100%';pickerLabel.style.display='block';
    points.forEach((point,index)=>{const option=document.createElement('option');option.value=point.key;option.textContent=(index+1)+' · '+point.selections.join(' / ');picker.append(option);});picker.value=selectedSpectrumKey;picker.onchange=()=>choose(picker.value);pickerLabel.append(picker);plotBox.append(pickerLabel);
    const point=points.find(point=>point.key===selectedSpectrumKey),record=point.record;
    let card=spectrumBox.querySelector('article');
    if(!card){
      card=document.createElement('article');card.style.cssText='border:1px solid '+border+';border-radius:9px;padding:12px;min-width:0';
      const mount=document.createElement('div');mount.dataset.msentityMount='';
      const details=document.createElement('div');details.dataset.spectrumDetails='';
      card.append(mount,details);spectrumBox.replaceChildren(card);
    }
    const details=card.querySelector('[data-spectrum-details]');details.replaceChildren();
    spectrumViewer.update(card.querySelector('[data-msentity-mount]'),comparisonPayload(record,validation.file+':'+point.key));
    const title=document.createElement('strong');title.style.cssText='display:block;height:1.5em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';title.textContent='Saved assignment '+number(record['assignment_score'+suffix])+' · Saved cosine '+number(record['cosine_similarity'+suffix])+' · '+(include?'with':'without')+' precursor scores';details.append(title);
    const note=document.createElement('p');note.className='muted';note.style.cssText='height:2.8em;overflow:auto';note.textContent='Saved validation scores drive the scatter plot. The msentity Similarity control independently recalculates similarity using its selected method and Da tolerance.';details.append(note);
    const smiles=document.createElement('p');smiles.style.cssText='height:2.8em;overflow:auto;overflow-wrap:anywhere';smiles.textContent='SMILES: '+(record.smiles||record.SMILES||'Unavailable in this legacy artifact.');details.append(smiles);
    const reasons=document.createElement('p');reasons.className='muted';reasons.style.cssText='height:2.8em;overflow:auto';reasons.textContent='Selected by: '+point.selections.join(' / ');details.append(reasons);
    el('representativeStatus').textContent=(validation.axisMode==='step'?'Step ':'Epoch ')+epochAxis.format(validation.epoch)+(validation.estimated?' (estimated)':'')+' · '+validation.file+' · '+points.length+' representative spectra · '+(validation.msdsFile||'legacy JSON');
  }
  if (typeof ResizeObserver !== 'undefined') {
    let lastWidth=0;
    new ResizeObserver(entries=>{const width=entries[0].contentRect.width;if(Math.abs(width-lastWidth)>1){lastWidth=width;if(validation&&!epochLoading)renderSpectra();}}).observe(el('representativeScatter'));
  }
  el('metricsFilter').oninput = updateChoices;
  el('metricsAddAll').onclick = () => {
    selected = [...new Set([...selected, ...matchingCharts().map(chart => chart.name)])];
    save(); render();
  };
  el('metricsClear').onclick = () => { selected = []; save(); render(); };
  el('metricsLayout').onchange = () => { applyLayout(); save(); };
  el('metricsLoad').onclick = load;
  el('metricsBrowse').onclick = () => vscode.postMessage({ type:'metricsPick' });
  el('metricsAdd').onclick = () => { const name = el('metricsChoice').value; if (name && !selected.includes(name)) { selected.push(name); save(); render(); } };
  el('representativeEpoch').oninput=()=>{clearTimeout(epochTimer);const value=el('representativeEpoch').value;epochTimer=setTimeout(()=>requestEpoch(value,true),250);};
  el('representativeEpoch').onchange=()=>requestEpoch(el('representativeEpoch').value,true);
  el('representativeEpochSlider').oninput=()=>requestEpoch(el('representativeEpochSlider').value);
  el('representativeEpochSlider').onchange=()=>requestEpoch(el('representativeEpochSlider').value,true);
  el('representativeCheckpoint').onchange=()=>requestEpoch(selectedEpoch??0,true,el('representativeCheckpoint').value);
  el('representativePrecursor').addEventListener('change', () => { save(); renderSpectra(); });
  window.addEventListener('message', async ({ data: message }) => {
    if (message.type === 'metricsPicked') { el('metricsDirectory').value = message.directory; save(); }
    if (message.type === 'metricsValidationData') {
      if (message.request!==request || message.validationRequest!==validationRequest) return;
      epochLoading=false;
      if (message.error) {renderSpectra();el('representativeStatus').textContent=message.error;return;}
      validation=message.data;selectedEpoch=validation.epoch;selectedValidationKey=validation.key||'epoch_'+validation.epoch;save();renderSpectra();
      window.dispatchEvent(new CustomEvent('trainingValidationRendered'));return;
    }
    if (message.type === 'metricsChartsData') {
      if (message.request !== request || !pendingCharts.has(message.chartRequest)) return;
      const names = pendingCharts.get(message.chartRequest);pendingCharts.delete(message.chartRequest);
      const boundariesChanged=!!message.data?.epochBoundaries&&JSON.stringify(message.data.epochBoundaries)!==JSON.stringify(epochBoundaries);
      if(message.data?.epochBoundaries){epochBoundaries=message.data.epochBoundaries;epochAxis=createEpochAxis(epochBoundaries);}
      const charts = new Map(groupSeries(message.data?.series || [], maxDepth).map(chart=>[chart.name,chart]));
      for (const name of names) {
        loadedCharts.add(name);
        if (message.error) chartErrors.set(name,message.error);
        else if (charts.has(name)) {data=data.map(chart=>chart.name===name?charts.get(name):chart);chartsByName.set(name,charts.get(name));}
        if (selected.includes(name)) refreshCard(name);
      }
      if(boundariesChanged)for(const name of selected)if(!names.includes(name))refreshCard(name);
      window.dispatchEvent(new CustomEvent('trainingMetricChartsRendered'));
      return;
    }
    if (message.type !== 'metricsData' || message.request !== request) return;
    lazy=!!message.data?.lazy;maxDepth=message.data?.maxDepth ?? null;loadedCharts.clear();pendingCharts.clear();chartErrors.clear();
    busy = false; el('metricsLoad').disabled = false;
    epochBoundaries=message.data?.epochBoundaries||[];epochAxis=createEpochAxis(epochBoundaries);
    data = groupSeries(message.data?.series || [],message.data?.maxDepth ?? null);
    report = message.data?.report || null;
    validationRequest++;clearTimeout(epochTimer);epochLoading=false;
    validation = message.data?.validation || null;
    const savedCheckpoint=validationCheckpoints().find(item=>item.key===selectedValidationKey)||validationCheckpoints().find(item=>item.epoch===selectedEpoch);
    selectedEpoch=savedCheckpoint?.epoch??validation?.epoch??null;selectedValidationKey=savedCheckpoint?.key??validation?.key??null;
    if (!message.error) {
      selected = [...new Set(selected.map(name => data.find(chart => chart.name===name||chart.sources.includes(name))?.name || name.replace(/^(mean|distribution|metrics)\//,'')))].filter(name=>data.some(chart=>chart.name===name));
      save();
    }
    el('metricsStatus').textContent = message.error || message.data.directory+' · '+data.length+' charts · '+new Date().toLocaleTimeString();
    updateChoices();
    await render();
    renderSpectra();
    if(selectedEpoch!=null&&selectedValidationKey!==(validation?.key||'epoch_'+validation?.epoch))requestEpoch(selectedEpoch,true,selectedValidationKey);
    window.dispatchEvent(new CustomEvent('trainingMetricsRendered'));
  });
  setInterval(() => { if (el('metricsAuto')?.checked && !el('metricsApp').hidden && el('metricsDirectory').value.trim()) load(); }, 15000);
  render();
  renderSpectra();
}
function script() { return '(' + client.toString() + ')(' + groupSeries.toString() + ',' + metricInfo.toString() + ',' + '(function(metricInfo){return ' + metricOrder.toString() + ';})(' + metricInfo.toString() + ')' + ',' + representativePoints.toString() + ',' + JSON.stringify(msentityViewer.assets()).replace(/</g,'\\u003c') + ',' + msentityViewer.installMsentityViewer.toString() + ',' + msentityViewer.comparisonPayload.toString() + ',' + createEpochAxis.toString() + ',' + availableDisplayModes.toString() + ');'; }
module.exports = { availableDisplayModes, readEpochBoundaries, validationResult, representativePoints, readRun, readCharts, readProgressMetrics, groupSeries, metricInfo, metricOrder, attach, html, script };
