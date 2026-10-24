const fs = require('fs');
const path = require('path');
const readline = require('readline');
const TRAINING_CONFIG_NAME = 'fragment-tree' + require('../../file-formats').format('training-config').suffix;

function safeResolve(root, relative) {
  if (!relative) return null;
  const target = path.resolve(root, relative), prefix = root.endsWith(path.sep) ? root : root + path.sep;
  return target === root || target.startsWith(prefix) ? target : null;
}

async function validationResult(root, manifest) {
  let target = safeResolve(root, manifest.latestValidation);
  if (!target || !fs.existsSync(target)) {
    const directory = path.join(root, manifest.artifacts?.validationDirectory || 'spectrum_validation');
    let files = [];
    try { files = (await fs.promises.readdir(directory)).filter(file => /^(epoch|step)_\d+\.json$/.test(file)); } catch {}
    files.sort((a,b) => (Number(a.match(/\d+/)?.[0]) || 0) - (Number(b.match(/\d+/)?.[0]) || 0));
    target = files.length ? path.join(directory, files.at(-1)) : null;
  }
  if (!target) return null;
  const value = await require('./validation-reader').readValidation(target);
  return { file: path.relative(root, target), summary: value.summary || {}, representatives: value.representatives || {} };
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
  const series = new Map();
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
        if (!row.global_step || !Number.isFinite(step)) continue;
        const add = (key, raw) => {
          const value = Number(raw);
          if (!raw?.trim() || !Number.isFinite(value)) return;
          if (!series.has(key)) series.set(key, new Map());
          series.get(key).set(step, value);
        };
        if (file === 'metric_distributions.tsv') add('distributions/' + row.split + '/' + row.metric, row.value);
        else for (const key of headers) {
          if (['event', 'epoch', 'global_step', 'branch_id'].includes(key)) continue;
          // Interval and epoch rows share train_* columns in the recorded TSV.
          const name = row.event === 'train_step' && key.startsWith('train_') && !key.startsWith('train_window_')
            ? 'train_window_' + key.slice('train_'.length) : key;
          add('metrics/' + name, row[key]);
        }
      }
    } finally { lines.close(); input.destroy(); }
  }
  if (!files.length) throw new Error('No metrics.tsv or metric_distributions.tsv found. Select a training run directory.');
  let maxDepth = null;
  try {
    const configPath = safeResolve(directory, report?.artifacts?.trainingConfiguration || TRAINING_CONFIG_NAME);
    const config = JSON.parse(await fs.promises.readFile(configPath, 'utf8'));
    const value = config.modelConfig?.fragmenter_params?.fragment_ion_tree_builder?.max_reaction_count;
    if (Number.isInteger(value) && value >= 0) maxDepth = value;
  } catch {}
  const validation = report ? await validationResult(directory, report) : null;
  return { directory, report, validation, files, maxDepth, series: [...series].sort(([a], [b]) => a.localeCompare(b)).map(([name, values]) => ({ name, points: [...values].sort((a, b) => a[0] - b[0]) })) };
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
      const match = metric.match(/^(train_window|train|validation|val)_(.*)$/);
      split = match ? match[1] : 'all';
      if (match) metric = match[2];
    }
    if (split === 'val') split = 'validation';
    const facet = metric.match(/^(.*?)@([^:]+):(.+)$/);
    const dimension = facet ? facet[2] : '';
    const category = facet ? facet[3] : 'all';
    let base = (facet ? facet[1] : metric).replace(/(peak_(?:precision|recall)_top)(\d+)/g, (_, prefix, n) => prefix + n.padStart(2, '0'));
    const depth = dimension === 'depth' ? Number(category) : Number(base.match(/(?:^|_)depth_(\d+)$/)?.[1]);
    if (maxDepth !== null && Number.isFinite(depth) && depth > maxDepth) continue;
    const precursor = base.endsWith('_without_precursor') ? 'withoutPrecursor' : 'withPrecursor';
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
  return [...charts.values()].sort((a,b) => a.name.localeCompare(b.name)).map(chart => {
    for (const lanes of Object.values(chart.variants)) lanes.sort((a,b) => (chart.dimension === 'collision_energy' ? (({low:0,mid:1,high:2}[a.category.toLowerCase()] ?? 3) - ({low:0,mid:1,high:2}[b.category.toLowerCase()] ?? 3)) : 0) || a.category.localeCompare(b.category, undefined, {numeric:true}) || a.split.localeCompare(b.split));
    return chart;
  });
}
function attach(panel) {
  const vscode = require('vscode');
  panel.webview.onDidReceiveMessage(async message => {
    if (message.type === 'metricsPick') {
      const picked = await vscode.window.showOpenDialog({ canSelectFiles: true, canSelectFolders: true, canSelectMany: false });
      if (picked?.[0]) panel.webview.postMessage({ type: 'metricsPicked', directory: picked[0].fsPath });
    }
    if (message.type !== 'metricsLoad') return;
    try {
      if (typeof message.directory !== 'string' || !path.isAbsolute(message.directory.trim())) throw new Error('Enter an absolute path to the training run or result.training.pft.');
      const data = await readRun(message.directory.trim());
      panel.webview.postMessage({ type: 'metricsData', request: message.request, data });
    } catch (error) { panel.webview.postMessage({ type: 'metricsData', request: message.request, error: error.message }); }
  });
}
function html({ hidden = true, autoRefresh = true } = {}) {
  return `<div id="metricsApp"${hidden ? ' hidden' : ''}><section><h2>Training Report</h2><label>Run directory or result.training.pft<input id="metricsDirectory" type="text" data-path-kind="folder" placeholder="/path/to/run/result.training.pft" style="width:100%"></label><div class="actions"><button id="metricsBrowse">Browse…</button><button id="metricsLoad">Load / Refresh</button>${autoRefresh ? '<label><input id="metricsAuto" type="checkbox"> Auto refresh (15s)</label>' : ''}</div><p id="metricsStatus" role="status"></p><p class="muted">Each metric appears once. After adding it, switch that card between mean, median, and q10–q90 distribution views, or exclude precursor peaks. Train uses warm colors and validation uses cool colors; a breakdown (collision energy, adduct, depth) colors each category instead.</p><div class="actions"><input id="metricsFilter" type="search" placeholder="Filter charts" aria-label="Filter charts"><select id="metricsChoice" aria-label="Chart metric" style="max-width:80%"></select><button id="metricsAdd" disabled>＋ Add chart</button><button id="metricsAddAll" disabled>Add all charts</button><button id="metricsClear">Clear charts</button><label>Layout <select id="metricsLayout"><option value="grid">Grid</option><option value="vertical">Vertical</option></select></label></div></section><div id="metricsCharts" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px"></div><section><h2>Representative validation spectra</h2><div class="actions"><label>Rank by <select id="representativeMetric"><option value="cosine_similarity">Cosine similarity</option><option value="assignment_score">Assignment score (m/z coverage)</option></select></label><label><input id="representativePrecursor" type="checkbox" checked> Include precursor</label><span id="representativeStatus" class="muted"></span></div><p class="muted">Representatives are selected separately by cosine similarity or assignment score, calculated with or without precursor peaks. Compare generated (top) and observed (bottom) spectra at q10, q25, median (q50), q75, and q90. Available once a result.training.pft report with spectrum validation is loaded.</p><div id="representativeSpectra" style="display:grid;grid-template-columns:minmax(min(100%,760px),1fr);gap:12px"></div></section></div>`;
}
function client(groupSeries) {
  const el = id => document.getElementById(id);
  const state = vscode.getState() || {};
  let selected = state.metricsSelected || [], data = [], report = null, validation = null, request = 0, busy = false;
  const cardSettings = new Map(Object.entries(state.metricsCardSettings || {}));
  const hiddenLanes = new Map();
  const splitColors={train:'#e76f51',train_window:'#f4a261',validation:'#3a86ff',intermediate_validation:'#2a9d8f',all:'#8b8f98'};
  const splitDashes={train:'',train_window:'4 2',validation:'',intermediate_validation:'6 3',all:''};
  const categoryColors=['#e76f51','#2a9d8f','#e9c46a','#264653','#8ab17d','#f4a261','#577590','#b56576','#6d597a','#ee6c4d'];
  const METRIC_GROUPS=[
    ['Spectrum similarity',/cosine_similarity|assignment_score|spectrum_nonempty_fraction|teacher_spectrum/],
    ['Peak assignment (top N)',/^peak_(recall|precision)_top\d+/],
    ['Loss',/loss/],
    ['Branch supervision',/branch_(positive|negative|recall)|num_training_states_depth/],
    ['Optimization',/^(learning_rate|gradient_norm)$/],
    ['Dataset & throughput',/samples|batches|fragment_nodes|validation_fraction/],
    ['Performance & resources',/seconds|memory/],
  ];
  const GROUP_ORDER=METRIC_GROUPS.map(item=>item[0]).concat('Other');
  const metricGroup=metric=>{for(const [label,pattern] of METRIC_GROUPS)if(pattern.test(metric))return label;return 'Other';};
  const border='var(--border,rgba(128,128,128,.35))', accent='var(--accent,#36c5a2)';
  el('metricsDirectory').value = state.metricsDirectory || '';
  el('metricsLayout').value = state.metricsLayout === 'vertical' ? 'vertical' : 'grid';
  if (['cosine_similarity','assignment_score'].includes(state.representativeMetric)) el('representativeMetric').value = state.representativeMetric;
  if (typeof state.representativePrecursor === 'boolean') el('representativePrecursor').checked = state.representativePrecursor;
  const applyLayout = () => { el('metricsCharts').style.gridTemplateColumns = el('metricsLayout').value === 'vertical' ? 'minmax(0, 1fr)' : 'repeat(auto-fit,minmax(min(100%,440px),1fr))'; };
  applyLayout();
  const save = () => vscode.setState({ ...(vscode.getState() || {}), metricsSelected: selected, metricsCardSettings:Object.fromEntries(cardSettings), metricsLayout: el('metricsLayout').value, metricsDirectory: el('metricsDirectory').value, representativeMetric: el('representativeMetric').value, representativePrecursor: el('representativePrecursor').checked, representativeQuantile: el('representativeLevel')?.value || state.representativeQuantile });
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
    const matching = matchingCharts();
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
    const availableWith = chart.variants.withPrecursor.length > 0, availableWithout = chart.variants.withoutPrecursor.length > 0;
    const settings = { mode:'mean', withoutPrecursor:!availableWith && availableWithout, ...(cardSettings.get(name) || {}) };
    if (settings.withoutPrecursor && !availableWithout) settings.withoutPrecursor=false;
    if (!settings.withoutPrecursor && !availableWith) settings.withoutPrecursor=true;
    cardSettings.set(name,settings);
    const controls=document.createElement('div');controls.style.cssText='display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0';
    const modeLabel=document.createElement('label');modeLabel.append(document.createTextNode('Display '));const mode=document.createElement('select');
    for(const [value,labelText] of [['mean','Mean'],['median','Median'],['distribution','Distribution']]){const option=document.createElement('option');option.value=value;option.textContent=labelText;mode.append(option);}mode.value=settings.mode;mode.onchange=()=>{settings.mode=mode.value;save();refreshCard(name);};modeLabel.append(mode);controls.append(modeLabel);
    if (availableWith && availableWithout) {
    const precursorLabel=document.createElement('label'),precursor=document.createElement('input');precursor.type='checkbox';precursor.checked=settings.withoutPrecursor;precursor.onchange=()=>{settings.withoutPrecursor=precursor.checked;save();refreshCard(name);};precursorLabel.append(precursor,document.createTextNode(' Exclude precursor'));controls.append(precursorLabel);
    }
    card.append(controls);
    const hidden = hiddenLanes.get(name) || new Set();
    hiddenLanes.set(name, hidden);
    const allLanes = chart.variants[settings.withoutPrecursor?'withoutPrecursor':'withPrecursor'];
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
    const visibleStats=lane=>settings.mode==='distribution'?['q10','q25','median','q75','q90']:settings.mode==='median'?['median','value']:['mean','value'];
    const points = lanes.flatMap(lane => visibleStats(lane).flatMap(stat=>lane.stats[stat]||[]));
    if (!points.length) { const p = document.createElement('p'); p.textContent = 'No measurements available to display.'; card.append(p); }
    else {
      const ns = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 600 260'); svg.style.width = '100%'; svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', name);
      const node = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
      let min = Infinity, max = -Infinity, first = Infinity, last = -Infinity;
      for (const [step, value] of points) { min = Math.min(min, value); max = Math.max(max, value); first = Math.min(first, step); last = Math.max(last, step); }
      if (min === max) { min -= Math.abs(min) * .05 || 1; max += Math.abs(max) * .05 || 1; }
      const x = step => 75 + (step - first) / (last - first || 1) * 500;
      const y = value => 215 - (value - min) / (max - min) * 190;
      const coords = values => values.map(p => x(p[0])+','+y(p[1])).join(' ');
      for (let i = 0; i <= 4; i++) {
        const value = min + (max-min)*i/4;
        node('line', { x1:75, x2:575, y1:y(value), y2:y(value), stroke:'currentColor', opacity:'.15' });
        node('text', { x:70, y:y(value)+4, fill:'currentColor', 'font-size':11, 'text-anchor':'end' }, value.toPrecision(4));
      }
      for (const lane of lanes) {
        const line = (values, opacity = 1, width = 2) => {
          if (!values?.length) return;
          node('polyline', { points:coords(values), fill:'none', stroke:color(lane), 'stroke-width':width, 'stroke-dasharray':dash(lane), opacity });
          if (values.length === 1) node('circle', { cx:x(values[0][0]), cy:y(values[0][1]), r:3, fill:color(lane), opacity });
        };
        const band = (lower, upper, opacity) => {
          const upperMap = new Map(upper || []);
          const paired = (lower || []).filter(p => upperMap.has(p[0]));
          if (paired.length === 1) node('line', { x1:x(paired[0][0]), x2:x(paired[0][0]), y1:y(paired[0][1]), y2:y(upperMap.get(paired[0][0])), stroke:color(lane), 'stroke-width':8, opacity });
          else if (paired.length) node('polygon', { points:coords(paired)+' '+coords(paired.map(p => [p[0], upperMap.get(p[0])]).reverse()), fill:color(lane), opacity });
        };
        if (settings.mode === 'distribution') {
          band(lane.stats.q10, lane.stats.q90, .10); band(lane.stats.q25, lane.stats.q75, .25);
          for (const stat of ['q10', 'q90', 'q25', 'q75']) line(lane.stats[stat], .45, 1);
          line(lane.stats.median);
        } else line(lane.stats[settings.mode] || lane.stats.value);
      }
      node('text', { x:75, y:240, fill:'currentColor', 'font-size':12 }, String(first));
      node('text', { x:575, y:240, fill:'currentColor', 'font-size':12, 'text-anchor':'end' }, String(last));
      const info = document.createElement('p'); info.className = 'muted'; info.style.whiteSpace = 'pre-line'; info.textContent = (faceted ? 'Color: category · Dashed: intermediate validation' : 'Color: split') + ' · Hover to inspect';
      svg.onmousemove = event => {
        const rect = svg.getBoundingClientRect(); const step = first + ((event.clientX-rect.left)/rect.width*600-75)/500*(last-first);
        info.textContent = lanes.map(lane => {
          const values = visibleStats(lane).filter(stat=>lane.stats[stat]?.length).map(stat => { const samples=lane.stats[stat];
            let closest = samples[0]; for (const p of samples) if (Math.abs(p[0]-step)<Math.abs(closest[0]-step)) closest=p;
            return stat+'='+closest[1].toPrecision(5)+' (step '+closest[0]+')';
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
    const version = ++renderVersion;
    let rendered = 0;
    el('metricsCharts').replaceChildren();
    cardElements.clear();
    chartsByName = new Map(data.map(chart => [chart.name, chart]));
    const orderedSelected = [...selected].sort((a,b) => {
      const ca=chartsByName.get(a), cb=chartsByName.get(b);
      const ga = ca ? GROUP_ORDER.indexOf(metricGroup(ca.metric)) : GROUP_ORDER.length;
      const gb = cb ? GROUP_ORDER.indexOf(metricGroup(cb.metric)) : GROUP_ORDER.length;
      return ga !== gb ? ga - gb : a.localeCompare(b);
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
  function renderSpectra() {
    const available = validation?.representatives || {};
    for (const option of el('representativeMetric').options) option.disabled = !available[option.value];
    if (!available[el('representativeMetric').value]) el('representativeMetric').value = 'cosine_similarity';
    const metric = el('representativeMetric').value + (el('representativePrecursor').checked ? '' : '_without_precursor');
    const entries = available[metric] || {};
    const levels = ['q10','q25','median','q75','q90'].filter(level => entries[level]);
    el('representativeSpectra').replaceChildren();
    if (levels.length) {
      const card = document.createElement('article'); card.style.cssText = 'border:1px solid '+border+';border-radius:9px;padding:12px';
      const heading = document.createElement('h4'); heading.style.cssText = 'display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:0 0 6px';
      const level = document.createElement('select'); level.id = 'representativeLevel';
      for (const value of levels) { const option = document.createElement('option'); option.value = value; option.textContent = value; level.append(option); }
      if (levels.includes(state.representativeQuantile)) level.value = state.representativeQuantile;
      const control = document.createElement('label'); control.append(document.createTextNode('Representative level '), level);
      card.append(heading, control);
      const plot = document.createElement('div'), meta = document.createElement('p'); meta.className = 'muted'; card.append(plot, meta);
      const draw = () => {
        save();
        const quantile = level.value, item = entries[quantile], record = item.spectrum, include = el('representativePrecursor').checked,
              generated = peaks(record,'generated').filter(p=>include||!p.precursor),
              original = (record.original_peaks||[]).filter(p=>include||!p.precursor),
              all = [...generated,...original],
              maximum = Math.max(...all.map(p=>Number(p.mz)),1),
              gmax = Math.max(...generated.map(p=>Number(p.intensity)),1e-8),
              omax = Math.max(...original.map(p=>Number(p.intensity)),1e-8),
              x = mz => 42+Number(mz)/maximum*516;
        heading.innerHTML = '';
        const levelLabel = document.createElement('span'); levelLabel.style.fontWeight = '600'; levelLabel.textContent = quantile+' representative';
        const badge = document.createElement('span'); badge.style.cssText = 'background:color-mix(in srgb,'+accent+' 22%,transparent);color:'+accent+';padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px';
        badge.textContent = el('representativeMetric').selectedOptions[0].textContent+' '+Number(record[metric] ?? item.value).toFixed(4);
        heading.append(levelLabel, badge);
        const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 570 270'); svg.style.display='block'; svg.style.width='100%'; svg.setAttribute('role','img'); svg.setAttribute('aria-label',quantile+' representative mirror spectrum');
        svgNode(svg,'line',{x1:42,x2:558,y1:130,y2:130,stroke:'currentColor'});
        for (const peak of generated) svgNode(svg,'line',{x1:x(peak.mz),x2:x(peak.mz),y1:130,y2:130-105*peak.intensity/gmax,stroke:peak.precursor?'#f0883e':'#58a6ff','stroke-width':2});
        for (const peak of original) svgNode(svg,'line',{x1:x(peak.mz),x2:x(peak.mz),y1:130,y2:130+105*peak.intensity/omax,stroke:peak.precursor?'#f0883e':'#3fb950','stroke-width':2});
        svgNode(svg,'text',{x:45,y:18,fill:'#58a6ff','font-size':11},'Generated');
        svgNode(svg,'text',{x:45,y:258,fill:'#3fb950','font-size':11},'Observed');
        svgNode(svg,'text',{x:558,y:148,fill:'currentColor','font-size':10,'text-anchor':'end'},maximum.toFixed(2)+' m/z');
        plot.replaceChildren(svg);
        meta.textContent = record.main_adduct+' · '+record.collision_energy+' eV · '+generated.length+' / '+original.length+' peaks';
      };
      level.onchange = draw; draw();
      el('representativeSpectra').append(card);
    }
    el('representativeStatus').textContent = validation ? validation.file+' · '+(validation.summary.samples||0)+' spectra' : (report ? 'No validation spectrum artifact is available yet.' : 'Load a result.training.pft report to see representative spectra.');
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
  el('representativeMetric').addEventListener('change', () => { save(); renderSpectra(); });
  el('representativePrecursor').addEventListener('change', () => { save(); renderSpectra(); });
  window.addEventListener('message', async ({ data: message }) => {
    if (message.type === 'metricsPicked') { el('metricsDirectory').value = message.directory; save(); }
    if (message.type !== 'metricsData' || message.request !== request) return;
    busy = false; el('metricsLoad').disabled = false;
    data = groupSeries(message.data?.series || [],message.data?.maxDepth ?? null);
    report = message.data?.report || null;
    validation = message.data?.validation || null;
    if (!message.error) {
      selected = [...new Set(selected.map(name => data.find(chart => chart.name===name||chart.sources.includes(name))?.name || name.replace(/^(mean|distribution|metrics)\//,'')))].filter(name=>data.some(chart=>chart.name===name));
      save();
    }
    el('metricsStatus').textContent = message.error || message.data.directory+' · '+data.length+' charts · '+new Date().toLocaleTimeString();
    updateChoices();
    await render();
    renderSpectra();
    window.dispatchEvent(new CustomEvent('trainingMetricsRendered'));
  });
  setInterval(() => { if (el('metricsAuto')?.checked && !el('metricsApp').hidden && el('metricsDirectory').value.trim()) load(); }, 15000);
  render();
  renderSpectra();
}
function script() { return '(' + client.toString() + ')(' + groupSeries.toString() + ');'; }
module.exports = { readRun, groupSeries, attach, html, script };
