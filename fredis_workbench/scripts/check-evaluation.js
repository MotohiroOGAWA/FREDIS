const assert = require('assert/strict');
const fs = require('fs');
const Module = require('module');

const panels = [];
const vscode = {
  ViewColumn: { One: 1 },
  window: {
    createWebviewPanel(viewType, title) {
      const panel = { viewType, title, handlers: [], messages: [] };
      panel.webview = {
        onDidReceiveMessage(handler) { panel.handlers.push(handler); },
        async postMessage(message) { panel.messages.push(message); }
      };
      panels.push(panel);
      return panel;
    }
  }
};
const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscode;
  return originalLoad.call(this, request, parent, isMain);
};
const {
  open, html, summaryArgs, groupedArgs, evaluationConfigDocument, parseEvaluationConfig,
  evaluationConfigSuffix, ensureEvaluationConfigSuffix, ensureSvgSuffix
} = require('../src/features/evaluation/editor');
Module._load = originalLoad;

const page = html();
assert.match(page, /id="bins"[^>]*disabled/);
assert.doesNotMatch(page, /Aggregation|Value column/);
assert.match(page, /Plot color/);
assert.match(page, /id="plotType"/);
assert.match(page, /id="groupedPlotType"/);
assert.match(page, /Violin plot/);
assert.match(page, /data-eval-tab="column"/);
assert.match(page, /data-eval-tab="grouped"/);
assert.match(page, /id="groupedAddGroup"/);
assert.match(page, /id="groupedAddSeries"/);
assert.match(page, /data-no-data/);
assert.match(page, /Transparent background/);
assert.match(page, /id="loadColumnConfig"/);
assert.match(page, /id="saveColumnConfig"/);
assert.match(page, /id="groupedLoadConfig"/);
assert.match(page, /id="groupedSaveConfig"/);
assert.match(page, /id="groupedGroupGap"/);
assert.match(page, /id="groupedSeriesGap"/);
assert.match(page, /id="groupedBoxWidth"/);
assert.match(page, /Collision Energy parser/);
assert.match(page, /SMILES chemical information/);
assert.match(page, /SMARTS substructure matching/);
assert.match(page, /id="smartsQueries"/);
assert.match(page, /Alcohol = \[OX2H\]/);
assert.match(page, /line\.indexOf\(' = '\)/);
assert.doesNotMatch(page, /label \+ Tab \+ SMARTS/);
assert.match(page, /Matched rows/);
assert.match(page, /Unique compounds/);
assert.match(page, /HeavyAtomCount/);
assert.match(page, /Changes are applied only when Generate plot is pressed/);
assert.match(page, /id="graphOpacity"/);
assert.match(page, /id="xLabelSize"/);
assert.match(page, /id="yLabelSize"/);
assert.match(page, /id="xAxisTitleSize"/);
assert.match(page, /id="yAxisTitleSize"/);
assert.match(page, /id="title" placeholder=/);
assert.match(page, /id="titleSize"/);
for (const id of ['categorySearch', 'categorySort', 'categorySelection', 'allShown', 'noneShown']) {
  assert.match(page, new RegExp(`id="${id}"`));
}
for (const sort of ['name-asc', 'name-desc', 'count-desc', 'count-asc', 'median-desc', 'median-asc']) {
  assert.match(page, new RegExp(`value="${sort}"`));
}
assert.match(page, /function categorySpark/);
assert.match(page, /class="category-density"/);
assert.match(page, /function sortCategories/);
assert.match(page, /setCategoryChecks\(categoryIndices\(\),true\)/);
assert.match(page, /updateCategorySelection\(\);markDraft\(\)/);
assert.match(page, /m\.result\.categoryRows\|\|m\.result\.rows/);
assert.match(page, /Distribution spans 0–1/);
assert.match(page, /id="groupedGraphOpacity"/);
assert.match(page, /id="groupedXLabelSize"/);
assert.match(page, /id="groupedYLabelSize"/);
assert.doesNotMatch(page, /id="groupedXAxisTitleSize"/);
assert.doesNotMatch(page, /id="groupedShowXAxisTitle"/);
assert.match(page, /showXAxisTitle: false/);
assert.match(page, /id="groupedYAxisTitleSize"/);
assert.match(page, /id="groupedTitleSize"/);
assert.match(page, /let groups = \[\{ id: 'g1', name: '' \}\]/);
assert.match(page, /id="progress" class="eval-progress"/);
assert.match(page, /id="groupedProgress" class="eval-progress"/);
assert.match(page, /id="input" data-path-kind="file"/);
assert.match(page, /id="joinColumn"/);
assert.match(page, /id="metadataJoinColumn"/);
assert.match(page, /SourceSpecID → SourceSpecID/);
assert.match(page, /then SourceSpecID → SpecID/);
assert.match(page, /data-result-path data-path-kind="file"/);
assert.match(page, /id="saveToast"/);
assert.match(page, /window\.showSaveToast/);
assert.match(page, /quickSaveEvaluationConfig/);
assert.match(page, /window\.evalColumn=/);
assert.match(page, /window\.evalGrouped =/);
assert.match(page, /id="xAxisTitleGap"/);
assert.match(page, /id="yAxisTitleGap"/);
assert.match(page, /id="showXAxisTitle"/);
assert.match(page, /id="showYAxisTitle"/);
assert.match(page, /id="xLabelRotation"/);
assert.doesNotMatch(page, /id="groupedXAxisTitleGap"/);
assert.match(page, /id="groupedYAxisTitleGap"/);
assert.match(page, /id="groupedShowYAxisTitle"/);
assert.match(page, /id="groupedXLabelRotation"/);
for (const match of page.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);

const compare = groupedArgs({
  width: 800, height: 500, transparent: false, backgroundColor: '#ffffff',
  groupGap: 72, seriesGap: 5, boxWidth: 30,
  xAxisTitleSize: 18, yAxisTitleSize: 19,
  groups: [{ id: 'g1', name: 'MoNA' }],
  series: [{ id: 's1', name: 'FIORA', color: '#123456' }], entries: []
}, '/tmp/result.svg');
assert.equal(compare[0], 'compare');
assert.ok(compare.includes('--request-json'));
assert.ok(compare.includes('--opaque'));
assert.ok(compare.includes('/tmp/result.svg'));
for (const option of ['--graph-opacity', '--x-label-size', '--y-label-size', '--x-axis-title-size', '--y-axis-title-size', '--title-size']) assert.ok(compare.includes(option));
assert.deepEqual(compare.slice(compare.indexOf('--x-axis-title-size'), compare.indexOf('--x-axis-title-size') + 2), ['--x-axis-title-size', '18']);
assert.deepEqual(compare.slice(compare.indexOf('--y-axis-title-size'), compare.indexOf('--y-axis-title-size') + 2), ['--y-axis-title-size', '19']);
assert.deepEqual(compare.slice(compare.indexOf('--plot-type'), compare.indexOf('--plot-type') + 2), ['--plot-type', 'box']);
const hiddenTitles = groupedArgs({
  xAxisTitleGap: 15, yAxisTitleGap: 22, showXAxisTitle: false, showYAxisTitle: false, xLabelRotation: '45',
  groups: [{ id: 'g1', name: 'MoNA' }], series: [{ id: 's1', name: 'FIORA', color: '#123456' }], entries: []
});
assert.deepEqual(hiddenTitles.slice(hiddenTitles.indexOf('--x-axis-title-gap'), hiddenTitles.indexOf('--x-axis-title-gap') + 2), ['--x-axis-title-gap', '15']);
assert.deepEqual(hiddenTitles.slice(hiddenTitles.indexOf('--y-axis-title-gap'), hiddenTitles.indexOf('--y-axis-title-gap') + 2), ['--y-axis-title-gap', '22']);
assert.deepEqual(hiddenTitles.slice(hiddenTitles.indexOf('--x-label-rotation'), hiddenTitles.indexOf('--x-label-rotation') + 2), ['--x-label-rotation', '45']);
assert.ok(hiddenTitles.includes('--hide-x-axis-title'));
assert.ok(hiddenTitles.includes('--hide-y-axis-title'));
const configDocument = evaluationConfigDocument('column', { input: '/tmp/results.mssim' });
assert.equal(configDocument.schema, 'fredis.evaluation.config');
assert.deepEqual(parseEvaluationConfig(JSON.parse(JSON.stringify(configDocument)), 'column'), configDocument.config);
assert.throws(() => parseEvaluationConfig(configDocument, 'grouped'), /Expected a grouped/);
assert.throws(() => parseEvaluationConfig({ schemaVersion: 1 }, 'column'), /not a supported/);
const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 3);
assert.equal(evaluationConfigSuffix('column'), '.evalcol.pft');
assert.equal(evaluationConfigSuffix('grouped'), '.evalgroup.pft');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.json', 'column'), '/tmp/result.evalcol.pft');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.evalgroup.pft', 'grouped'), '/tmp/result.evalgroup.pft');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.evalcol.pft', 'grouped'), '/tmp/result.evalgroup.pft');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.evalcol.pft', 'column'), '/tmp/result.evalcol.pft');
assert.equal(ensureSvgSuffix('/tmp/chart.svg'), '/tmp/chart.svg');
assert.equal(ensureSvgSuffix('/tmp/chart.SVG'), '/tmp/chart.SVG');
assert.equal(ensureSvgSuffix('/tmp/chart'), '/tmp/chart.svg');
assert.match(page, /evaluationConfigOpened/);
assert.match(page, /evaluationReady/);
const manifest = require('../package.json');
const evaluationEditor = manifest.contributes.customEditors.find(item => item.viewType === 'fredis.evaluationConfigEditor');
assert.deepEqual(evaluationEditor.selector.map(item => item.filenamePattern),
  ['*.evaltrainable.pft', '*.evalcol.pft', '*.evalgroup.pft']);
assert.match(fs.readFileSync(require.resolve('../src/extension'), 'utf8'), /evaluation\.register\(context, output, projectRoot\)/);
const editorSource = fs.readFileSync(require.resolve('../src/features/evaluation/editor'), 'utf8');
assert.match(editorSource, /FREDIS_PROGRESS/);
assert.match(editorSource, /reportProgress\('progress'\)/);
assert.match(editorSource, /reportProgress\('groupedProgress'\)/);

const args = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'AdductType', mode: 'categorical',
  metadata: '/tmp/source.msds', joinColumn: 'SourceSpecID', metadataJoinColumn: 'SpecID',
  title: 'Adduct evaluation', width: 500, height: 300, color: '#123456',
  xAxisTitleSize: 20, yAxisTitleSize: 21,
  transparent: true, include: [], order: []
});
assert.ok(args.includes('--include'));
assert.ok(args.includes('[]'));
assert.ok(!args.includes('--opaque'));
assert.deepEqual(args.slice(args.indexOf('--metadata'), args.indexOf('--metadata') + 2), ['--metadata', '/tmp/source.msds']);
assert.deepEqual(args.slice(args.indexOf('--join-column'), args.indexOf('--join-column') + 2), ['--join-column', 'SourceSpecID']);
assert.deepEqual(args.slice(args.indexOf('--metadata-join-column'), args.indexOf('--metadata-join-column') + 2), ['--metadata-join-column', 'SpecID']);
assert.ok(!args.includes('--bins'));
for (const option of ['--graph-opacity', '--x-label-size', '--y-label-size', '--x-axis-title-size', '--y-axis-title-size', '--title-size', '--title']) assert.ok(args.includes(option));
assert.deepEqual(args.slice(args.indexOf('--title'), args.indexOf('--title') + 2), ['--title', 'Adduct evaluation']);
assert.deepEqual(args.slice(args.indexOf('--x-axis-title-size'), args.indexOf('--x-axis-title-size') + 2), ['--x-axis-title-size', '20']);
assert.deepEqual(args.slice(args.indexOf('--y-axis-title-size'), args.indexOf('--y-axis-title-size') + 2), ['--y-axis-title-size', '21']);
assert.deepEqual(args.slice(args.indexOf('--plot-type'), args.indexOf('--plot-type') + 2), ['--plot-type', 'box']);
assert.deepEqual(args.slice(args.indexOf('--transform'), args.indexOf('--transform') + 2), ['--transform', 'none']);
const hiddenSummary = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'AdductType',
  xAxisTitleGap: 30, yAxisTitleGap: 40, showXAxisTitle: false, showYAxisTitle: false, xLabelRotation: '90',
});
assert.deepEqual(hiddenSummary.slice(hiddenSummary.indexOf('--x-axis-title-gap'), hiddenSummary.indexOf('--x-axis-title-gap') + 2), ['--x-axis-title-gap', '30']);
assert.deepEqual(hiddenSummary.slice(hiddenSummary.indexOf('--y-axis-title-gap'), hiddenSummary.indexOf('--y-axis-title-gap') + 2), ['--y-axis-title-gap', '40']);
assert.deepEqual(hiddenSummary.slice(hiddenSummary.indexOf('--x-label-rotation'), hiddenSummary.indexOf('--x-label-rotation') + 2), ['--x-label-rotation', '90']);
assert.ok(hiddenSummary.includes('--hide-x-axis-title'));
assert.ok(hiddenSummary.includes('--hide-y-axis-title'));
const transformed = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'CollisionEnergy', mode: 'numeric',
  transform: 'collision-energy', precursorMzColumn: 'PrecursorMZ',
  instrumentColumn: 'InstrumentType', bins: '0,10,20'
});
assert.ok(transformed.includes('--chemical-descriptor') === false);
assert.deepEqual(transformed.slice(transformed.indexOf('--transform'), transformed.indexOf('--transform') + 2), ['--transform', 'collision-energy']);
assert.ok(transformed.includes('--bins'));
const smarts = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'SMILES', mode: 'categorical',
  transform: 'smarts', smilesColumn: 'SMILES',
  smartsQueries: [{ label: 'Aromatic', smarts: '[c]' }]
});
assert.deepEqual(smarts.slice(smarts.indexOf('--transform'), smarts.indexOf('--transform') + 2), ['--transform', 'smarts']);
assert.deepEqual(JSON.parse(smarts[smarts.indexOf('--smarts-json') + 1]), [{ label: 'Aromatic', smarts: '[c]' }]);
console.log('Evaluation webview and CLI argument checks passed.');


(async () => {
  const manifest = require('../package.json');
  for (const command of ['fredis.openEvaluation', 'fredis.openColumnEvaluation', 'fredis.openGroupedEvaluation']) {
    assert(manifest.contributes.commands.some(item => item.command === command));
    assert(manifest.activationEvents.includes('onCommand:' + command));
  }
  for (const kind of [null, 'column', 'grouped']) {
    const panel = open({}, {}, () => '', kind);
    assert.equal(panel.viewType, 'fredis.evaluation');
    assert.equal(panel.title, 'FREDIS Evaluation');
    assert.deepEqual(panel.messages, []); // Wait for the webview to attach its listeners.
    for (const handler of panel.handlers) await handler({ type: 'evaluationReady' });
    assert.deepEqual(panel.messages, kind ? [{ type: 'evaluationTabSelected', kind }] : []);
    assert(panel.webview.html.includes('data-eval-tab="column"'));
    assert(panel.webview.html.includes('data-eval-tab="grouped"'));
  }
  const count = panels.length;
  assert.throws(() => open({}, {}, () => '', 'unknown'), /Unknown evaluation kind/);
  assert.equal(panels.length, count);
  console.log('Standalone evaluation commands and initial tab selection checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });

// Preparation evaluation opens first, and settings restore columns and tool rows.
{
  const {JSDOM} = require('jsdom');
  const messages=[];
  const dom=new JSDOM(page,{runScripts:'dangerously',beforeParse(window){window.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m)});}});
  const w=dom.window, d=w.document;
  assert.equal(d.getElementById('filteringEvaluation').hidden,false);
  assert.equal(d.getElementById('columnEvaluation').hidden,true);
  d.getElementById('filterAdd').click();
  assert.equal(d.querySelectorAll('[data-tool]').length,2);
  d.querySelector('[data-remove="1"]').click();
  assert.equal(d.querySelectorAll('[data-tool]').length,1);
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'filteringConfigLoaded',path:'/tmp/settings.evaltrainable.pft',config:{metadata:'/tmp/data.msds',idColumn:'SpecID',groupColumn:'Class',tools:[{name:'FREDIS',path:'/tmp/ids.txt',color:'#123456'}]}}}));
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'filteringColumns',result:{columns:['SpecID','Class'],rows:4}}}));
  assert.equal(w.evalFiltering.request().groupColumn,'Class');
  assert.equal(w.evalFiltering.request().tools[0].name,'FREDIS');
  d.getElementById('filterRun').click();
  assert.equal(messages.at(-1).type,'filteringRun');
  const beforeRestore=messages.length;
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'filteringConfigLoaded',path:'/moved/settings.evaltrainable.pft',config:{metadata:'/missing/source.msds',idColumn:'SpecID',groupColumn:'Class',resultPath:'results.json',tools:[],view:{categories:['A'],tools:['Common'],showOverall:false}}}}));
  assert.equal(messages.length,beforeRestore); // no metadata read on cached restore
  const rows=[{scope:'category',category:'A',tool:'Common',totalN:2,retainedN:1,excludedN:1,retainedPercent:50}];
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'filteringResult',restored:true,result:{rows,displayRows:rows,diagnostics:[],sourceN:2,title:'Cached',series:[{name:'Common',color:'#666666'}],png:''}}}));
  assert.equal(d.getElementById('filterPresentation').hidden,false);
  assert.equal(w.evalFiltering.request().view.showOverall,false);
  for(const id of ['filterOverall','filterVisibleTools','filterMinN','filterCategories'])assert.equal(d.getElementById(id),null);
  assert.equal(w.evalFiltering.request().view.minN,0);
  assert(d.getElementById('filterGraphTab').compareDocumentPosition(d.getElementById('filterCommonColor')) & w.Node.DOCUMENT_POSITION_FOLLOWING);
  assert(!d.getElementById('filterClassificationColumns'));
  assert.equal(d.getElementById('filterGraphPage').style.overflow,'auto');
  assert.equal(d.getElementById('filterChart').style.maxWidth,'none');
  // The list filter may show an unchecked category while a checked one is hidden.
  const graphRows=[{scope:'category',category:'A',tool:'Common',totalN:2,retainedN:1,excludedN:1,retainedPercent:50}];
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'filteringPresentation',result:{displayRows:graphRows,categoryRows:[{scope:'category',category:'A',totalN:2}],listRows:[{scope:'category',category:'B',totalN:1}],fields:[{id:'category',label:'Category',type:'text'},{id:'totalN',label:'Original N',type:'number'}],png:''}}}));
  d.getElementById('filterUnselectShown').click();
  assert.deepEqual(Array.from(w.evalFiltering.request().view.categories),['A']);
  d.getElementById('filterSelectShown').click();
  assert.deepEqual(Array.from(w.evalFiltering.request().view.categories),['A','B']);
  const check=d.querySelector('[data-category-check]');
  assert(check.checked);check.checked=false;check.dispatchEvent(new w.Event('change',{bubbles:true}));
  assert.deepEqual(Array.from(w.evalFiltering.request().view.categories),['A']);
  assert.match(d.getElementById('filterCategoryResults').textContent,/Distribution/);
  d.getElementById('filterListTab').click();
  assert.equal(d.getElementById('filterListPage').hidden,false);
  assert.equal(d.getElementById('filterGraphPage').hidden,true);
  d.getElementById('filterCommonHex').value='#abcdef';
  d.getElementById('filterCommonHex').dispatchEvent(new w.Event('input',{bubbles:true}));
  assert.equal(d.getElementById('filterCommonColor').value,'#abcdef');
  assert.equal(w.evalFiltering.request().view.orientation,'vertical');
  assert.equal(w.evalFiltering.request().view.colors.Common,'#abcdef');
  d.getElementById('filterAddSort').click();
  assert.equal(w.evalFiltering.request().view.sortRules.length,2);
  d.getElementById('filterAddFilter').click();
  assert.equal(w.evalFiltering.request().view.filters[0].field,'totalN');
  const threshold=d.querySelector('#filterFilterRules [data-key="value"]');
  threshold.value='10';threshold.dispatchEvent(new w.Event('input',{bubbles:true}));
  assert.equal(w.evalFiltering.request().view.filters[0].value,'10');
  d.getElementById('filterExportTsv').click();
  assert.equal(messages.at(-1).type,'filteringExport');
  assert.equal(messages.at(-1).format,'tsv');
  assert.deepEqual(Array.from(messages.at(-1).view.categories),['A']);
  assert.equal(messages.at(-1).request,undefined); // exports never recalculate inputs
  d.getElementById('filterExportSvg').click();
  assert.equal(messages.at(-1).format,'svg');
  d.getElementById('filterApply').click();
  assert.equal(messages.at(-1).type,'filteringView');
  d.getElementById('filterSave').click();
  assert.deepEqual(Array.from(messages.at(-1).config.view.tools),['Common']);
  d.querySelector('[data-eval-tab="column"]').click();
  assert.equal(d.getElementById('filteringEvaluation').hidden,true);
  assert.equal(d.getElementById('columnEvaluation').hidden,false);
  dom.window.close();
}

// Saving repeatedly must keep exactly one evaluation extension.
for (const [kind, suffix] of [['filtering','.evaltrainable.pft'],['column','.evalcol.pft'],['grouped','.evalgroup.pft']]) {
  const expected='/tmp/m_h_plus.msds'+suffix;
  let filename=expected;
  for(let i=0;i<5;i++) {
    filename=ensureEvaluationConfigSuffix(filename,kind);
    assert.equal(filename,expected);
  }
  assert.equal(ensureEvaluationConfigSuffix(expected+suffix+suffix,kind),expected);
  assert.equal(ensureEvaluationConfigSuffix('/tmp/result.evalcol.pft.evalgroup.pft',kind),'/tmp/result'+suffix);
}
assert.doesNotMatch(editorSource, /\[suffix\.slice\(1\)\]/);

assert.equal(evaluationConfigSuffix('filtering'), '.evaltrainable.pft');
