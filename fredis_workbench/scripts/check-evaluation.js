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
  ['*.evalcol.pft', '*.evalgroup.pft']);
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
