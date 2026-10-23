const assert = require('assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { readRun, groupSeries, html, script } = require('../src/features/training-metrics/editor');
(async () => {
  new Function('vscode', script());
  assert.match(html(), /Each metric appears once/);
  assert.match(script(), /Exclude precursor/);
  assert.match(script(), /#e76f51/);
  assert.match(script(), /#3a86ff/);
  const fixture = [];
  for (const split of ['train', 'train_window', 'validation']) {
    for (const adduct of ['[M+H]+', '[M+Na]+']) {
      for (const stat of ['min', 'q1', 'median', 'q3', 'max', 'mean']) {
        fixture.push({ name: `distributions/${split}/peak_recall@by_adduct:${adduct}_${stat}`, points: [[10, .5]] });
      }
    }
  }
  fixture.push({ name: 'distributions/validation/peak_recall@by_ce_range:min-to-q1_mean', points: [[20, .6]] });
  fixture.push({ name: 'distributions/validation/peak_recall_without_precursor@by_ce_range:min-to-q1_median', points: [[20, .55]] });
  fixture.push({ name: 'distributions/train/rollout_step_0/edge_selection/depth_1/recall_mean', points: [[10, .7]] });
  fixture.push({ name: 'distributions/train/recall_depth_10_mean', points: [[10, .1]] });
  fixture.push({ name: 'metrics/train_loss', points: [[10, 3]] }, { name: 'metrics/val_loss', points: [[20, 2]] });
  const grouped = groupSeries(fixture, 3);
  const chart = grouped.find(c => c.name === 'peak_recall @by_adduct');
  assert.equal(chart.variants.withPrecursor.length, 6);
  assert.deepEqual(Object.keys(chart.variants.withPrecursor[0].stats).sort(), ['mean', 'median', 'q10', 'q25', 'q75', 'q90']);
  const ce = grouped.find(c => c.name === 'peak_recall @by_ce_range');
  assert.equal(ce.variants.withPrecursor[0].category, 'min-to-q1');
  assert.equal(ce.variants.withoutPrecursor[0].stats.median[0][1], .55);
  assert.ok(grouped.some(c => c.name === 'rollout_step_0/edge_selection/depth_1/recall'));
  assert.ok(!grouped.some(c => c.name === 'recall_depth_10'));
  assert.deepEqual(grouped.find(c => c.name === 'loss').variants.withPrecursor.map(l => l.split), ['train', 'validation']);
  assert.equal(grouped.flatMap(c => c.sources).length, fixture.length - 1);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'fredis-metrics-'));
  try {
    await assert.rejects(readRun(dir), /No metrics\.tsv or metric_distributions\.tsv found/);
    await fs.writeFile(path.join(dir, 'metrics.tsv'), 'event\tepoch\tglobal_step\ttrain_loss\tval_loss\ntrain_step\t1\t10\t3\tnan\nvalidation\t1\t10\t3\t2\ntrain_step\t1\t20\tInfinity\t\ntrain_step\t1\t30');
    await fs.writeFile(path.join(dir, 'metric_distributions.tsv'), 'event\tepoch\tglobal_step\tsplit\tmetric\tvalue\r\ntrain_step\t1\t10\ttrain\tloss_q1\t0.5\r\n');
    await fs.writeFile(path.join(dir, 'fragment-tree.training-config.pft'), JSON.stringify({modelConfig:{fragmenter_params:{fragment_ion_tree_builder:{max_reaction_count:3}}}}));
    const result = await readRun(dir);
    assert.equal(result.maxDepth, 3);
    assert.deepEqual(result.series.find(s => s.name === 'metrics/val_loss').points, [[10, 2]]);
    assert.deepEqual(result.series.find(s => s.name === 'metrics/train_loss').points, [[10, 3]]);
    assert.deepEqual(result.series.find(s => s.name === 'distributions/train/loss_q1').points, [[10, .5]]);
    assert.equal(result.series.length, 3);

    // A training report is recognized by actually being a file with the
    // right schema, not by its extension -- so any filename works.
    const reportPath = path.join(dir, 'whatever_i_want.txt');
    await fs.writeFile(reportPath, JSON.stringify({ schema: 'fredis.training-report', status: 'completed' }));
    const withReport = await readRun(reportPath);
    assert.equal(withReport.report.schema, 'fredis.training-report');
    await fs.rm(reportPath);
    await fs.writeFile(reportPath, 'not json');
    await assert.rejects(readRun(reportPath), /not a training report/);

    console.log('Training metrics parsing and webview syntax checks passed.');
  } finally { await fs.rm(dir, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
