const assert = require('assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { readRun, groupSeries, metricInfo, html, script } = require('../src/features/training-metrics/editor');
(async () => {
  new Function('vscode', script());
  assert.match(html(), /Each metric appears once/);
  assert.match(script(), /Exclude precursor/);
  assert.match(script(), /#e76f51/);
  assert.match(script(), /#3a86ff/);
  assert.equal(metricInfo('num_labelled_edges').group, 'Training supervision');
  assert.equal(metricInfo('num_positive_edges').group, 'Training supervision');
  assert.equal(metricInfo('retained_probability').group, 'Probability flow');
  assert.equal(metricInfo('num_ion_states').group, 'Ion states & transitions');
  assert.equal(metricInfo('mean_path_depth').group, 'Fragment generation');
  assert.equal(metricInfo('transition_cross_entropy').group, 'Loss');
  assert.match(metricInfo('num_labelled_edges').description, /labelled reaction edges/);
  assert.match(metricInfo('cosine_similarity', 'depth').description, /rather than cumulatively/);
  assert.match(script(), /About this metric/);
  assert.ok(script().indexOf("'Loss','Training supervision'") >= 0);
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
  const facets = groupSeries([
    ...['high', 'low', 'mid'].map(category => ({name: `distributions/validation/cosine_similarity@collision_energy:${category}_mean`, points:[[10,.2],[20,.4]]})),
    ...['[M+H]+', '[M+Na]+'].map(category => ({name: `distributions/validation/cosine_similarity@main_adduct:${category}_mean`, points:[[10,.2],[20,.4]]})),
    ...[0,1,2,3].map(category => ({name: `distributions/validation/assignment_score@depth:${category}_mean`, points:[[10,.2],[20,.4]]})),
    ...[20,5,1,10].map(n => ({name: `metrics/val_peak_precision_top${n}`, points:[[10,.2]]})),
  ], 3);
  assert.deepEqual(facets.find(c => c.dimension === 'collision_energy').variants.withPrecursor.map(l => l.category), ['low','mid','high']);
  assert.equal(facets.find(c => c.dimension === 'main_adduct').variants.withPrecursor.length, 2);
  assert.ok(facets.find(c => c.dimension === 'main_adduct').variants.withPrecursor.every(l => l.stats.mean.length === 2));
  assert.equal(facets.find(c => c.dimension === 'depth').variants.withPrecursor.length, 4);
  assert.deepEqual(facets.filter(c => c.metric.startsWith('peak_precision')).map(c => c.metric), ['peak_precision_top01','peak_precision_top05','peak_precision_top10','peak_precision_top20']);
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
    assert.deepEqual(result.series.find(s => s.name === 'metrics/train_window_loss').points, [[10, 3]]);
    assert.deepEqual(result.series.find(s => s.name === 'distributions/train/loss_q1').points, [[10, .5]]);
    assert.equal(result.series.length, 4);
    await fs.appendFile(path.join(dir, 'metrics.tsv'), '\nepoch_end\t1\t100\t1.5\t1\ntrain_step\t2\t110\t0.9\t\n');
    const updated = await readRun(dir);
    assert.deepEqual(updated.series.find(s => s.name === 'metrics/train_loss').points, [[10, 3], [100, 1.5]]);
    assert.deepEqual(updated.series.find(s => s.name === 'metrics/train_window_loss').points, [[10, 3], [110, .9]]);

    // A training report is recognized by actually being a file with the
    // right schema, not by its extension -- so any filename works.
    const reportPath = path.join(dir, 'whatever_i_want.txt');
    await fs.writeFile(reportPath, JSON.stringify({ schema: 'fredis.training-report', status: 'completed' }));
    const withReport = await readRun(reportPath);
    assert.equal(withReport.report.schema, 'fredis.training-report');
    await fs.rm(reportPath);
    await fs.writeFile(reportPath, 'not json');
    await assert.rejects(readRun(reportPath), /not a training report/);

    // Large validation reports must not be read as one V8 string. Include
    // escaped delimiters and Unicode across stream boundaries and both key orders.
    const rawFs = require('fs');
    const {readValidation} = require('../src/features/training-metrics/validation-reader');
    const validationFile = path.join(dir, 'epoch_0.json');
    const records = [
      {name:'first', generated:[{mz:10,intensity:.1}]},
      {name:'skip', padding:'x'.repeat(180000)},
      {name:'日本語 "quoted" \u007b brackets } [', generated:[{mz:20,intensity:.9}]}
    ];
    const representatives = {cosine_similarity:{q10:{spectrum_index:0,value:.1},q90:{spectrum_index:2,value:.9}}};
    const summary = {samples:3};
    const originalReadStream = rawFs.createReadStream;
    rawFs.createReadStream = (file, options) => {
      assert.notEqual(options?.encoding, 'buffer');
      return originalReadStream(file, {...options, highWaterMark:1021});
    };
    try {
      await fs.writeFile(validationFile, JSON.stringify({summary,representatives,spectra:records}));
      const selected = await readValidation(validationFile);
      assert.deepEqual(selected.summary, summary);
      assert.deepEqual(selected.representatives.cosine_similarity.q90.spectrum, records[2]);
      assert(!JSON.stringify(selected).includes('padding'));
      rawFs.createReadStream = () => {throw new Error('unchanged validation report reread');};
      assert.strictEqual(await readValidation(validationFile), selected);
      rawFs.createReadStream = originalReadStream;
      records[2].name = 'updated';
      await fs.writeFile(validationFile, JSON.stringify({spectra:records,representatives,summary}));
      const reordered = await readValidation(validationFile);
      assert.equal(reordered.representatives.cosine_similarity.q90.spectrum.name, 'updated');
      const compact = {summary, representatives:{cosine_similarity:{q10:{spectrum_index:0,value:.1,spectrum:records[0]}}}};
      const compactFile = validationFile.replace('.json','.summary.json');
      await fs.writeFile(compactFile, JSON.stringify(compact));
      const future = new Date(Date.now()+2000); await fs.utimes(compactFile,future,future);
      assert.deepEqual(await readValidation(validationFile), compact);
      await fs.rm(compactFile);
      await fs.writeFile(validationFile, '{"spectra":[{"unfinished":');
      await assert.rejects(readValidation(validationFile), /incomplete/);
    } finally { rawFs.createReadStream = originalReadStream; }

    // Read old totals as per-loop counts without changing the saved run, and
    // derive additive loss components from historical means and configured weights.
    await fs.writeFile(path.join(dir, 'metrics.tsv'),
      'event\tepoch\tglobal_step\ttrain_samples\ttrain_batches\ttrain_total_samples\ttrain_total_batches\ttrain_intensity_loss\ttrain_spectrum_loss_with_precursor\ttrain_spectrum_loss_without_precursor\ttrain_transition_cross_entropy\ttrain_transition_cross_entropy_without_precursor\ttrain_cross_entropy_loss\n' +
      'epoch_end\t1\t10\t100\t5\t\t\t10\t6\t4\t2\t1\t\n' +
      'epoch_end\t2\t20\t20\t1\t100\t5\t8\t5\t3\t1.5\t1\t5\n');
    await fs.writeFile(path.join(dir, 'fragment-tree.training-config.pft'), JSON.stringify({intensityLossWeight:2}));
    const legacy = await readRun(dir);
    const points = name => legacy.series.find(s=>s.name === 'metrics/train_' + name).points;
    assert.deepEqual(points('samples'), [[10,20],[20,20]]);
    assert.deepEqual(points('batches'), [[10,5],[20,5]]);
    assert.deepEqual(points('total_samples'), [[10,100],[20,100]]);
    assert.deepEqual(points('total_batches'), [[10,5],[20,5]]);
    assert.deepEqual(points('spectrum_loss'), [[10,10],[20,8]]);
    assert.deepEqual(points('cross_entropy_loss'), [[10,6],[20,5]]);
    assert.deepEqual(points('cosine_loss'), [[10,4],[20,3]]);
    console.log('Training metrics parsing and webview syntax checks passed.');
  } finally { await fs.rm(dir, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
