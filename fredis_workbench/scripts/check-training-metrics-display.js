const assert = require('assert/strict');
const {JSDOM} = require('jsdom');
const {html, script, metricOrder, groupSeries, metricInfo} = require('../src/features/training-metrics/editor');
(async () => {
  const losses = ['spectrum_loss_without_precursor','branch_negative_loss','loss','branch_loss','branch_positive_loss','spectrum_loss_with_precursor','spectrum_loss','cross_entropy_loss','cross_entropy_loss_with_precursor','cross_entropy_loss_without_precursor','cosine_loss','cosine_loss_with_precursor','cosine_loss_without_precursor'];
  const series = losses.map(metric => ({name:'metrics/train_' + metric,points:[[1,2],[2,1]]}));
  for(const dimension of ['depth','collision_energy','main_adduct']) {
    series.push({name:`distributions/validation/cosine_similarity@${dimension}:${dimension === 'depth' ? '1' : 'low'}_mean`,points:[[1,.5]]});
  }
  series.push({name:'metrics/cuda_peak_memory_mb',points:[[1,30720],[2,20480]]});
  series.push({name:'metrics/train_samples',points:[[1,8]]});
  series.push({name:'metrics/train_total_samples',points:[[1,80]]});
  const charts = groupSeries(series);
  assert.deepEqual(charts.filter(c => metricInfo(c.metric).group === 'Loss').sort(metricOrder).slice(0,4).map(c=>c.metric), ['loss','branch_loss','branch_positive_loss','branch_negative_loss']);
  assert.deepEqual(charts.filter(c=>c.dimension).sort(metricOrder).map(c=>c.dimension), ['main_adduct','collision_energy','depth']);
  assert.equal(metricInfo('total_samples').group, 'Dataset');
  assert.equal(metricInfo('samples').group, 'Throughput');
  const dom = new JSDOM(html({hidden:false}), {runScripts:'outside-only'});
  const {window} = dom;
  let state = {metricsSelected:charts.map(c=>c.name)};
  window.vscode = {getState:()=>state,setState:value=>{state=value;},postMessage:()=>{}};
  // Disable the periodic refresh timer in this isolated rendering test.
  window.setInterval = () => 0;
  window.eval(script());
  const rendered = new Promise(resolve=>window.addEventListener('trainingMetricsRendered',resolve,{once:true}));
  window.dispatchEvent(new window.MessageEvent('message',{data:{type:'metricsData',request:0,data:{directory:'/test',series}}}));
  await rendered;
  const document = window.document;
  const cards = () => [...document.querySelectorAll('#metricsCharts > section')];
  const findCard = metric => cards().find(card => card.querySelector('strong').textContent === metric);
  const orderedLosses = cards().map(c=>c.querySelector('strong').textContent).filter(name=>losses.includes(name));
  assert.deepEqual(orderedLosses.slice(0,4), ['loss','branch_loss','branch_positive_loss','branch_negative_loss']);
  for (const metric of losses) {
    const card = findCard(metric);
    assert.ok(card, metric + ' must have its own chart');
    assert.ok(!card.textContent.includes('Exclude precursor'), metric + ' must not have a precursor toggle');
    assert.ok(card.querySelector('details').textContent.includes('About this metric'));
  }
  let memory = cards().find(c=>c.querySelector('strong').textContent.includes('cuda_peak_memory'));
  const select = [...memory.querySelectorAll('select')].find(s=>s.value==='GB');
  assert.ok(select, '30 GB must default to GB');
  assert.match(memory.textContent, /30\.00 GB/);
  select.value='MB';select.dispatchEvent(new window.Event('change'));
  memory = cards().find(c=>c.querySelector('strong').textContent.includes('cuda_peak_memory'));
  assert.match(memory.querySelector('strong').textContent, /\(MB\)/);
  assert.match(memory.textContent, /3\.072e\+4 MB/);
  assert.equal(state.metricsCardSettings.cuda_peak_memory_mb.memoryUnit, 'MB');
  dom.window.close();
  console.log('Training metric ordering, separate loss charts, help, and memory unit rendering passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
