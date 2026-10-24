const assert = require('assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {JSDOM} = require('jsdom');
const {readRun,readCharts,readProgressMetrics,html,script} = require('../src/features/training-metrics/editor');
(async () => {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-metric-store-'));
  try {
    fs.mkdirSync(path.join(root,'metrics/scalars'),{recursive:true});
    const index={schema:'fredis.metric-index',schemaVersion:1,series:[
      {name:'metrics/train_loss',file:'metrics/scalars/loss.tsv'},
      {name:'metrics/validation_loss',file:'metrics/scalars/loss.tsv'},
      // Deliberately absent: neither listing nor loading loss may open this file.
      {name:'metrics/train_batches',file:'metrics/scalars/batches.tsv'},
      {name:'metrics/train_total_batches',file:'metrics/scalars/total-batches.tsv'},
      {name:'metrics/train_num_ion_states',file:'metrics/scalars/ion-states.tsv'},
    ]};
    const indexFile=path.join(root,'metrics/index.json');
    fs.writeFileSync(indexFile,JSON.stringify(index));
    fs.writeFileSync(path.join(root,'metrics/scalars/loss.tsv'),'series\tglobal_step\tepoch\tevent\tvalue\nmetrics/train_loss\t0\t0\tepoch_end\t3\nmetrics/train_loss\t10\t1\tepoch_end\t2\nmetrics/validation_loss\t10\t1\tepoch_end\t1\nmetrics/train_loss\t20\t2\tepoch_end\t0.9');
    fs.writeFileSync(path.join(root,'metrics/scalars/batches.tsv'),'series\tglobal_step\tepoch\tevent\tvalue\nmetrics/train_batches\t10\t1\tepoch_end\t1\n');
    fs.writeFileSync(path.join(root,'metrics/scalars/total-batches.tsv'),'series\tglobal_step\tepoch\tevent\tvalue\nmetrics/train_total_batches\t10\t1\tepoch_end\t7\n');
    let opens=[];
    const original=fs.createReadStream;
    fs.createReadStream=(filename,...args)=>{opens.push(filename);return original(filename,...args);};
    try {
      const run=await readRun(root);
      assert.equal(run.lazy,true);
      assert.equal(opens.length,0,'catalog loading must not open chart files');
      assert.ok(run.series.every(series=>series.points.length===0));
      const selected=await readCharts(root,['loss']);
      assert.deepEqual(opens,[path.join(root,'metrics/scalars/loss.tsv')]);
      assert.deepEqual(selected.series.find(s=>s.name==='metrics/train_loss').points,[[0,3],[10,2]]);
      assert.equal(selected.series.length,2);
      const progress=await readProgressMetrics(root);
      assert.deepEqual(progress.rows,[{event:'epoch_end',global_step:0,epoch:0,train_loss:3},{event:'epoch_end',global_step:10,epoch:1,train_loss:2,validation_loss:1}]);
      const batches=await readCharts(root,['batches']);assert.deepEqual(batches.series.map(s=>s.name),['metrics/train_batches']);assert.deepEqual(batches.series[0].points,[[10,7]]);
      await assert.rejects(readCharts(root,['num_ion_states']),/ENOENT/);
      index.series[0].file='../outside.tsv';fs.writeFileSync(indexFile,JSON.stringify(index));
      await assert.rejects(readRun(root),/Invalid metric catalog entry/);
      index.series[0].file='metrics/scalars/loss.tsv';fs.writeFileSync(indexFile,JSON.stringify(index));
    } finally {fs.createReadStream=original;}

    const dom=new JSDOM(html({hidden:false}),{runScripts:'outside-only'});
    const {window}=dom;
    let state={},posted=[];
    window.vscode={getState:()=>state,setState:value=>{state=value;},postMessage:message=>posted.push(message)};
    window.setInterval=()=>0;
    window.eval(script());
    const send=message=>window.dispatchEvent(new window.MessageEvent('message',{data:message}));
    const rendered=new Promise(resolve=>window.addEventListener('trainingMetricsRendered',resolve,{once:true}));
    send({type:'metricsData',request:0,data:await readRun(root)});
    await rendered;
    assert.equal(posted.length,0,'an empty selection must not request chart data');
    const document=window.document;
    document.getElementById('metricsChoice').value='loss';document.getElementById('metricsAdd').click();
    assert.equal(posted.length,1);
    assert.equal(posted[0].type,'metricsCharts');
    assert.deepEqual([...posted[0].names],['loss']);
    assert.match(document.getElementById('metricsCharts').textContent,/Loading chart/);
    const request=posted[0];
    send({type:'metricsChartsData',request:0,chartRequest:request.chartRequest,data:await readCharts(root,['loss'])});
    assert.equal(document.querySelectorAll('#metricsCharts svg[role="img"]').length,1);
    assert.doesNotMatch(document.getElementById('metricsCharts').textContent,/Loading chart/);
    const mode=document.querySelector('#metricsCharts select');mode.value='median';mode.dispatchEvent(new window.Event('change'));
    assert.equal(posted.length,1,'display changes must use cached chart data');
    // Refresh loads metadata and requests only the still-selected chart.
    document.getElementById('metricsLoad').click();
    const refresh=posted.at(-1);
    send({type:'metricsData',request:refresh.request,data:await readRun(root)});
    await new Promise(resolve=>setTimeout(resolve,0));
    const reload=posted.at(-1);assert.equal(reload.type,'metricsCharts');assert.deepEqual([...reload.names],['loss']);
    send({type:'metricsChartsData',request:0,chartRequest:request.chartRequest,data:{series:[]}});
    assert.match(document.getElementById('metricsCharts').textContent,/Loading chart/,'stale responses must be ignored');
    send({type:'metricsChartsData',request:reload.request,chartRequest:reload.chartRequest,error:'Temporary read failure'});
    assert.match(document.getElementById('metricsCharts').textContent,/Temporary read failure/);
    [...document.querySelectorAll('#metricsCharts button')].find(b=>b.textContent==='Retry').click();
    assert.equal(posted.at(-1).type,'metricsCharts');
    dom.window.close();
    const legacyFile=path.join(root,'metrics.tsv');
    fs.writeFileSync(legacyFile,'event\tepoch\tglobal_step\ttrain_loss\nepoch_end\t1\t10\t2\n');
    const originalStat=fs.statSync(legacyFile);
    index.legacySources=[{file:'metrics.tsv',size:originalStat.size,mtimeMs:originalStat.mtimeMs}];
    fs.writeFileSync(indexFile,JSON.stringify(index));
    assert.equal((await readRun(root)).lazy,true);
    fs.appendFileSync(legacyFile,'epoch_end\t2\t20\t1\n');
    const live=await readRun(root);
    assert.ok(!live.lazy,'changed legacy logs must invalidate the migrated snapshot');
    assert.deepEqual(live.series.find(s=>s.name==='metrics/train_loss').points,[[10,2],[20,1]]);
    console.log('Metric catalog, selective disk reads, lazy chart loading, refresh, retry, and stale-response checks passed.');
  } finally {fs.rmSync(root,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
