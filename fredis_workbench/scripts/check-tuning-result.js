const assert=require('assert/strict'),{JSDOM,VirtualConsole}=require('jsdom'),{html}=require('../src/features/tuning-result/view');
const messages=[],errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
const report={version:1,status:'completed',totalTrials:3,trials:[
 {trial:1,status:'completed',exitCode:0,outputDir:'trial-001',config:{name:'<script>unsafe</script>'},validationHistory:[
  {epoch:1,validation_loss:0.1,validation_cosine_similarity_mean:0.4,validation_cosine_similarity_q25:0.8,validation_cosine_similarity_q75:0.9,validation_cosine_similarity_median:0.85},
  {epoch:2,validation_loss:0.8,validation_cosine_similarity_mean:0.5,validation_cosine_similarity_q25:0.2,validation_cosine_similarity_q75:0.7,validation_cosine_similarity_median:0.5}]},
 {trial:2,status:'completed',exitCode:0,outputDir:'trial-002',config:{},validationHistory:[
  {epoch:1,validation_loss:0.2,validation_cosine_similarity_mean:0.6,validation_cosine_similarity_q25:0.3,validation_cosine_similarity_q75:0.8,validation_cosine_similarity_median:0.55},
  {epoch:2,validation_loss:0.3,validation_cosine_similarity_mean:0.9,validation_cosine_similarity_q25:0.7,validation_cosine_similarity_q75:0.95,validation_cosine_similarity_median:0.9}]},
 {trial:3,status:'failed',exitCode:1,outputDir:'trial-003',config:{},validationHistory:[{validation_loss:0.01,validation_cosine_similarity_mean:1}]}]};
report.trials[0].scoreSummaries={validation_cosine_similarity_q25:{max:0.8,last:0.2}};
const dom=new JSDOM(html(report),{runScripts:'dangerously',virtualConsole:vc,beforeParse(w){let state={};w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>state,setState:s=>{state=s;}});}}),w=dom.window,d=w.document;
const change=(id,value)=>{d.getElementById(id).value=value;d.getElementById(id).dispatchEvent(new w.Event('change'));};
assert.equal(d.getElementById('metric').value,'validation_cosine_similarity_mean');
assert(d.getElementById('best').textContent.includes('trial 2'));assert.equal(d.querySelectorAll('.point').length,2);assert.equal(d.querySelectorAll('#trials tr').length,3);
assert.equal(d.getElementById('direction'),null);assert(d.querySelector('#aggregation [value=min]').hidden);
const headers=d.getElementById('trialHeaders').textContent;for(const name of ['Q1','Q3','median','mean'])for(const stat of ['max','last'])assert(headers.includes(name+' · '+stat));
assert(headers.includes('loss · min'));assert.equal(d.querySelectorAll('#trials tr')[2].children[3].textContent,'—');
change('metric','validation_cosine_similarity_q25');assert(d.getElementById('best').textContent.includes('trial 1'));
change('aggregation','last');assert(d.getElementById('best').textContent.includes('trial 2'));
change('metric','validation_cosine_similarity_q75');change('aggregation','max');assert(d.getElementById('best').textContent.includes('trial 2'));
change('metric','validation_loss');change('aggregation','min');assert(d.getElementById('best').textContent.includes('trial 1'));
change('aggregation','last');assert(d.getElementById('best').textContent.includes('trial 2'));
change('order','score');assert.equal(d.querySelector('#trials tr td').textContent,'2');
change('metric','validation_cosine_similarity_mean');change('aggregation','min');assert.equal(d.getElementById('aggregation').value,'max');assert(d.getElementById('best').textContent.includes('trial 2'));
d.querySelector('.point').dispatchEvent(new w.MouseEvent('click'));assert.equal(d.getElementById('detail').hidden,false);d.getElementById('openTrial').click();assert.equal(messages.at(-1).path,'trial-002/result.training.pft');
d.getElementById('scoresExport').click();assert.equal(messages.at(-1).path,'tuning-scores.tsv');
d.getElementById('refresh').click();assert.equal(messages.at(-1).type,'refresh');assert.deepEqual(errors,[]);dom.window.close();
console.log('Tuning cosine quartiles, all-score table, max/last switching, loss ranking and report links passed.');
