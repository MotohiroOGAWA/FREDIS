const assert=require('assert/strict');
const fs=require('fs');const os=require('os');const path=require('path');
const {JSDOM}=require('jsdom');
const {html,script,representativePoints,validationResult}=require('../src/features/training-metrics/editor');
(async()=>{
  const records=Array.from({length:4},(_,i)=>({smiles:'C'.repeat(i+1)+'O',main_adduct:'[M+H]+',collision_energy:20,
    assignment_score:.1+i*.2,cosine_similarity:.9-i*.2,assignment_score_without_precursor:i===3?null:.8-i*.2,cosine_similarity_without_precursor:i===3?null:.2+i*.2,
    generated_peaks:[{mz:100,intensity:1,precursor:true},{mz:50,intensity:.5}],original_peaks:[{mz:100,intensity:.7,precursor:true},{mz:50,intensity:1}]}));
  const item=i=>({spectrum_index:i,source_spectrum_index:10+i,spectrum:records[i],value:records[i].cosine_similarity});
  const representatives={cosine_similarity:{q10:item(0),q90:item(1)},assignment_score:{q10:item(2),q90:item(0)},cosine_similarity_without_precursor:{median:item(3)},assignment_score_without_precursor:{q25:item(1)}};
  const points=representativePoints(representatives);
  assert.equal(points.length,4);assert.equal(points.find(p=>p.key==='10').selections.length,2);
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-validation-scatter-'));
  let dom;
  try{
    fs.mkdirSync(path.join(root,'spectrum_validation'));
    for(const epoch of [0,2]) {
      fs.writeFileSync(path.join(root,`spectrum_validation/epoch_${epoch}.json`),'{}');
      fs.writeFileSync(path.join(root,`spectrum_validation/epoch_${epoch}.summary.json`),JSON.stringify({summary:{samples:100},representatives}));
      fs.writeFileSync(path.join(root,`spectrum_validation/epoch_${epoch}.msds`),'dataset placeholder');
    }
    const manifest={latestValidation:'spectrum_validation/epoch_2.json'};
    const validation=await validationResult(root,manifest);
    assert.deepEqual(validation.epochs,[0,2]);assert.equal(validation.epoch,2);assert.equal(validation.msdsFile,'spectrum_validation/epoch_2.msds');
    assert.equal((await validationResult(root,manifest,0)).file,'spectrum_validation/epoch_0.json');
    await assert.rejects(validationResult(root,manifest,1),/No validation artifact/);
    await assert.rejects(validationResult(root,manifest,-1),/No validation artifact/);
    dom=new JSDOM(html({hidden:false}),{runScripts:'outside-only'});
    const {window}=dom,document=window.document;let state={},posted=[];
    window.vscode={getState:()=>state,setState:value=>{state=value;},postMessage:message=>posted.push(message)};window.setInterval=()=>0;
    Object.defineProperty(window.HTMLElement.prototype,'clientWidth',{get(){return ['representativeScatter','representativeSpectra'].includes(this.id)?1200:0;}});
    window.eval(script());
    const send=data=>window.dispatchEvent(new window.MessageEvent('message',{data}));
    const rendered=new Promise(resolve=>window.addEventListener('trainingMetricsRendered',resolve,{once:true}));
    send({type:'metricsData',request:0,data:{directory:root,report:manifest,series:[],validation}});await rendered;
    const circles=()=>[...document.querySelectorAll('#representativeScatter circle')];
    const before=circles().map(node=>[node.dataset.spectrumKey,node.getAttribute('cx'),node.getAttribute('cy')]);
    assert.equal(before.length,4);
    const initialScatter=document.querySelector('#representativeScatter > svg');
    assert.equal(initialScatter.style.height,'300px');assert.equal(initialScatter.style.maxWidth,'');
    assert.equal(initialScatter.getAttribute('viewBox'),'0 0 1200 300');
    assert.equal(document.querySelector('#representativeSpectra svg').style.height,'240px');
    assert.ok(circles().every(node=>Number(node.getAttribute('r'))<=4));assert.equal(document.getElementById('representativeEpoch').value,'2');
    circles().find(node=>node.dataset.spectrumKey==='11').dispatchEvent(new window.Event('click'));
    assert.match(document.getElementById('representativeSpectra').textContent,/SMILES: CCO/);
    const lineCount=document.querySelectorAll('#representativeSpectra svg line').length;
    document.getElementById('representativePrecursor').checked=false;document.getElementById('representativePrecursor').dispatchEvent(new window.Event('change'));
    const after=circles().map(node=>[node.dataset.spectrumKey,node.getAttribute('cx'),node.getAttribute('cy')]);
    assert.deepEqual(after.map(p=>p[0]),before.map(p=>p[0]),'score mode must retain every representative');
    assert.notDeepEqual(after,before,'coordinates must change');
    assert.equal(document.getElementById('representativePoint').value,'11','selection must stay stable');
    assert.equal(document.querySelectorAll('#representativeSpectra svg line').length,lineCount,'score mode must not hide precursor peaks');
    assert.equal(posted.length,0,'score mode must not load a different subset');
    assert.equal(circles().find(node=>node.dataset.spectrumKey==='13').getAttribute('fill'),'none','missing scores must keep a selectable point');
    const stableScatter=document.querySelector('#representativeScatter > svg'),stableSpectrum=document.querySelector('#representativeSpectra article');
    const number=document.getElementById('representativeEpoch');number.value=0;number.dispatchEvent(new window.Event('change'));
    assert.strictEqual(document.querySelector('#representativeScatter > svg'),stableScatter,'epoch loading must preserve plot geometry');
    assert.strictEqual(document.querySelector('#representativeSpectra article'),stableSpectrum,'epoch loading must not collapse the spectrum');
    assert.equal(document.getElementById('representativeScatter').getAttribute('aria-busy'),'true');
    const first=posted.at(-1);assert.equal(first.type,'metricsValidation');assert.equal(first.epoch,0);
    const slider=document.getElementById('representativeEpochSlider');slider.value=2;slider.dispatchEvent(new window.Event('change'));
    send({type:'metricsValidationData',request:0,validationRequest:first.validationRequest,data:{...validation,epoch:0,file:'wrong.json'}});
    assert.doesNotMatch(document.getElementById('representativeStatus').textContent,/wrong/,'stale epoch replies must be ignored');
    slider.value=0;slider.dispatchEvent(new window.Event('change'));const current=posted.at(-1);
    const zero=await validationResult(root,manifest,0);
    send({type:'metricsValidationData',request:0,validationRequest:current.validationRequest,data:zero});
    assert.equal(number.value,'0');assert.equal(slider.value,'0');assert.match(document.getElementById('representativeStatus').textContent,/epoch_0/);
    assert.equal(circles().length,4);
    assert.equal(document.querySelector('#representativeScatter > svg').style.height,'300px');
    assert.equal(document.querySelector('#representativeSpectra svg').style.height,'240px');
    assert.equal(document.getElementById('representativeScatter').getAttribute('aria-busy'),'false');
    console.log('Validation scatter: union, score-only toggle, click selection, SMILES, epoch controls, and stale replies passed.');
  }finally{dom?.window.close();fs.rmSync(root,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
