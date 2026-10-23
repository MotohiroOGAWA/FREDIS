const assert=require('assert/strict'),Module=require('module'),load=Module._load;
Module._load=function(name,parent,main){return name==='vscode'?{}:load.call(this,name,parent,main);};
const {JSDOM,VirtualConsole}=require('jsdom'),ext=require('../src/extension'),fixture=require('./fixtures');
const {plan}=require('../src/workbench/tuning');
(async()=>{
 const messages=[],errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.detail?.stack||e.message));
 const dom=new JSDOM(ext.workbenchHtml(fixture.dataConfig,{modelConfig:fixture.model,epochs:1,batchSize:4,lr:.001}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){} disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:s=>s};}});
 const w=dom.window,d=w.document,form=d.getElementById('trainingForm');
 const change=node=>node.dispatchEvent(new w.Event('change',{bubbles:true}));
 const input=(node,value)=>{node.value=value;node.dispatchEvent(new w.Event('input',{bubbles:true}));};
 const snapshot=()=>JSON.parse(JSON.stringify(w.eval('getTrainingConfig()')));
 const flush=async()=>{for(let i=0;i<4;i++)await Promise.resolve();};
 try{
  const architecture={hidden_dim:8,graph_dim:12,num_layers:1,num_heads:2,neighborhood_hops:3,neighborhood_hidden_dim:5,input_mode:'chemical',neighborhood_edge_features:true,dropout:0};
  const patterns=[0,1,2].map(id=>({pattern:{name:'reaction '+id,pattern_id:id,reactant_smarts:'C'.repeat(id+1),products:[]},architecture:{...architecture,...(id===2?{hidden_dim:24,graph_dim:32}: {})},atom_count:id+1,...(id===2?{source:'/pretrained.pt'}:{})}));
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'training/sources',requestId:form.dataset.sourcesRequest,modelConfig:{...fixture.model,fine_tuning:{frozen_pattern_ids:[2]}},patternModels:patterns}}));
  await flush();
  assert.equal(d.getElementById('patternParameterMode').value,'shared');
  assert.equal(d.querySelectorAll('[data-shared-pattern-parameter]').length,6);
  assert.equal(d.querySelectorAll('[data-pattern-parameter]').length,0);
  input(d.querySelector('[data-shared-pattern-parameter=hidden_dim]'),'16');
  let c=snapshot(),params=c.modelConfig.reaction_model_params.pattern_model_params;
  assert.equal(params[0].hidden_dim,16);assert.equal(params[1].hidden_dim,16);assert.equal(params[2].hidden_dim,24);
  for(const pid of [0,1]){
   const row=d.querySelector('[data-pattern-id="'+pid+'"]');
   assert.equal(row.querySelector('.pattern-architecture-grid'),null);
   assert.equal(row.querySelector('[data-model-architecture]'),null);
   assert(row.textContent.includes('Untrained reaction pattern'));
   assert(!row.textContent.includes('saved parameters'));
  }
  assert(d.querySelector('[data-pattern-id="2"] .pattern-architecture-grid'));
  assert(d.querySelector('[data-pattern-id="2"] [data-model-architecture]'));
  let args=ext.buildTrainingArgs(c);
  assert(args.includes('0=16'));assert(args.includes('1=16'));assert(!args.includes('2=24'));
  for(const key of ['hidden-dim','graph-dim','num-layers','num-heads','neighborhood-hops','neighborhood-hidden-dim'])assert.equal(args.filter(a=>a==='--pattern-'+key).length,2);
  d.getElementById('trainingTuningToggle').click();await flush();
  const control=d.querySelector('[data-shared-pattern-parameter=hidden_dim]').parentElement.querySelector('.tuning-control'),[check,values]=control.querySelectorAll('input');
  check.checked=true;change(check);input(values,'8, 16');await flush();
  c=snapshot();assert.deepEqual(c.tuning.parameters['patternArchitecture.shared.hidden_dim'],[8,16]);
  const trials=plan(c,ext.buildTrainingArgs);assert.equal(trials.length,2);
  for(const [i,trial]of trials.entries()){assert.equal(trial.config.modelConfig.reaction_model_params.pattern_model_params[0].hidden_dim,[8,16][i]);assert.equal(trial.config.modelConfig.reaction_model_params.pattern_model_params[1].hidden_dim,[8,16][i]);assert.equal(trial.config.modelConfig.reaction_model_params.pattern_model_params[2].hidden_dim,24);}
  assert(!d.querySelector('[data-tuning-group]'));assert(!d.body.textContent.includes('Link all patterns'));
  const mode=d.getElementById('patternParameterMode');mode.value='individual';change(mode);await flush();
  assert.equal(d.querySelectorAll('[data-shared-pattern-parameter]').length,0);
  assert.equal(d.querySelectorAll('[data-pattern-parameter]').length,12);
  assert.equal(d.querySelectorAll('[data-pattern-id="2"] input').length,0);
  assert.equal(d.querySelector('[data-pattern-id="0"] .pattern-architecture-grid'),null);
  assert.equal(snapshot().tuning.parameters['patternArchitecture.shared.hidden_dim'],undefined);
  input(d.querySelector('[data-pattern-id="0"] [data-pattern-parameter=hidden_dim]'),'12');
  c=snapshot();assert.equal(c.modelConfig.reaction_model_params.pattern_model_params[0].hidden_dim,12);assert.equal(c.modelConfig.reaction_model_params.pattern_model_params[1].hidden_dim,16);
  const individual=d.querySelector('[data-pattern-id="0"] [data-pattern-parameter=hidden_dim]').parentElement.querySelector('.tuning-control'),[individualCheck,individualValues]=individual.querySelectorAll('input');
  individualCheck.checked=true;change(individualCheck);input(individualValues,'8,12');await flush();
  const individualTrials=plan(snapshot(),ext.buildTrainingArgs);assert.equal(individualTrials.length,2);assert.equal(individualTrials[0].config.modelConfig.reaction_model_params.pattern_model_params[1].hidden_dim,16);
  const saved=ext.normalizeTrainingConfig(snapshot());assert.equal(saved.patternArchitecture.mode,'individual');
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'trainingConfig',config:saved}}));await flush();w.dispatchEvent(new w.MessageEvent('message',{data:{type:'training/sources',requestId:form.dataset.sourcesRequest,modelConfig:{...fixture.model,fine_tuning:{frozen_pattern_ids:[2]}},patternModels:patterns}}));await flush();assert.equal(d.getElementById('patternParameterMode').value,'individual');assert.equal(snapshot().modelConfig.reaction_model_params.pattern_model_params[0].hidden_dim,12);
  assert.deepEqual(errors,[]);console.log('Shared/default and individual reaction editing, CLI expansion, tuning, pretrained exclusion and reload checks passed.');
 }finally{dom.window.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
