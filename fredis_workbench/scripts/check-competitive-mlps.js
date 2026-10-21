const assert=require('assert/strict'),Module=require('module'),load=Module._load;
Module._load=function(name,parent,main){return name==='vscode'?{}:load.call(this,name,parent,main);};
const {JSDOM,VirtualConsole}=require('jsdom'),ext=require('../src/extension'),fixtures=require('./fixtures');
const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
const model=JSON.parse(JSON.stringify(fixtures.model));model.spectrum_model_params={};
const dom=new JSDOM(ext.workbenchHtml(fixtures.dataConfig,{modelConfig:model,epochs:1,batchSize:4,lr:.001}),{runScripts:'dangerously',virtualConsole:vc,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage(){},getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){} disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:s=>s};}});
const w=dom.window,d=w.document;
const input=key=>d.querySelector('[data-parameter-path="spectrum_model_params.'+key+'"]');
for(const [key,value]of Object.entries({ce_fc_dims:'24, 12',fragment_fc_dims:'48, 24',ce_scale:'50',ce_center:'10',adduct_embedding_dim:'11',fragment_dropout:'.3',state_fc_dims:'32, 16',state_dropout:'.2',collision_energy_dim:'7'})){assert(input(key),key);input(key).value=value;input(key).dispatchEvent(new w.Event('input',{bubbles:true}));}
assert(!input('propagation_steps'));assert(!input('ce_dropout'));assert(input('state_dropout'));
for(const key of ['adduct_embedding_dim','collision_energy_dim','ce_fc_dims']){const details=input(key).closest('details');assert(details&&!details.open,key+' must start collapsed');}
const config=w.eval('getTrainingConfig()'),s=config.modelConfig.spectrum_model_params;
assert.equal(JSON.stringify(s.ce_fc_dims),'[24,12]');assert.equal(JSON.stringify(s.fragment_fc_dims),'[48,24]');assert.equal(s.ce_scale,50);assert.equal(s.ce_center,10);assert.equal(s.collision_energy_dim,7);
const args=ext.buildTrainingArgs(config);assert(!args.includes('--ce-dropout'));assert.equal(s.adduct_embedding_dim,11);
for(const [flag,value]of Object.entries({'adduct-embedding-dim':'11','ce-fc-dims':'24,12','fragment-fc-dims':'48,24','state-fc-dims':'32,16','state-dropout':'0.2','ce-scale':'50','ce-center':'10'}))assert.equal(args[args.indexOf('--'+flag)+1],value);
input('fragment_fc_dims').value='3, 0';input('fragment_fc_dims').dispatchEvent(new w.Event('input',{bubbles:true}));assert(input('fragment_fc_dims').validationMessage);
input('fragment_fc_dims').value='';input('fragment_fc_dims').dispatchEvent(new w.Event('input',{bubbles:true}));assert.equal(JSON.stringify(w.eval('getTrainingConfig()').modelConfig.spectrum_model_params.fragment_fc_dims),'[]');
setImmediate(()=>{assert.deepEqual(errors,[]);dom.window.close();});console.log('Competitive spectrum MLP editing, validation, snapshots and CLI forwarding passed.');
