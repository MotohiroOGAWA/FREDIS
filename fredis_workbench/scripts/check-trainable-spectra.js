(async()=>{
const assert=require('assert/strict'),{JSDOM,VirtualConsole}=require('jsdom'),Module=require('module'),load=Module._load;
Module._load=function(name,parent,main){if(name==='vscode')return {};return load.call(this,name,parent,main)};
const ext=require('../src/extension'),feature=require('../src/workbench/trainable-spectra'),fixture=require('./fixtures');
const messages=[],errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.detail?.stack||e.message));
const dom=new JSDOM(ext.workbenchHtml(fixture.dataConfig,{modelConfig:fixture.model,epochs:1,batchSize:4,lr:.001}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>({}),setState:()=>{}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){} disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:s=>s};}});
const w=dom.window,d=w.document,post=data=>w.dispatchEvent(new w.MessageEvent('message',{data}));
d.querySelector('#navigationRail [data-page=trainable]').click();assert.equal(d.getElementById('trainablePage').hidden,false);assert.equal(d.getElementById('homePage').hidden,true);
const form=d.getElementById('trainableExtractForm');form.elements.input.value='/tmp/input.msds';form.elements.outputDir.value='/tmp/fredis';form.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));const request=messages.at(-1);assert.equal(request.type,'trainable/run');assert.equal(request.mode,'extract');assert(request.config.fragmenterParams);
const args=feature.buildArgs('extract',request.config);assert(args.includes('trainable-ids'));assert(args.includes('--params-json'));assert(!args.includes('--model-architecture-json'));assert(d.getElementById('trainableStop').disabled===false);
d.getElementById('trainableCopyCommand').click();const copied=messages.at(-1);assert.equal(copied.type,'trainable/copyCommand');assert.deepEqual(copied.config,request.config);post({type:'trainable/copied',command:'python -m fredis.cli train trainable-ids'});assert.equal(d.getElementById('trainableExtractStatus').textContent,'Command copied.');
const tricky=["/tmp/O'Brien dataset.msds",JSON.stringify({smarts:'[!#1:1]-[!#1:2]',name:'$HOME `echo unsafe`'})];const command=feature.shellDisplay('python',['-c','import sys,json;print(json.dumps(sys.argv[1:]))',...tricky]);const roundTrip=require('child_process').execFileSync('sh',['-c',command],{encoding:'utf8'});assert.deepEqual(JSON.parse(roundTrip),tricky);
// Trainable owns a fragmenter copy and uses the shared file and set editors.
const mount=d.getElementById('trainableParameterEditor');
assert.deepEqual(JSON.parse(JSON.stringify(w.getTrainableFragmenterConfig().fragment_ion_tree_builder.fragment_ion_adduct_rule_set.adduct_rules.map(rule=>rule.adduct_type))),['[M+H]+']);
assert(!mount.querySelector('[data-model-block=mol_encoder_params]'));
const tolerance=mount.querySelector('[data-parameter-path="fragmenter_params.mass_tolerance"]');
tolerance.value='7ppm';tolerance.dispatchEvent(new w.Event('input',{bubbles:true}));
assert.notEqual(w.eval('getConfig().fragmenterParams.mass_tolerance'),'7ppm');
d.getElementById('trainableSaveDefaults').click();assert.equal(messages.at(-1).type,'defaults/save');assert.equal(messages.at(-1).workflow,'trainable');assert.equal(messages.at(-1).config.fragmenterParams.mass_tolerance,'7ppm');assert.deepEqual(Object.keys(messages.at(-1).config),['fragmenterParams']);
post({type:'defaults/saved',workflow:'trainable'});assert(d.getElementById('trainableDefaultsStatus').textContent.includes('Defaults saved'));
d.getElementById('trainableParameterSave').click();assert.equal(messages.at(-1).target,'trainable');assert.equal(messages.at(-1).fragmenterParams.mass_tolerance,'7ppm');
d.getElementById('trainableParameterLoad').click();assert.equal(messages.at(-1).type,'parameters/pick');assert.equal(messages.at(-1).target,'trainable');
const imported=JSON.parse(JSON.stringify(fixture.dataConfig.fragmenterParams));imported.mass_tolerance='9ppm';
post({type:'parameters/loaded',target:'trainable',fragmenterParams:imported,path:'/tmp/custom.fragmenter.pft'});
for(const [prefix,message] of [['Reaction Pattern Set','reactionPatternSet'],['Fragment Ion Adduct Rule Set','adductRuleSet']]){
 const details=[...mount.querySelectorAll('details')].find(node=>node.querySelector('summary').textContent.startsWith(prefix));details.open=true;details.dispatchEvent(new w.Event('toggle'));
 if(message==='reactionPatternSet')post({type:message,value:{reaction_pattern_set:{name:'custom',patterns:[]}},path:'/tmp/custom.reactionset.pft'});
 else post({type:message,value:{fragment_ion_adduct_rule_set:{name:'custom',hydrogen_shift_min:-2,hydrogen_shift_max:1,adduct_rules:[{name:'custom',adduct_type:'[M+Na]+',fragment_ion_adducts:['[M+Na]+']}]}},path:'/tmp/custom.adductset.pft'});
}
d.getElementById('trainableCopyCommand').click();const custom=messages.at(-1).config;
assert.equal(custom.fragmenterParams.mass_tolerance,'9ppm');assert.equal(custom.fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set.name,'custom');assert.equal(custom.fragmenterParams.fragment_ion_tree_builder.fragment_ion_adduct_rule_set.hydrogen_shift_min,-2);
const customArgs=feature.buildArgs('extract',custom);assert.deepEqual(JSON.parse(customArgs[customArgs.indexOf('--params-json')+1]),JSON.parse(JSON.stringify(custom.fragmenterParams)));
assert.notEqual(w.eval('getConfig().fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set.name'),'custom');
d.getElementById('trainableUseFragmenter').click();assert.deepEqual(JSON.parse(JSON.stringify(w.getTrainableFragmenterConfig())),JSON.parse(w.eval('JSON.stringify(getConfig().fragmenterParams)')));
post({type:'trainable/progress',completed:5,total:10,trainable:4});assert.equal(d.getElementById('trainableProgress').value,5);
post({type:'trainable/result',mode:'extract',outputDir:'/tmp/fredis',stats:{total_records:10,trainable:8,excluded:2,exclusion_reasons:{'Invalid adduct':2}}});assert(d.getElementById('trainableExtractSummary').textContent.includes('Trainable: 8'));
assert.equal(d.getElementById('trainableIntersectForm'),null);assert.equal(form.elements.timeout,undefined);assert.equal(form.elements.device,undefined);assert(!args.includes('--timeout'));assert(!args.includes('--device'));assert(args.includes('--workers'));assert.throws(()=>feature.buildArgs('intersect',{outputDir:'/tmp/new'}),/Unsupported/);
d.querySelector('#navigationRail [data-page=training]').click();assert.equal(d.getElementById('trainablePage').hidden,true);assert.deepEqual(errors,[]);const savedDom=new JSDOM(ext.workbenchHtml(fixture.dataConfig,{modelConfig:fixture.model},{},{},{fragmenterParams:custom.fragmenterParams}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(win){win.acquireVsCodeApi=()=>({postMessage(){},getState:()=>({}),setState(){}});win.HTMLElement.prototype.scrollIntoView=function(){};win.ResizeObserver=class{observe(){}disconnect(){}};win.matchMedia=()=>({matches:false,addEventListener(){}});win.CSS={escape:s=>s};}});
assert.equal(savedDom.window.getTrainableFragmenterConfig().mass_tolerance,'9ppm');assert.equal(savedDom.window.getTrainableFragmenterConfig().fragment_ion_tree_builder.fragment_ion_adduct_rule_set.adduct_rules[0].adduct_type,'[M+Na]+');
await new Promise(resolve=>setTimeout(resolve,50));savedDom.window.close();dom.window.close();console.log('Trainable Spectra: dedicated navigation, native extraction arguments, progress/results, absence of intersection/device/timeout controls passed.');

})().catch(error=>{console.error(error);process.exitCode=1;});
