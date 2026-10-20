const assert=require('assert/strict');
const fs=require('fs');
const path=require('path');
const os=require('os');
const Module=require('module');
const load=Module._load;
const vscodeMock={window:{},Uri:{file:fsPath=>({fsPath})}};
Module._load=function(name,parent,main){if(name==='vscode')return vscodeMock;return load.call(this,name,parent,main);};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.hidden=false;}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;}
 setAttribute(key,value){this[key]=value;}
 setCustomValidity(message){this.validationMessage=message;}
 querySelectorAll(){return walk(this).filter(node=>'advanced' in node.dataset);}
}
function walk(node){return [node,...(node.children||[]).filter(child=>child&&typeof child==='object').flatMap(walk)];}
global.document={createElement:tag=>new Element(tag)};
const {createParameterEditor,importHtml}=require('../src/workbench/parameter-editor');
const service=require('../src/workbench/parameter-service');
const {buildArgs,buildTrainingArgs}=require('../src/extension');
const app=path.resolve(__dirname,'../..'),preset=service.chemistryDefaults(app);
(async()=>{
 const root=new Element('div'),editor=createParameterEditor(root,preset,'data');
 assert(!walk(root).some(node=>node.dataset.parameterPath?.includes('hidden_dim')));assert(!walk(root).some(node=>node.dataset.parameterPath?.includes('adduct_type_strs')));assert.equal(walk(root).filter(node=>node.dataset.symbol).length,118);
 const hydrogen=walk(root).find(node=>node.dataset.symbol==='H');hydrogen.onclick();assert(editor.getValue().symbols.includes('H'));
 assert.equal(editor.getValue().max_unique_fragment_smiles,-1);assert.equal(editor.getValue().max_reaction_combinations,-1);
 const maxUnique=walk(root).find(node=>node.dataset.parameterPath==='max_unique_fragment_smiles');maxUnique.value='500';maxUnique.oninput();assert.equal(editor.getValue().max_unique_fragment_smiles,500);
 const maxCombinations=walk(root).find(node=>node.dataset.parameterPath==='max_reaction_combinations');maxCombinations.value='1000';maxCombinations.oninput();assert.equal(editor.getValue().max_reaction_combinations,1000);
 const find=predicate=>walk(root).find(predicate);
 const mass=find(node=>node.dataset.parameterPath==='fragmenter_params.mass_tolerance');mass.value='0.03Da';mass.oninput();assert.equal(editor.getValue().fragmenter_params.mass_tolerance,'0.03Da');assert.equal(preset.fragmenter_params.mass_tolerance,'0.01Da,10ppm');
 const basic=find(node=>node.dataset.parameterPath==='fragmenter_params.fragment_ion_tree_builder.max_reaction_count');basic.value='';basic.oninput();assert(basic.validationMessage);assert.equal(editor.getValue().fragmenter_params.fragment_ion_tree_builder.max_reaction_count,3);
 assert(!find(node=>node.dataset.advanced==='fragmenter_params'));assert(!find(node=>node.tag==='button'&&node.textContent==='Advanced Fragmenter Params'));
 const fragmenterHtml=importHtml('data');assert(fragmenterHtml.includes('dataParameterLoad'));assert(fragmenterHtml.includes('dataParameterSave'));assert(!fragmenterHtml.includes('Import Parameters'));assert(!fragmenterHtml.includes('model JSON'));assert(!fragmenterHtml.includes('dataParameterFile'));
 find(node=>node.tag==='button'&&node.textContent==='Add Adduct Rules').onclick();assert.equal(editor.getValue().fragmenter_params.fragment_ion_tree_builder.fragment_ion_adduct_rule_set.adduct_rules.length,5);
 const current=editor.getValue(),args=buildArgs({input:'input.msds',outputDir:'output',fragmenterParams:current.fragmenter_params});assert(!args.includes('--params'));assert.deepEqual(JSON.parse(args[args.indexOf('--params-json')+1]),current.fragmenter_params);
 const dataArgs=buildArgs({input:'input.msds',outputDir:'out',fragmenterParams:current.fragmenter_params,symbols:current.symbols,maxUniqueFragmentSmiles:500,maxReactionCombinations:1000});assert(dataArgs.includes('--symbols-json'));assert.equal(dataArgs[dataArgs.indexOf('--max-unique-fragment-smiles')+1],'500');assert.equal(dataArgs[dataArgs.indexOf('--max-reaction-combinations')+1],'1000');assert(!dataArgs.includes('--max-node')&&!dataArgs.includes('--max-edge'));
 const trainingArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:current});assert(!trainingArgs.includes('--params'));assert(!trainingArgs.includes('--params-json'));
 // Branch dropout remains linked to the shared setting; the new State Head
 // dropout is independently editable in the spectrum model.
 const dropoutConfigRoot=new Element('div');
 createParameterEditor(dropoutConfigRoot,{reaction_model_params:{state_dropout:0.3},spectrum_model_params:{}},'training');
 assert(!walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='reaction_model_params.state_dropout'));
 assert(walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='spectrum_model_params.state_dropout'));
 assert(walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='spectrum_model_params.fragment_dropout'));
 // Generation settings are their own model_config section, filled in with
 // defaults immediately -- no "Override" button to click first.
 const ionRoot=new Element('div'),ionEditor=createParameterEditor(ionRoot,{spectrum_model_params:{}},'training');
 // Descriptors are chosen only on the post-branch intensity head.
 assert.equal(ionEditor.getValue().mol_encoder_params.external_feature_dim,0);assert.deepEqual(ionEditor.getValue().spectrum_model_params.descriptors,[]);
 const descriptorBox=name=>walk(ionRoot).find(node=>node.dataset.parameterPath==='spectrum_model_params.descriptors.'+name);
 assert.equal(descriptorBox('FormalCharge').type,'checkbox');assert(!descriptorBox('FormalCharge').checked);assert(!descriptorBox('TPSA').checked);
 descriptorBox('TPSA').checked=true;descriptorBox('TPSA').onchange();assert.deepEqual(ionEditor.getValue().spectrum_model_params.descriptors,['TPSA']);
 descriptorBox('FormalCharge').checked=false;descriptorBox('FormalCharge').onchange();assert.deepEqual(ionEditor.getValue().spectrum_model_params.descriptors,['TPSA']);
 const descriptorArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:ionEditor.getValue()});
 assert.deepEqual(descriptorArgs.slice(descriptorArgs.indexOf('--intensity-descriptors'),descriptorArgs.indexOf('--intensity-descriptors')+2),['--intensity-descriptors','TPSA']);
 descriptorBox('TPSA').checked=false;descriptorBox('TPSA').onchange();
 assert(!walk(ionRoot).some(node=>node.tag==='button'&&node.textContent.startsWith('Override ')));
 assert.equal(walk(ionRoot).find(node=>node.dataset.parameterPath==='generation_params.max_fragment_nodes').value,100);
 assert(!walk(ionRoot).some(node=>node.dataset.parameterPath==='generation_params.ion_prediction_threshold'));
 assert(!walk(ionRoot).some(node=>/loss_weight|intensity_power/.test(node.dataset.parameterPath||'')));
 assert(walk(ionRoot).some(node=>node.className==='parameter-group-heading'&&node.textContent==='Fragment Tree Search'));
 const ionArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:ionEditor.getValue()});
 assert.equal(ionArgs[ionArgs.indexOf('--max-fragment-nodes')+1],'100');
 assert(!ionArgs.includes('--ion-prediction-threshold'));
 assert(!ionArgs.includes('--fragment-loss-weight'));
 const lossArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',intensityLossWeight:0.4,cosineLossWeight:1});
 assert.equal(lossArgs[lossArgs.indexOf('--intensity-loss-weight')+1],'0.4');
 assert.equal(lossArgs[lossArgs.indexOf('--cosine-loss-weight')+1],'1');
 // Neighborhood settings are owned by the per-pattern architecture editor.
 const neighborhoodRoot=new Element('div');createParameterEditor(neighborhoodRoot,{reaction_model_params:{}},'training');
 assert(!walk(neighborhoodRoot).some(node=>node.dataset.parameterPath==='reaction_model_params.reaction_neighborhood_mode'));
 assert(!walk(neighborhoodRoot).some(node=>node.dataset.parameterPath==='reaction_model_params.reaction_neighborhood_max_hop'));
 // Dropout is one shared training-level flag, not a per-component model_config override.
 const dropoutArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',dropout:0.3});
 assert.equal(dropoutArgs[dropoutArgs.indexOf('--fragment-tree-dropout')+1],'0.3');
 const sharedDropoutArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:{reaction_model_params:{state_dropout:0.1},spectrum_model_params:{dropout:0.1}}});
 assert(!sharedDropoutArgs.includes('--reaction-state-dropout'));assert(!sharedDropoutArgs.includes('--spectrum-dropout'));
 const temp=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-parameter-import-'));try{
 const file=path.join(temp,'fragmenter config.json');fs.writeFileSync(file,JSON.stringify({fragment_ion_tree_builder:{max_reaction_count:2},mass_tolerance:'0.04Da'}));let handler;const messages=[];service.attach({webview:{onDidReceiveMessage:fn=>handler=fn,postMessage:data=>messages.push(data)}},{},()=>app);
 await handler({type:'parameters/load',target:'data',path:file});assert.equal(messages.at(-1).type,'parameters/loaded');assert.equal(messages.at(-1).fragmenterParams.mass_tolerance,'0.04Da');assert.equal(messages.at(-1).fragmenterParams.fragment_ion_tree_builder.max_reaction_count,2);assert(messages.at(-1).fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set);
 await handler({type:'parameters/import',target:'data',name:'dropped.json',json:JSON.stringify({reaction_pattern_set:{name:'imported',patterns:[]},fragment_ion_adduct_rule_set:{name:'imported rules',adduct_rules:[]}})});assert.equal(messages.at(-1).fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set.name,'imported');assert.equal(messages.at(-1).fragmenterParams.fragment_ion_tree_builder.fragment_ion_adduct_rule_set.name,'imported rules');
 await handler({type:'parameters/import',target:'data',name:'model.json',json:JSON.stringify({fragmenter_params:{fragment_ion_tree_builder:{}},mol_encoder_params:{hidden_dim:128}})});assert.equal(messages.at(-1).type,'parameters/error');assert.match(messages.at(-1).error,/not a model config/);
 const saved=path.join(temp,'saved.fragmenter.pft');vscodeMock.window.showSaveDialog=async()=>({fsPath:saved});await handler({type:'parameters/save',target:'data',fragmenterParams:{fragment_ion_tree_builder:{max_reaction_count:4},mass_tolerance:'0.02Da'}});assert.equal(messages.at(-1).type,'parameters/saved');assert.deepEqual(JSON.parse(fs.readFileSync(saved,'utf8')),{fragment_ion_tree_builder:{max_reaction_count:4},mass_tolerance:'0.02Da'});
 await handler({type:'parameters/load',target:'training',path:file+'missing'});assert.equal(messages.at(-1).type,'parameters/error');
 }finally{fs.rmSync(temp,{recursive:true,force:true});}
 console.log('Parameter form editing, array updates, advanced settings, execution snapshots and file expansion checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
