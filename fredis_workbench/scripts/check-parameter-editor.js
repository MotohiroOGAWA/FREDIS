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
 // A resumed config always carries reaction_model_params.state_dropout and
 // spectrum_model_params.dropout (training_model_config writes both from the one
 // shared --dropout), but only that one shared field may ever be edited.
 const dropoutConfigRoot=new Element('div');
 createParameterEditor(dropoutConfigRoot,{reaction_model_params:{state_dropout:0.3},spectrum_model_params:{dropout:0.3}},'training');
 assert(!walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='reaction_model_params.state_dropout'));
 assert(!walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='spectrum_model_params.dropout'));
 // Generation settings are their own model_config section, filled in with
 // defaults immediately -- no "Override" button to click first.
 const ionRoot=new Element('div'),ionEditor=createParameterEditor(ionRoot,{spectrum_model_params:{}},'training');
 // Mol Encoder descriptors are checkboxes; the default is FormalCharge only.
 assert.deepEqual(ionEditor.getValue().mol_encoder_params.descriptors,['FormalCharge']);
 const descriptorBox=name=>walk(ionRoot).find(node=>node.dataset.parameterPath==='mol_encoder_params.descriptors.'+name);
 assert.equal(descriptorBox('FormalCharge').type,'checkbox');assert(descriptorBox('FormalCharge').checked);assert(!descriptorBox('TPSA').checked);
 descriptorBox('TPSA').checked=true;descriptorBox('TPSA').onchange();assert.deepEqual(ionEditor.getValue().mol_encoder_params.descriptors,['TPSA','FormalCharge']);
 descriptorBox('FormalCharge').checked=false;descriptorBox('FormalCharge').onchange();assert.deepEqual(ionEditor.getValue().mol_encoder_params.descriptors,['TPSA']);
 const descriptorArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:ionEditor.getValue()});
 assert.deepEqual(descriptorArgs.slice(descriptorArgs.indexOf('--descriptors'),descriptorArgs.indexOf('--descriptors')+2),['--descriptors','TPSA']);
 descriptorBox('TPSA').checked=false;descriptorBox('TPSA').onchange();
 assert(!walk(ionRoot).some(node=>node.tag==='button'&&node.textContent.startsWith('Override ')));
 assert.equal(walk(ionRoot).find(node=>node.dataset.parameterPath==='generation_params.max_fragment_nodes').value,100);
 assert.equal(walk(ionRoot).find(node=>node.dataset.parameterPath==='generation_params.ion_prediction_threshold').value,0.5);
 assert(!walk(ionRoot).some(node=>/loss_weight|intensity_power/.test(node.dataset.parameterPath||'')));
 assert(walk(ionRoot).some(node=>node.className==='parameter-group-heading'&&node.textContent==='Fragment Tree Search'));
 const ionArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:ionEditor.getValue()});
 assert.equal(ionArgs[ionArgs.indexOf('--max-fragment-nodes')+1],'100');
 assert.equal(ionArgs[ionArgs.indexOf('--ion-prediction-threshold')+1],'0.5');
 assert(!ionArgs.includes('--fragment-loss-weight'));
 const lossArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',fragmentLossWeight:0.4,intensityPower:1});
 assert.equal(lossArgs[lossArgs.indexOf('--fragment-loss-weight')+1],'0.4');
 assert.equal(lossArgs[lossArgs.indexOf('--intensity-power')+1],'1');
 // reaction_neighborhood_mode is a closed-choice Primitive Reaction Encoder setting,
 // defaulted like every other optional reaction_model_params field.
 const neighborhoodRoot=new Element('div'),neighborhoodEditor=createParameterEditor(neighborhoodRoot,{reaction_model_params:{}},'training');
 const neighborhoodMode=walk(neighborhoodRoot).find(node=>node.dataset.parameterPath==='reaction_model_params.reaction_neighborhood_mode');
 assert.equal(neighborhoodMode.tag,'select');assert.deepEqual(neighborhoodMode.children.map(option=>option.value),['hop_pooling','none']);assert.equal(neighborhoodMode.value,'hop_pooling');
 // Neighborhood Max Hop defaults to 3 and starts enabled alongside hop_pooling.
 const maxHop=walk(neighborhoodRoot).find(node=>node.dataset.parameterPath==='reaction_model_params.reaction_neighborhood_max_hop');
 assert.equal(maxHop.value,3);assert(!maxHop.disabled);
 maxHop.value='2';maxHop.oninput();assert.equal(neighborhoodEditor.getValue().reaction_model_params.reaction_neighborhood_max_hop,2);
 neighborhoodMode.value='none';neighborhoodMode.onchange();assert.equal(neighborhoodEditor.getValue().reaction_model_params.reaction_neighborhood_mode,'none');
 // Switching to Mode=none re-renders and disables the now-meaningless Max Hop field.
 assert(walk(neighborhoodRoot).find(node=>node.dataset.parameterPath==='reaction_model_params.reaction_neighborhood_max_hop').disabled);
 const neighborhoodArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:neighborhoodEditor.getValue()});
 assert.equal(neighborhoodArgs[neighborhoodArgs.indexOf('--reaction-neighborhood-mode')+1],'none');
 assert.equal(neighborhoodArgs[neighborhoodArgs.indexOf('--reaction-neighborhood-max-hop')+1],'2');
 // max_roles is sized from the prepared dataset, never a form field, even
 // when a resumed/loaded config already carries one.
 const maxRolesRoot=new Element('div');
 createParameterEditor(maxRolesRoot,{reaction_model_params:{max_roles:96}},'training');
 assert(!walk(maxRolesRoot).some(node=>node.dataset.parameterPath==='reaction_model_params.max_roles'));
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
