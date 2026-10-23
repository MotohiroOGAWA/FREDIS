// Shared read-only model inspection in Models and architecture assembly.
function renderPatternArchitecture(host, item) {
  host.replaceChildren();
  const config=item.architecture||{},grid=document.createElement('div');grid.className='pattern-architecture-grid';
  function group(title, pairs){const section=document.createElement('section'),heading=document.createElement('h4'),list=document.createElement('dl');heading.textContent=title;section.append(heading);for(const [label,value] of pairs){if(value==null)continue;const term=document.createElement('dt'),definition=document.createElement('dd');term.textContent=label;definition.textContent=String(value);list.append(term,definition);}section.append(list);grid.append(section);}
  group('Reactant Graphormer', [['SMARTS atoms',item.atom_count],['Hidden dim',config.hidden_dim??config.node_dim],['Graph output',config.graph_dim],['Layers',config.num_layers],['Heads',config.num_heads],['Dropout',config.dropout]]);
  group('External neighborhood', config.neighborhood_hops===0?[['Status','Disabled']]:[['Radius (hops)',config.neighborhood_hops],['Hidden dim',config.neighborhood_hidden_dim],['Bond features',config.neighborhood_edge_features?'Used':'Connectivity only (saved model)']]);
  group('Chemical inputs', [['Atom features','Chemical features + SMARTS position one-hot']]);
  const flow=document.createElement('p');flow.className='model-feature-flow';flow.textContent='Chemical atoms'+(config.neighborhood_hops?' + external neighborhood':'')+' + SMARTS positions → Reactant Graphormer → graph features';host.append(flow,grid);
}
function renderPatternModelDetails(host, item) {
  const details=document.createElement('details'),summary=document.createElement('summary');details.className='pattern-model-detail';
  summary.textContent=item.pattern.name+' · '+item.atom_count+' SMARTS atoms';details.append(summary);
  const info=document.createElement('p');info.className='model-meta';info.textContent=(item.parameter_count==null?'':item.parameter_count.toLocaleString()+' parameters · ')+(item.frozen?'Frozen feature':item.source?'Pretrained feature':'Untrained reaction pattern');details.append(info);
  const smarts=document.createElement('code');smarts.className='model-smarts';smarts.textContent=item.pattern.reactant_smarts;details.append(smarts);
  if(item.architecture){const architecture=document.createElement('div');details.append(architecture);details.updateArchitecture=()=>{renderPatternArchitecture(architecture,item);const value=details.querySelector('[data-model-architecture]');if(value)value.textContent=JSON.stringify(item.architecture,null,2);};details.updateArchitecture();}
  const raw=document.createElement('details'),rawSummary=document.createElement('summary'),definition=document.createElement('pre');rawSummary.textContent=item.architecture?'Reaction definition and saved parameters':'Reaction definition';definition.textContent=JSON.stringify(item.pattern,null,2);raw.append(rawSummary,definition);
  if(item.architecture){const value=document.createElement('pre');value.dataset.modelArchitecture='';value.textContent=JSON.stringify(item.architecture,null,2);raw.append(value);}if(item.source){const source=document.createElement('p');source.className='model-meta';source.textContent=item.source;raw.append(source);}details.append(raw);host.append(details);return details;
}
function client(){
 const dataForm=document.getElementById('form'),trainingForm=document.getElementById('trainingForm');
 const host=document.getElementById('dataPatternModels');
 let records=JSON.parse(JSON.stringify(initial.reactionModels||[])),updating=false;
 let architecture=JSON.parse(JSON.stringify(initialTraining.patternArchitecture||{mode:'shared',shared:{}}));architecture.shared??={};
 let individualInitialized=architecture.mode==='individual';
 const parameterKeys=['hidden_dim','graph_dim','num_layers','num_heads','neighborhood_hops','neighborhood_hidden_dim'];
 let trainingPatterns=[],settings=JSON.parse(JSON.stringify(initialTraining.modelConfig?.reaction_model_params?.pattern_model_params||{}));
 const originalData=getConfig,originalTraining=getTrainingConfig;
 const same=(a,b)=>a.reactant_smarts===b.reactant_smarts&&JSON.stringify((a.products||[]).map(item=>item.smarts).sort())===JSON.stringify((b.products||[]).map(item=>item.smarts).sort());
 function reconcile(){const patterns=originalData().fragmenterParams?.fragment_ion_tree_builder?.reaction_pattern_set?.patterns||[];records=patterns.map(pattern=>{const found=records.find(item=>same(item.pattern,pattern));return found?{...found,pattern}:{pattern};});}
 getConfig=()=>{reconcile();return {...originalData(),reactionModels:records.map(({pattern,source,source_pattern_id})=>({pattern,...(source?{source,source_pattern_id}:{})}))};};
 getTrainingConfig=()=>{const config=originalTraining();delete config.pretrainedPatterns;if(trainingForm.querySelector('input[name=trainingMode]:checked')?.value!=='resume'){const reaction=config.modelConfig.reaction_model_params??={};config.modelConfig.dropout=config.dropout;config.modelConfig.spectrum_model_params??={};delete config.modelConfig.spectrum_model_params.fragment_dropout;delete config.modelConfig.spectrum_model_params.state_dropout;if(trainingForm.dataset.hasPretrainedPatterns!=='true'){config.modelConfig.mol_encoder_params.dropout=config.dropout;config.modelConfig.mol_encoder_params.external_feature_dim=0;}delete reaction.edge_dropout;delete reaction.reactant_dropout;config.patternArchitecture={...JSON.parse(JSON.stringify(architecture)),frozenPatternIds:trainingPatterns.filter(item=>item.source).map(item=>String(item.pattern.pattern_id))};reaction.pattern_model_params=Object.fromEntries(Object.entries(settings).map(([pid,params])=>{const copy={...params};delete copy.dropout;return [pid,copy];}));}return resolvePatternArchitecture(config);};
 function changed(){dataForm.dispatchEvent(new Event('input',{bubbles:true}));}
 function trainingChanged(){const invalid=Object.entries(getTrainingConfig().modelConfig.reaction_model_params.pattern_model_params||{}).filter(([pid])=>!trainingPatterns.some(item=>String(item.pattern.pattern_id)===pid&&item.source)).some(([,config])=>!['hidden_dim','graph_dim','num_heads','num_layers','neighborhood_hidden_dim'].every(key=>Number.isInteger(config[key])&&config[key]>0)||config.hidden_dim%config.num_heads!==0||!Number.isInteger(config.neighborhood_hops)||config.neighborhood_hops<0);trainingForm.dataset.patternModelsValid=invalid?'false':'true';trainingForm.dispatchEvent(new Event('input',{bubbles:true}));}
 function setPatterns(patterns){updating=true;window.dispatchEvent(new MessageEvent('message',{data:{type:'data/reactionPatterns',patterns}}));updating=false;render();changed();}
 function add(items){reconcile();const patterns=(originalData().fragmenterParams?.fragment_ion_tree_builder?.reaction_pattern_set?.patterns||[]).slice();for(const item of items){records=records.filter(record=>!same(record.pattern,item.pattern));records.push(item);if(!patterns.some(pattern=>same(pattern,item.pattern)))patterns.push(item.pattern);}setPatterns(patterns);}
 function render(){reconcile();host.replaceChildren();for(const record of records){const view=renderPatternModelDetails(host,{pattern:record.pattern,architecture:record.source?record.architecture:null,atom_count:record.atom_count||'fixed',parameter_count:record.parameter_count,source:record.source});const remove=document.createElement('button');remove.type='button';remove.textContent='Remove pattern';remove.onclick=()=>{records=records.filter(item=>item!==record);setPatterns((originalData().fragmenterParams?.fragment_ion_tree_builder?.reaction_pattern_set?.patterns||[]).filter(pattern=>!same(pattern,record.pattern)));};view.append(remove);}}
 function addParameterInputs(host,config,shared=false,view=null){
  for(const key of parameterKeys){
   const label=document.createElement('label'),input=document.createElement('input');
   label.textContent=key==='neighborhood_hops'?'Neighborhood hops (0 disables)':key;
   input.type='number';input.step='1';input.min=key==='neighborhood_hops'?'0':'1';input.value=config[key];
   if(shared){input.dataset.sharedPatternParameter=key;input.dataset.tuningKey='patternArchitecture.shared.'+key;}
   else input.dataset.patternParameter=key;
   if(key==='neighborhood_hidden_dim')input.disabled=config.neighborhood_hops===0;
   input.oninput=()=>{config[key]=Number(input.value);if(key==='neighborhood_hops'){const peer=host.querySelector(shared?'[data-shared-pattern-parameter=neighborhood_hidden_dim]':'[data-pattern-parameter=neighborhood_hidden_dim]');if(peer)peer.disabled=config.neighborhood_hops===0;}view?.updateArchitecture?.();trainingChanged();};
   label.append(input);host.append(label);
  }
 }
 function renderTraining(){
  const target=document.getElementById('trainingPatternModels');target.replaceChildren();
  const locked=trainingForm.querySelector('input[name=trainingMode]:checked')?.value==='resume';
  const editable=trainingPatterns.filter(item=>!item.source);
  const first=editable[0];
  if(first)for(const key of parameterKeys)architecture.shared[key]??=(settings[String(first.pattern.pattern_id)]||first.architecture)[key]??({neighborhood_hops:3,neighborhood_hidden_dim:32}[key]);
  if(!locked){
   const label=document.createElement('label'),select=document.createElement('select');select.id='patternParameterMode';
   label.textContent='Reaction parameter settings';
   for(const [value,text]of [['shared','Shared · all new reactions (default)'],['individual','Individual · per reaction']]){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}
   select.value=architecture.mode;select.disabled=!editable.length;
   select.onchange=()=>{architecture.mode=select.value;if(select.value==='individual'&&!individualInitialized){for(const item of editable)Object.assign(settings[String(item.pattern.pattern_id)],architecture.shared);individualInitialized=true;}renderTraining();trainingChanged();};label.append(select);target.append(label);
   if(architecture.mode==='shared'){
    const section=document.createElement('section');section.id='sharedPatternParameters';
    const title=document.createElement('h4');title.textContent='Architecture for all new reactions';section.append(title);
    if(editable.length)addParameterInputs(section,architecture.shared,true);
    else{const note=document.createElement('p');note.textContent='No new reactions to configure. Pretrained architectures remain fixed.';section.append(note);}
    target.append(section);
   }
  }
  for(const item of trainingPatterns){
   const pid=String(item.pattern.pattern_id),config=settings[pid]??={...item.architecture,input_mode:'chemical'};
   if(!locked&&!item.source){config.input_mode='chemical';config.neighborhood_edge_features=true;delete config.descriptors;}
   config.neighborhood_hops??=3;config.neighborhood_hidden_dim??=32;
   const effective=!locked&&!item.source&&architecture.mode==='shared'?{...config,...architecture.shared}:config;
   if(!locked&&!item.source)effective.dropout=Number(trainingForm.elements.dropout.value);
   const trained=locked||!!item.source;
   const view=renderPatternModelDetails(target,{...item,source:item.source||(locked?'Resume checkpoint (architecture fixed)':null),architecture:trained?effective:null,parameter_count:trained?item.parameter_count:null});view.dataset.patternId=pid;
   if(!locked&&!item.source&&architecture.mode==='individual'){
    addParameterInputs(view,config,false,view);
   }
  }
 }
 const editor=document.getElementById('dataPatternEditor');
 document.getElementById('dataPatternEdit').onclick=()=>{
  reconcile();editor.hidden=false;host.hidden=true;for(const id of ['dataPatternImport','dataPatternAdd','dataPatternEdit'])document.getElementById(id).disabled=true;
  window.dispatchEvent(new CustomEvent('data/editNewPatterns',{detail:{patterns:records.filter(item=>!item.source).map(item=>item.pattern),imported:records.filter(item=>item.source).map(item=>item.pattern)}}));
 };
 document.getElementById('dataPatternEditClose').onclick=()=>{editor.hidden=true;host.hidden=false;for(const id of ['dataPatternImport','dataPatternAdd','dataPatternEdit'])document.getElementById(id).disabled=false;const app=document.getElementById('reactionPatternApp');app.hidden=true;document.getElementById('reactionPatternPageMount').append(app);onReactionPatternChange=()=>{};render();changed();};
 window.addEventListener('data/patternsEdited',()=>render());
 document.getElementById('dataPatternImport').onclick=()=>vscode.postMessage({type:'models/pick',target:'data'});
 document.getElementById('dataPatternAdd').onclick=()=>vscode.postMessage({type:'models/patternPick',target:'data'});
 dataForm.addEventListener('input',event=>{if(!updating&&!host.contains(event.target))render();});
 trainingForm.addEventListener('input',event=>{if(event.target.name==='dropout')renderTraining();});
 trainingForm.addEventListener('change',event=>{if(event.target.name==='trainingMode')renderTraining();});
 window.addEventListener('message',event=>{const m=event.data;
  if(m.type==='models/inspected'&&m.target==='data'){if(m.error){document.getElementById('dataPatternStatus').textContent=m.error;return;}add(m.patterns.map(item=>({pattern:item.pattern,architecture:item.architecture,source:m.path,source_pattern_id:item.pattern.pattern_id,atom_count:item.atom_count,parameter_count:item.parameter_count})));document.getElementById('dataPatternStatus').textContent='Added '+m.patterns.length+' pretrained reaction models to dataset chemistry.';}
  if(m.type==='models/patterns'){if(m.error){document.getElementById('dataPatternStatus').textContent=m.error;return;}add(m.patterns.map(pattern=>({pattern})));document.getElementById('dataPatternStatus').textContent='Added '+m.patterns.length+' new reaction patterns. Configure their architectures in Training.';}
  if(m.type==='config'){records=JSON.parse(JSON.stringify(m.config.reactionModels||[]));render();}
  if(m.type==='parameters/loaded'&&m.target==='data')render();
  if(m.type==='trainingConfig'){architecture=JSON.parse(JSON.stringify(m.config.patternArchitecture||{mode:'shared',shared:{}}));architecture.shared??={};individualInitialized=architecture.mode==='individual';settings=JSON.parse(JSON.stringify(m.config.modelConfig?.reaction_model_params?.pattern_model_params||{}));renderTraining();}
  if(m.type==='training/sources'&&!m.error&&m.requestId===trainingForm.dataset.sourcesRequest){const incoming=m.patternModels||[];const retained={};for(const item of incoming){const previous=trainingPatterns.find(old=>same(old.pattern,item.pattern));const prior=previous?settings[String(previous.pattern.pattern_id)]:settings[String(item.pattern.pattern_id)];retained[String(item.pattern.pattern_id)]=item.source?item.architecture:(prior||{...item.architecture,input_mode:'chemical'});}settings=retained;trainingPatterns=incoming;trainingForm.dataset.hasPretrainedPatterns=m.modelConfig.fine_tuning?.frozen_pattern_ids?.length?'true':'false';renderTraining();if(m.modelConfig.fine_tuning)window.dispatchEvent(new MessageEvent('message',{data:{type:'training/importedFeatures',modelConfig:m.modelConfig}}));}
  if(m.type==='training/checkpoint'&&!m.error&&m.path===trainingForm.elements.resume.value){trainingPatterns=m.patternModels||[];settings=Object.fromEntries(trainingPatterns.map(item=>[String(item.pattern.pattern_id),item.architecture]));renderTraining();}
 });render();
}
module.exports={renderPatternModelDetails,script:()=>`${require('./reaction-parameters').resolvePatternArchitecture.toString()}${renderPatternArchitecture.toString()}${renderPatternModelDetails.toString()}(${client.toString()})();`};
