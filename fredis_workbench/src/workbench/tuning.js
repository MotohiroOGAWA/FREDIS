// Grid search uses the same CLI builder as an ordinary training run.
const path=require('path');
function plan(config,build){
 if(config.resume||config.trainingMode==='resume')throw new Error('Tuning requires a new model.');
 const entries=Object.entries(config.tuning?.parameters||{});
 if(!entries.length)throw new Error('Select at least one parameter to tune.');
 const axes=[],groups=new Map();
 for(const [key,raw] of entries){
  const group=config.tuning.groups?.[key];
  if(group!==undefined&&typeof group!=='string')throw new Error('Invalid tuning group: '+key);
  if(group){
   if(!Array.isArray(raw))throw new Error('Enter numeric candidates for '+key);
   if(groups.has(group)){
    const axis=groups.get(group);
    if(JSON.stringify([...new Set(axis[1])])!==JSON.stringify([...new Set(raw)]))throw new Error('Linked parameters must use the same candidates: '+group);
    axis[0].push(key);
   }else{const axis=[[key],raw];groups.set(group,axis);axes.push(axis);}
  }else axes.push([[key],raw]);
 }
 let trials=[JSON.parse(JSON.stringify(config))];
 for(const [keys,raw]of axes){
  for(const key of keys){
  const parts=key.split('.');
  if(!/^(lr|batchSize|dropout|weightDecay|modelConfig\.(mol_encoder_params|reaction_model_params|spectrum_model_params)\.[\w.]+)$/.test(key)||parts.some(p=>['__proto__','constructor','prototype'].includes(p)))throw new Error('Unsupported tuning parameter: '+key);
  if(!Array.isArray(raw)||!raw.length||raw.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0))throw new Error('Enter numeric candidates for '+key);
  }
  const values=[...new Set(raw)];
  trials=trials.flatMap(c=>values.map(value=>{const next=JSON.parse(JSON.stringify(c));for(const key of keys){const parts=key.split('.');let target=next;for(const part of parts.slice(0,-1)){if(!target[part]||typeof target[part]!=='object')throw new Error('Parameter no longer available: '+key);target=target[part];}if(!(parts.at(-1) in target))throw new Error('Parameter no longer available: '+key);target[parts.at(-1)]=value;}return next;}));
 }
 return trials.map((c,index)=>{
  for(const key of Object.keys(c.tuning.parameters)){let target=c;for(const part of key.split('.'))target=target[part];if(/(dim|heads|layers|batchSize)$/.test(key)&&(!Number.isInteger(target)||target<1))throw new Error('Positive integer candidates required: '+key);if(/(hops|layers)$/.test(key)&&!Number.isInteger(target))throw new Error('Integer candidates required: '+key);if(key.endsWith('dropout')&&target>=1)throw new Error('Dropout candidates must be below 1.');}
  for(const section of [c.modelConfig.mol_encoder_params,c.modelConfig.reaction_model_params,c.modelConfig.spectrum_model_params,...Object.values(c.modelConfig.reaction_model_params?.pattern_model_params||{})]){if(!section)continue;const dim=section.node_dim??section.hidden_dim;if(dim&&section.num_heads&&dim%section.num_heads)throw new Error('Every hidden dimension must be divisible by its attention heads.');}
  if(!(c.lr>0)||!Number.isInteger(c.batchSize)||c.batchSize<1)throw new Error('Learning rate and batch size must be positive.');
  delete c.tuning;c.overwrite=false;c.outputDir=path.join(config.outputDir,'trial-'+String(index+1).padStart(3,'0'));return {config:c,args:build(c)};
 });
}
function client(){
 const form=document.getElementById('trainingForm'),toggle=document.getElementById('trainingTuningToggle'),panel=document.getElementById('trainingTuningPanel'),status=document.getElementById('trainingTuningStatus');
 let state=JSON.parse(JSON.stringify(initialTraining.tuning||{enabled:false,parameters:{}}));const original=getTrainingConfig;
 getTrainingConfig=()=>{const c=original();delete state.evaluation;c.tuning=JSON.parse(JSON.stringify(state));return c;};
 state.parameters??={};state.groups??={};
 const active=()=>state.enabled&&form.querySelector('[name=trainingMode]:checked')?.value!=='resume';
 function refresh(){
  toggle.setAttribute('aria-pressed',String(!!state.enabled));panel.hidden=!state.enabled;form.classList.toggle('tuning-active',active());
  for(const input of form.querySelectorAll('[data-pattern-parameter],#trainingParameterEditor input[type=number],input[name=lr],input[name=batchSize],input[name=dropout],input[name=weightDecay]')){
   if(input.closest('.tuning-control'))continue;
   let key=input.dataset.parameterPath?'modelConfig.'+input.dataset.parameterPath:input.name;
   if(input.dataset.patternParameter){const pid=input.closest('[data-pattern-id]')?.dataset.patternId;if(pid===undefined)continue;key='modelConfig.reaction_model_params.pattern_model_params.'+pid+'.'+input.dataset.patternParameter;}
   if(!key||key.startsWith('modelConfig.generation_params.'))continue;
   let control=input.parentElement.querySelector('.tuning-control');
   if(!control){control=document.createElement('span');control.className='tuning-control';const check=document.createElement('input'),values=document.createElement('input');check.type='checkbox';check.setAttribute('aria-label','Tune '+key);values.type='text';values.placeholder='e.g. 1, 2, 3';values.title='Candidate values (comma-separated)';values.setAttribute('aria-label','Candidate values for '+key);control.append(check,document.createTextNode('Tune'),values);input.after(control);
    const change=()=>{if(check.checked)state.parameters[key]=values.value.split(',').map(v=>v.trim()===''?NaN:Number(v.trim()));else {delete state.parameters[key];delete state.groups[key];}if(check.checked&&state.groups[key])for(const other of Object.keys(state.parameters))if(state.groups[other]===state.groups[key])state.parameters[other]=[...state.parameters[key]];values.hidden=!check.checked;refresh();};check.oninput=change;check.onchange=change;values.oninput=change;
   }
   if(input.dataset.patternParameter&&!control.querySelector('[data-tuning-group]')){
    const group=document.createElement('input');group.type='text';group.dataset.tuningGroup='';group.placeholder='Link group (optional)';group.title='Parameters with the same group name use the same candidate in each trial. Clear to unlink.';group.setAttribute('aria-label','Link group for '+key);
    group.onchange=()=>{const name=group.value.trim();if(name){const other=Object.keys(state.parameters).find(k=>k!==key&&state.groups[k]===name);state.groups[key]=name;if(other)state.parameters[key]=[...state.parameters[other]];}else delete state.groups[key];refresh();};
    const all=document.createElement('button');all.type='button';all.textContent='Link all patterns';all.onclick=()=>{const name='all-patterns:'+input.dataset.patternParameter;const unlink=state.groups[key]===name;const candidates=state.parameters[key]||[Number(input.value)];for(const peer of form.querySelectorAll('[data-pattern-parameter]')){if(peer.dataset.patternParameter!==input.dataset.patternParameter||peer.disabled)continue;const pid=peer.closest('[data-pattern-id]')?.dataset.patternId;if(pid===undefined)continue;const k='modelConfig.reaction_model_params.pattern_model_params.'+pid+'.'+peer.dataset.patternParameter;if(unlink){if(state.groups[k]===name)delete state.groups[k];}else{state.parameters[k]=[...candidates];state.groups[k]=name;}}refresh();};control.append(group,all);
   }
   const [check,values]=control.querySelectorAll('input');control.hidden=!active()||input.disabled;if(input.disabled){delete state.parameters[key];delete state.groups[key];}check.checked=key in state.parameters;values.hidden=!check.checked;if(document.activeElement!==values)values.value=(state.parameters[key]||[Number(input.value)]).join(', ');
   const group=control.querySelector('[data-tuning-group]');if(group){group.hidden=!check.checked;if(document.activeElement!==group)group.value=state.groups[key]||'';const all=control.querySelector('button');const linked=state.groups[key]==='all-patterns:'+input.dataset.patternParameter;all.textContent=linked?'Unlink all patterns':'Link all patterns';all.setAttribute('aria-pressed',String(linked));all.title=linked?'Remove the shared group; keep Tune enabled for independent tuning.':'Use the same candidates across all editable patterns.';}
  }describe();
 }
 function describe(){const entries=Object.entries(state.parameters),axes=new Map();for(const [key,values]of entries)axes.set(state.groups[key]?'group:'+state.groups[key]:'parameter:'+key,values);const count=[...axes.values()].reduce((n,v)=>n*new Set(v).size,1);status.textContent=entries.length?entries.length+' parameters · '+axes.size+' search dimensions · '+count+' trials · each trial uses the configured epochs.':'Enable Tune beside a parameter and enter comma-separated candidates. Use Link all patterns, or enter the same link group for selected patterns.';if(form.querySelector('[name=trainingMode]:checked')?.value==='resume')status.textContent='Select New model to tune architecture and training parameters.';}

 toggle.onclick=()=>{state.enabled=!state.enabled;refresh();};
 form.addEventListener('input',()=>queueMicrotask(refresh));form.addEventListener('change',()=>queueMicrotask(refresh));
 window.addEventListener('message',e=>{if(e.data.type==='trainingConfig'){state=JSON.parse(JSON.stringify(e.data.config.tuning||{enabled:false,parameters:{}}));state.parameters??={};state.groups??={};}queueMicrotask(refresh);});
 refresh();
}
module.exports={plan,script:()=>`(${client.toString()})();`};
