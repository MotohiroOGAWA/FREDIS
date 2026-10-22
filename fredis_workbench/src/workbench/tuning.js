// Grid search expands the same shared/individual settings used by ordinary training.
const path=require('path');
const {resolvePatternArchitecture}=require('./reaction-parameters');
function plan(config,build){
 if(config.resume||config.trainingMode==='resume')throw new Error('Tuning requires a new model.');
 const entries=Object.entries(config.tuning?.parameters||{});
 if(!entries.length)throw new Error('Select at least one parameter to tune.');
 const frozen=new Set((config.patternArchitecture?.frozenPatternIds||[]).map(String));
 let trials=[JSON.parse(JSON.stringify(config))];
 for(const [key,raw]of entries){
  if(key==='modelConfig.mol_encoder_params.external_feature_dim')throw new Error('External feature width is determined by supplied features and cannot be tuned.');
  if(key!=='dropout'&&/dropout$/.test(key))throw new Error('Use the shared dropout tuning parameter.');
  const parts=key.split('.');
  if(!/^(lr|batchSize|dropout|weightDecay|patternArchitecture\.shared\.(hidden_dim|graph_dim|num_layers|num_heads|neighborhood_hops|neighborhood_hidden_dim)|modelConfig\.(mol_encoder_params|reaction_model_params|spectrum_model_params)\.[\w.]+)$/.test(key)||parts.some(p=>['__proto__','constructor','prototype'].includes(p)))throw new Error('Unsupported tuning parameter: '+key);
  const pattern=key.match(/^modelConfig\.reaction_model_params\.pattern_model_params\.(\d+)\./);
  if(pattern&&frozen.has(pattern[1]))throw new Error('Pretrained reaction parameters cannot be tuned.');
  if(config.patternArchitecture?.mode==='shared'&&pattern)throw new Error('Use the shared reaction settings to tune in Shared mode.');
  if(key.startsWith('patternArchitecture.shared.')&&config.patternArchitecture?.mode!=='shared')throw new Error('Shared candidates require Shared reaction settings.');
  if(!Array.isArray(raw)||!raw.length||raw.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0))throw new Error('Enter numeric candidates for '+key);
  if(raw.some(v=>/(dim|heads|layers|batchSize)$/.test(key)&&(!Number.isInteger(v)||v<1)||/hops$/.test(key)&&!Number.isInteger(v)))throw new Error('Positive integer candidates required (hops may be zero): '+key);
  if(key.endsWith('dropout')&&raw.some(v=>v>=1))throw new Error('Dropout candidates must be below 1.');
  trials=trials.flatMap(c=>[...new Set(raw)].map(value=>{
   const next=JSON.parse(JSON.stringify(c));let target=next;
   for(const part of parts.slice(0,-1)){if(!target[part]||typeof target[part]!=='object')throw new Error('Parameter no longer available: '+key);target=target[part];}
   if(!(parts.at(-1)in target))throw new Error('Parameter no longer available: '+key);
   target[parts.at(-1)]=value;return next;
  }));
 }
 return trials.map((c,index)=>{
  resolvePatternArchitecture(c);
  for(const section of [c.modelConfig.mol_encoder_params,c.modelConfig.reaction_model_params,c.modelConfig.spectrum_model_params,...Object.values(c.modelConfig.reaction_model_params?.pattern_model_params||{})]){
   if(!section)continue;const dim=section.node_dim??section.hidden_dim;
   if(dim&&section.num_heads&&dim%section.num_heads)throw new Error('Every hidden dimension must be divisible by its attention heads.');
  }
  if(!(c.lr>0)||!Number.isInteger(c.batchSize)||c.batchSize<1)throw new Error('Learning rate and batch size must be positive.');
  delete c.tuning;c.overwrite=false;c.outputDir=path.join(config.outputDir,'trial-'+String(index+1).padStart(3,'0'));return {config:c,args:build(c)};
 });
}
function client(){
 const form=document.getElementById('trainingForm'),toggle=document.getElementById('trainingTuningToggle'),panel=document.getElementById('trainingTuningPanel'),status=document.getElementById('trainingTuningStatus');
 let state=JSON.parse(JSON.stringify(initialTraining.tuning||{enabled:false,parameters:{}}));const original=getTrainingConfig;
 state.parameters??={};delete state.groups;
 getTrainingConfig=()=>{const c=original();delete state.evaluation;c.tuning=JSON.parse(JSON.stringify(state));return c;};
 const active=()=>state.enabled&&form.querySelector('[name=trainingMode]:checked')?.value!=='resume';
 function refresh(){
  toggle.setAttribute('aria-pressed',String(!!state.enabled));panel.hidden=!state.enabled;form.classList.toggle('tuning-active',active());
  const config=getTrainingConfig(),architecture=config.patternArchitecture,frozen=new Set((architecture?.frozenPatternIds||[]).map(String)),known=new Set(Object.keys(config.modelConfig?.reaction_model_params?.pattern_model_params||{}));
  for(const key of Object.keys(state.parameters)){
   const pattern=key.match(/^modelConfig\.reaction_model_params\.pattern_model_params\.(\d+)\./);
   if(pattern&&(!known.has(pattern[1])||frozen.has(pattern[1])||architecture?.mode==='shared')||key.startsWith('patternArchitecture.shared.')&&(architecture?.mode!=='shared'||!form.querySelector('[data-shared-pattern-parameter]')))delete state.parameters[key];
  }
  for(const input of form.querySelectorAll('[data-shared-pattern-parameter],[data-pattern-parameter],#trainingParameterEditor input[type=number],input[name=lr],input[name=batchSize],input[name=dropout],input[name=weightDecay]')){
   if(input.closest('.tuning-control'))continue;
   let key=input.dataset.tuningKey||(input.dataset.parameterPath?'modelConfig.'+input.dataset.parameterPath:input.name);
   if(input.dataset.patternParameter){const pid=input.closest('[data-pattern-id]')?.dataset.patternId;if(pid===undefined)continue;key='modelConfig.reaction_model_params.pattern_model_params.'+pid+'.'+input.dataset.patternParameter;}
   if(!key||key.startsWith('modelConfig.generation_params.'))continue;
   let control=input.parentElement.querySelector('.tuning-control');
   if(!control){
    control=document.createElement('span');control.className='tuning-control';const check=document.createElement('input'),values=document.createElement('input');check.type='checkbox';check.setAttribute('aria-label','Tune '+key);values.type='text';values.placeholder='e.g. 1, 2, 3';values.title='Candidate values (comma-separated)';values.setAttribute('aria-label','Candidate values for '+key);control.append(check,document.createTextNode('Tune'),values);input.after(control);
    const change=()=>{if(check.checked)state.parameters[key]=values.value.split(',').map(v=>v.trim()===''?NaN:Number(v.trim()));else delete state.parameters[key];values.hidden=!check.checked;refresh();};check.oninput=change;check.onchange=change;values.oninput=change;
   }
   const [check,values]=control.querySelectorAll('input');control.hidden=!active()||input.disabled;
   if(input.disabled)delete state.parameters[key];check.checked=key in state.parameters;values.hidden=!check.checked;if(document.activeElement!==values)values.value=(state.parameters[key]||[Number(input.value)]).join(', ');
  }
  const entries=Object.entries(state.parameters),count=entries.reduce((n,[,v])=>n*new Set(v).size,1);
  status.textContent=entries.length?entries.length+' parameters · '+entries.length+' search dimensions · '+count+' trials · each trial uses the configured epochs.':'Enable Tune beside a shared reaction parameter to tune all new reactions together, or use Individual settings to tune separately.';
  if(form.querySelector('[name=trainingMode]:checked')?.value==='resume')status.textContent='Select New model to tune architecture and training parameters.';
 }
 toggle.onclick=()=>{state.enabled=!state.enabled;refresh();};
 form.addEventListener('input',()=>queueMicrotask(refresh));form.addEventListener('change',()=>queueMicrotask(refresh));
 window.addEventListener('message',e=>{if(e.data.type==='trainingConfig'){state=JSON.parse(JSON.stringify(e.data.config.tuning||{enabled:false,parameters:{}}));state.parameters??={};delete state.groups;}queueMicrotask(refresh);});
 refresh();
}
module.exports={plan,script:()=>`(${client.toString()})();`};
