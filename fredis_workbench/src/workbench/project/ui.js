function html(){return `<section id="projectHome" aria-label="Project"><div class="section-title"><div><h2 id="projectTitle">Your research project</h2><p id="projectPath" class="muted">Choose a directory to keep settings, files and runs together.</p></div><div class="actions"><button id="projectOpen" class="primary">Open / Create Project</button><button id="projectReveal" hidden>Reveal Folder</button><button id="projectClose" hidden>Close Project</button></div></div><p id="projectError" class="danger" role="alert"></p><div id="projectRecent" class="actions"></div><div id="projectContent" hidden><p id="projectSummary"></p><p id="projectSaveStatus" class="muted" role="status">Form settings are saved automatically. Visual patterns are included after registering them.</p><details><summary>Project name and notes</summary><label>Project name<input id="projectName" maxlength="200"></label><label>Research notes<textarea id="projectNotes" rows="4" maxlength="20000" placeholder="Purpose, dataset versions, observations and next steps…"></textarea></label><button id="projectSaveNotes">Save Notes</button></details><div class="project-tools"><label>Search history and files<input id="projectSearch" type="search" placeholder="Name, path or workflow"></label><label>Workflow<select id="projectWorkflow"><option value="">All workflows</option><option value="data">Data preparation</option><option value="training">Training</option><option value="prediction">Prediction</option><option value="reactionPatterns">Reaction patterns</option></select></label><button id="projectRuns">Runs &amp; Logs</button><button id="projectRefresh">Refresh</button></div><div class="project-columns"><div><h3>Configuration history</h3><p class="muted">Run settings are recorded automatically. Restore prepares the form with a new output directory; it does not start a job.</p><div class="project-tools"><label>Save current settings<select id="projectSnapshotWorkflow"><option value="data">Data preparation</option><option value="training">Training</option><option value="prediction">Prediction</option><option value="reactionPatterns">Reaction patterns</option></select></label><label>Label<input id="projectSnapshotLabel" placeholder="e.g. baseline · 30 eV"></label><button id="projectSnapshot">Save Snapshot</button></div><details id="projectComparison" hidden><summary>Configuration changes</summary><pre id="projectComparisonText"></pre></details><div id="projectConfigurations"></div><button id="projectMoreConfigurations" hidden>Show More</button></div><div><div class="section-title"><h3>Tracked files</h3><div class="actions"><button id="projectDiscover">Discover Project Files</button><button id="projectAddFiles">Track Files</button></div></div><label class="check"><input id="projectPinned" type="checkbox">Favorites only</label><p class="muted">References to your original files. Missing or modified files are marked; data is not copied.</p><div id="projectFiles"></div><button id="projectMoreFiles" hidden>Show More</button></div></div></div></section>`;}
function css(){return `#projectHome{border-top:3px solid var(--accent)}#projectPath,.project-item small{overflow-wrap:anywhere}#projectComparisonText{white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto}#projectNotes{display:block;width:100%;box-sizing:border-box;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--border);padding:10px;font:inherit}.project-tools{display:flex;align-items:end;flex-wrap:wrap;gap:10px;margin:12px 0}.project-tools label{flex:1;min-width:140px}.project-columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.project-item{padding:12px 0;border-top:1px solid var(--border)}.project-item strong,.project-item small{display:block;margin-bottom:6px}.project-item .actions{flex-wrap:wrap;justify-content:flex-start}.project-item button{font-size:11px;padding:5px 9px}#projectRecent{flex-wrap:wrap;justify-content:flex-start}#projectHome [hidden]{display:none!important}@media(max-width:1100px){.project-columns{grid-template-columns:1fr}}`;}
function client(){
  const el=id=>document.getElementById(id),clone=value=>JSON.parse(JSON.stringify(value));
  let notesDirty=false,project=null,ready=false,restoring=false,switching=false,timer,lastDraft='',configLimit=20,fileLimit=20;
  const quickIds=['quickTrainDir','quickValDir','quickCheckpoint','quickTrainOutput','quickEpochs','quickBatchSize','quickLearningRate','quickPredictModel','quickSmiles','quickAdduct','quickCE'];
  const names={data:'Data preparation',training:'Training',prediction:'Prediction',reactionPatterns:'Reaction patterns'};
  function formValues(form){const value={};for(const input of form.elements)if(input.name){if(input.type==='radio'){if(input.checked)value[input.name]=input.value;}else value[input.name]=input.name==='adductType'?(input.value||input.dataset.restoreValue||''):input.type==='checkbox'?input.checked:input.type==='number'?(input.value===''?'':Number(input.value)):input.value;}return value;}
  function capture(){return clone({data:getConfig(),training:getTrainingConfig(),prediction:formValues(el('predictForm')),reactionPatterns:{value:reactionPatternModel,path:reactionPatternPath},quick:Object.fromEntries(quickIds.map(id=>[id,el(id).value]))});}
  const baseline=capture();
  function send(type,extra={}){vscode.postMessage({type,root:project?.root,...extra});}
  function status(text){el('projectSaveStatus').textContent=text;}
  function flush(){clearTimeout(timer);if(!ready||!project||restoring||switching)return;const drafts=capture(),raw=JSON.stringify(drafts);if(raw===lastDraft)return;lastDraft=raw;status('Saving form settings…');send('project/drafts',{drafts});}
  function schedule(){if(!ready||!project||restoring||switching)return;clearTimeout(timer);timer=setTimeout(flush,700);}
  document.addEventListener('input',event=>{if(!event.target.closest('#projectHome'))schedule();});
  document.addEventListener('change',event=>{if(!event.target.closest('#projectHome'))schedule();});
  document.addEventListener('click',event=>{if(!event.target.closest('#projectHome'))schedule();});
  function dispatch(data){window.dispatchEvent(new MessageEvent('message',{data}));}
  function apply(workflow,config){
    if(workflow==='data')dispatch({type:'config',config});
    if(workflow==='training')dispatch({type:'trainingConfig',config});
    if(workflow==='reactionPatterns'){dispatch({type:'reactionPatternSet',value:config.value||config,path:config.path||''});resetVisualBuilder();el('visualBuilder').hidden=true;}
    if(workflow==='prediction'){
      const fields={input:'batchInput',outputDir:'batchOutputDir',db:'batchDb',maxSamples:'batchMaxSamples',specIdColumn:'batchSpecIdColumn',smilesColumn:'batchSmilesColumn',precursorMzColumn:'batchPrecursorMzColumn',adductTypeColumn:'batchAdductTypeColumn',collisionEnergyColumn:'batchCollisionEnergyColumn',instrumentColumn:'batchInstrumentColumn',overwrite:'batchOverwrite',numWorkers:'batchNumWorkers',chunkSize:'batchChunkSize',keepTemp:'batchKeepTemp'};
      const values={...config};for(const [key,field]of Object.entries(fields))if(key in config)values[field]=config[key];
      setFormConfig(el('predictForm'),values);resetPredictModelState();
      if(config.adductType)el('predictForm').elements.adductType.dataset.restoreValue=config.adductType;
      el('predictResultSection').hidden=true;el('predictResultEmpty').hidden=false;el('predictMoleculePreview').hidden=true;lastPredictTree=null;lastPredictionResult=null;
    }
  }
  function restoreAll(message){
    restoring=true;clearTimeout(timer);
    const values=clone(baseline),saved=message.drafts||{};
    if(message.project){for(const workflow of ['data','training'])values[workflow].outputDir=message.outputs[workflow];values.prediction.batchOutputDir=message.outputs.prediction;values.quick.quickTrainOutput=message.outputs.training;}
    for(const workflow of Object.keys(names))apply(workflow,saved[workflow]||values[workflow]);
    for(const id of quickIds){el(id).value=(saved.quick||values.quick)[id]??'';el(id).dispatchEvent(new Event('input',{bubbles:true}));}
    lastDraft=JSON.stringify(capture());restoring=false;ready=true;
    status(message.project?'Form settings restored. Changes are saved automatically.':'Project closed.');
  }
  function button(label,action){const node=document.createElement('button');node.type='button';node.textContent=label;node.onclick=action;return node;}
  function item(title,description){const row=document.createElement('article');row.className='project-item';const heading=document.createElement('strong'),detail=document.createElement('small'),actions=document.createElement('div');heading.textContent=title;detail.textContent=description;actions.className='actions';row.append(heading,detail,actions);return {row,actions};}
  function matches(text,workflows){const query=el('projectSearch').value.trim().toLowerCase(),workflow=el('projectWorkflow').value;return (!query||text.toLowerCase().includes(query))&&(!workflow||workflows.includes(workflow));}
  function renderLists(){
    const configs=el('projectConfigurations'),files=el('projectFiles');configs.replaceChildren();files.replaceChildren();if(!project)return;
    const selected=project.configurations.filter(config=>matches(config.label+' '+config.workflow+' '+(config.sourcePath||''),[config.workflow]));
    for(const saved of selected.slice(0,configLimit)){
      const {row,actions}=item(saved.label,names[saved.workflow]+' · '+new Date(saved.createdAt).toLocaleString()+' · '+saved.reason);
      actions.append(button('Restore Settings',()=>{flush();send('project/restore',{id:saved.id});}),button('View JSON',()=>send('project/viewConfig',{id:saved.id})),button('Compare Previous',()=>send('project/compare',{id:saved.id})));
      if(saved.jobId)actions.append(button('Open Run',()=>dispatch({type:'workbench/jobSelected',jobId:saved.jobId})));
      if(saved.files?.length){const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent=saved.files.length+' file references at this point';details.append(summary);for(const file of saved.files){const line=document.createElement('small');line.textContent=file.role+': '+file.path+(file.recorded?.exists?'':' (not present when recorded)');details.append(line);}row.append(details);}
      configs.append(row);
    }
    if(!selected.length)configs.textContent='No saved settings yet. Save a snapshot or run a workflow.';
    el('projectMoreConfigurations').hidden=selected.length<=configLimit;
    const selectedFiles=project.files.filter(file=>(!el('projectPinned').checked||file.pinned)&&matches(file.path+' '+file.roles.join(' '),file.workflows)).sort((a,b)=>Number(b.pinned)-Number(a.pinned)||b.lastSeen.localeCompare(a.lastSeen));
    for(const file of selectedFiles.slice(0,fileLimit)){
      const state=!file.current.exists?'Missing':file.changed?'Modified since last recorded':file.current.directory?'Directory':(file.current.size/1024/1024).toFixed(2)+' MB';
      const {row,actions}=item((file.pinned?'★ ':'')+file.path.split(/[\\/]/).at(-1),file.path+' · '+state+' · '+file.roles.join(', '));
      if(!file.current.exists||file.changed)row.querySelector('small').classList.add('warning');
      actions.append(button(file.pinned?'Unfavorite':'Favorite',()=>send('project/pin',{id:file.id})),button('Copy Path',()=>send('project/file',{id:file.id,action:'copy'})));
      if(file.current.exists)actions.append(button('Reveal',()=>send('project/file',{id:file.id,action:'reveal'})));
      if(file.current.exists&&!file.current.directory&&/\.(json|txt|csv|tsv|smi|smiles|md|log)$/i.test(file.path))actions.append(button('Open',()=>send('project/file',{id:file.id,action:'open'})));
      if(file.configurationId)actions.append(button('Restore Related Settings',()=>{flush();send('project/restore',{id:file.configurationId});}));
      files.append(row);
    }
    if(!selectedFiles.length)files.textContent='Files appear as you configure workflows. You can also track files manually.';
    el('projectMoreFiles').hidden=selectedFiles.length<=fileLimit;
  }
  function render(message){
    project=message.project;
    el('projectTitle').textContent=project?project.name:'Your research project';el('projectPath').textContent=project?project.root:'Choose a directory to keep settings, files and runs together.';
    for(const id of ['projectContent','projectReveal','projectClose'])el(id).hidden=!project;
    const recent=el('projectRecent');recent.replaceChildren();for(const entry of message.recent||[])if(entry.path!==project?.root){const b=button(entry.name,()=>change('project/open',{path:entry.path}));b.title=entry.path;recent.append(b);}
    if(project){el('projectSummary').textContent=project.configurations.length+' saved configurations · '+project.files.length+' tracked files · '+project.files.filter(file=>!file.current.exists).length+' missing';if(message.restore||!notesDirty){el('projectName').value=project.name;el('projectNotes').value=project.notes;}}
    renderLists();
  }
  function change(type,extra={}){if(switching)return;clearTimeout(timer);switching=true;el('projectOpen').disabled=true;el('projectClose').disabled=true;send(type,{drafts:ready?capture():undefined,...extra});}
  el('projectOpen').onclick=()=>change('project/open');el('projectClose').onclick=()=>change('project/close');el('projectReveal').onclick=()=>send('project/reveal');
  el('projectRuns').onclick=()=>dispatch({type:'workbench/navigate',page:'jobs'});
  el('projectRefresh').onclick=()=>{flush();send('project/refresh');};el('projectAddFiles').onclick=()=>send('project/addFiles');el('projectDiscover').onclick=()=>{status('Looking for datasets, models and configurations in the project…');send('project/discover');};
  for(const id of ['projectName','projectNotes'])el(id).oninput=()=>{notesDirty=true;};
  el('projectSaveNotes').onclick=()=>send('project/notes',{name:el('projectName').value,notes:el('projectNotes').value});
  el('projectSnapshot').onclick=()=>{flush();const workflow=el('projectSnapshotWorkflow').value;send('project/snapshot',{workflow,config:capture()[workflow],label:el('projectSnapshotLabel').value||names[workflow]+' settings'});};
  for(const id of ['projectSearch','projectWorkflow','projectPinned'])el(id).addEventListener('input',()=>{configLimit=fileLimit=20;renderLists();});
  el('projectMoreConfigurations').onclick=()=>{configLimit+=20;renderLists();};el('projectMoreFiles').onclick=()=>{fileLimit+=20;renderLists();};
  window.addEventListener('message',event=>{
    const message=event.data;
    if(message.type==='project/state'){if(message.restore){notesDirty=false;el('projectComparison').hidden=true;switching=false;el('projectOpen').disabled=false;el('projectClose').disabled=false;el('projectError').textContent='';}render(message);if(message.restore)restoreAll(message);}
    if(message.type==='project/cancelled'||message.type==='project/error'){switching=false;el('projectOpen').disabled=false;el('projectClose').disabled=false;if(message.error){el('projectError').textContent=message.error;lastDraft='';}}
    if(message.type==='project/draftsSaved'&&message.root===project?.root)status('Form settings saved · '+new Date(message.at).toLocaleTimeString());
    if(message.type==='project/discovered')status('Found '+message.count+' files.'+(message.truncated?' Discovery reached its limit. Track Files can add specific files.':''));
    if(message.type==='project/notesSaved'){notesDirty=false;status('Project notes saved.');}
    if(message.type==='project/comparison'){el('projectComparison').hidden=false;el('projectComparison').open=true;el('projectComparisonText').textContent=message.before+' → '+message.after+'\n\n'+(message.changes.length?message.changes.map(item=>item.field+'\n  Before: '+JSON.stringify(item.before)+'\n  After:  '+JSON.stringify(item.after)).join('\n\n'):'No configuration differences.');}
    if(message.type==='project/restore'){restoring=true;apply(message.workflow,clone(baseline[message.workflow]));apply(message.workflow,message.config);restoring=false;dispatch({type:'workbench/navigate',page:message.workflow==='prediction'?'predictSingle':message.workflow});status('Restored '+message.label+'. Review the settings before running.');schedule();}
    if(['picked','library/picked','config','trainingConfig','predictConfigLoaded','predictBatchPicked','reactionPatternSet','reactionPatternLoaded','reactionPatternSetSaved','parameters/loaded'].includes(message.type))schedule();
    if(!restoring&&project&&['parameters/loaded','reactionPatternLoaded','library/picked','picked','predictBatchPicked'].includes(message.type)){const paths=message.paths||(message.path?[message.path]:message.value?[message.value]:[]);if(paths.length)send('project/track',{paths,workflow:message.target||message.form});}
    if(!restoring&&project&&['config','trainingConfig','predictConfigLoaded','reactionPatternSet','reactionPatternSetSaved','predictConfigSaved'].includes(message.type)){
      const workflow={config:'data',trainingConfig:'training',predictConfigLoaded:'prediction',reactionPatternSet:'reactionPatterns',reactionPatternSetSaved:'reactionPatterns',predictConfigSaved:'prediction'}[message.type];
      send('project/snapshot',{workflow,config:capture()[workflow],label:names[workflow]+' · '+(message.path?.split(/[\\/]/).at(-1)||'imported settings'),reason:message.type.endsWith('Saved')?'exported':'loaded',sourcePath:message.path});
    }
  });
  window.addEventListener('pagehide',flush);
  vscode.postMessage({type:'project/ready'});
}
module.exports={html,css,script:()=>`(${client.toString()})();`};
