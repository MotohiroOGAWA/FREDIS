const defaults=require('./defaults');
const parameterEditor=require('./parameter-editor');
const fs=require('fs'),path=require('path'),vscode=require('vscode');
const {spawn}=require('child_process');
function buildArgs(mode,c){
 const args=['-m','fredis.cli','train','trainable-ids','--output-dir',c.outputDir];
 if(mode!=='extract')throw new Error('Unsupported eligibility operation.');
 {
  if(!c.input)throw new Error('Select the input MSDataset.');args.push('--input',c.input,'--spec-id-column',c.specIdColumn||'SpecID');
  for(const [key,flag]of [['smilesColumn','smiles-column'],['adductTypeColumn','adduct-type-column'],['collisionEnergyColumn','collision-energy-column'],['precursorMzColumn','precursor-mz-column']])if(c[key])args.push('--'+flag,c[key]);
  if(c.fragmenterParams)args.push('--params-json',JSON.stringify(c.fragmenterParams));if(c.symbols?.length)args.push('--symbols-json',JSON.stringify(c.symbols));
  if(c.modelConfig)args.push('--model-architecture-json',JSON.stringify({model_config:c.modelConfig}));
  for(const [key,flag,def]of [['workers','workers',1],['batchSize','batch-size',8],['minimumRelativeIntensity','minimum-relative-intensity',0],['maxUniqueFragmentSmiles','max-unique-fragment-smiles',10000],['maxReactionCombinations','max-reaction-combinations',10000]])args.push('--'+flag,String(c[key]??def));
  if(c.resume)args.push('--resume');if(c.keepExport)args.push('--keep-export');
 }return args;
}
function shellDisplay(program,args){return [program,...args].map(value=>/^[A-Za-z0-9_./:=,-]+$/.test(value)?value:"'"+String(value).replace(/'/g,"'\\''")+"'").join(' ');}
function attach(panel,context,rootProvider){
 let active=null;const post=data=>panel.webview.postMessage(data);
 function stop(){if(active?.child){if(process.platform==='win32')spawn('taskkill',['/pid',String(active.child.pid),'/T','/F']);else try{process.kill(-active.child.pid,'SIGTERM');}catch{}}}
 panel.onDidDispose(stop);
 panel.webview.onDidReceiveMessage(async m=>{
  if(!m?.type?.startsWith('trainable/'))return;
  try{
   if(m.type==='trainable/stop'){stop();return;}
   if(m.type==='trainable/pick'){
    const chosen=await vscode.window.showOpenDialog({canSelectFiles:m.kind!=='folder',canSelectFolders:m.kind!=='file',canSelectMany:false});
    if(chosen?.length)post({type:'trainable/picked',target:m.target,paths:chosen.map(uri=>uri.fsPath)});return;
   }
   if(m.type==='trainable/copyCommand'){
    if(!m.config.outputDir)throw new Error('Select an output directory.');
    const python=vscode.workspace.getConfiguration('fredis').get('pythonPath','python'),command=shellDisplay(python,buildArgs(m.mode,m.config));
    await vscode.env.clipboard.writeText(command);post({type:'trainable/copied',command});return;
   }
   if(m.type!=='trainable/run')return;
   if(active)throw new Error('An eligibility operation is already running.');
   const c=m.config;if(!c.outputDir)throw new Error('Select an output directory.');const args=buildArgs(m.mode,c),root=rootProvider(context),python=vscode.workspace.getConfiguration('fredis').get('pythonPath','python');
   const child=spawn(python,args,{cwd:root,detached:process.platform!=='win32',stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)}});
   const operation={child,mode:m.mode,c};active=operation;let log='',pending='';post({type:'trainable/status',mode:m.mode,status:'running'});
   function append(chunk){log=(log+chunk).slice(-64000);post({type:'trainable/log',mode:m.mode,text:log});}
   child.stderr.on('data',append);child.stdout.on('data',chunk=>{append(chunk);pending+=chunk;const lines=pending.split(/\r?\n/);pending=lines.pop();for(const line of lines)try{const event=JSON.parse(line);if(event.event==='trainable_progress')post({type:'trainable/progress',...event});}catch{}});
   const fail=error=>{if(active!==operation)return;active=null;post({type:'trainable/status',mode:m.mode,status:'failed',error});};
   child.on('error',error=>fail(error.message));child.on('close',async(code,signal)=>{if(active!==operation)return;if(code!==0){fail(signal?'Stopped. Resume extraction from the same output directory.':log.slice(-4000)||'Operation failed');return;}
    try{const output=path.resolve(root,c.outputDir),stats=JSON.parse(await fs.promises.readFile(path.join(output,'stats.json'),'utf8'));active=null;post({type:'trainable/result',mode:m.mode,outputDir:output,stats});}catch(error){fail(error.message);}
   });
  }catch(error){post({type:'trainable/status',mode:m.mode,status:'failed',error:error.message});}
 });
}
function html(){return `<div class="page-heading"><h1>Trainable Spectra</h1><p>Export verified FREDIS trainable SpecIDs for use with other tools.</p></div><div class="trainable-columns"><section><h2>FREDIS trainable SpecIDs</h2><p>Runs native preparation, forward, loss and backward. No optimizer updates.</p><form id="trainableExtractForm"><label>Input MSDataset<div class="path"><input name="input" data-path-kind="file" required><button type="button" data-trainable-pick="extractInput" data-kind="file">Browse</button></div></label><button type="button" id="trainableUseData">Use Training Data input and mappings</button><details><summary>Column mapping</summary>${[['specIdColumn','SpecID'],['smilesColumn','SMILES'],['adductTypeColumn','AdductType'],['collisionEnergyColumn','CollisionEnergy'],['precursorMzColumn','PrecursorMZ']].map(([key,value])=>`<label>${key}<input name="${key}" value="${value}"></label>`).join('')}</details><section><h2>Fragmenter</h2><p class="muted">Edit reaction patterns, adduct rules and fragmentation parameters for this extraction. Symbols come from Training Data. A compact model is used by default (16-dimensional features, one layer).</p><button type="button" id="trainableUseFragmenter">Use Training Data fragmenter</button>${parameterEditor.importHtml('trainable')}<div class="grid"><label>Max reaction combinations<input name="maxReactionCombinations" type="number" min="-1" step="1" value="10000"></label><label>Max unique fragment SMILES<input name="maxUniqueFragmentSmiles" type="number" min="-1" step="1" value="10000"></label></div><p class="muted">Limits apply to preparation; -1 means unlimited. Lower limits or reaction counts can change which spectra qualify.</p>${defaults.html('trainable')}</section><label class="check"><input name="useArchitecture" type="checkbox">Use current Training architecture</label><div class="grid"><label>Workers<input name="workers" type="number" min="1" value="1"></label><label>Batch size<input name="batchSize" type="number" min="1" value="8"></label></div><label>Output directory<div class="path"><input name="outputDir" data-path-kind="folder" required><button type="button" data-trainable-pick="extractOutput" data-kind="folder">Browse</button></div></label><label class="check"><input name="resume" type="checkbox">Resume an interrupted extraction</label><label class="check"><input name="keepExport" type="checkbox">Keep intermediate exported data</label><button type="button" id="trainableCopyCommand">Copy command</button><button id="trainableExtractRun" class="primary" type="submit">Verify and export SpecIDs</button></form><p id="trainableExtractStatus" role="status"></p><progress id="trainableProgress" max="100" value="0" hidden></progress><div id="trainableExtractSummary"></div></section></div><section><div class="actions"><button id="trainableStop" disabled>Stop</button></div><details><summary>Processing log</summary><pre id="trainableLog" role="log"></pre></details></section>`;}
function client(){
 const el=id=>document.getElementById(id),send=m=>vscode.postMessage(m),extract=el('trainableExtractForm');
 for(const key of ['maxReactionCombinations','maxUniqueFragmentSmiles'])extract.elements[key].value=initialTrainable[key]??10000;
 function busy(value){el('trainableExtractRun').disabled=value;el('trainableStop').disabled=!value;}
 el('trainableStop').onclick=()=>send({type:'trainable/stop'});
 for(const button of document.querySelectorAll('[data-trainable-pick]'))button.onclick=()=>send({type:'trainable/pick',target:button.dataset.trainablePick,kind:button.dataset.kind});
 el('trainableUseData').onclick=()=>{const c=getConfig();for(const key of ['input','smilesColumn','adductTypeColumn','collisionEnergyColumn','precursorMzColumn'])if(c[key])extract.elements[key].value=c[key];};
 function extractionConfig(){const c=getConfig(),value={...c,fragmenterParams:window.getTrainableFragmenterConfig(),input:extract.elements.input.value,outputDir:extract.elements.outputDir.value};for(const key of ['specIdColumn','smilesColumn','adductTypeColumn','collisionEnergyColumn','precursorMzColumn'])value[key]=extract.elements[key].value;for(const key of ['workers','batchSize','maxReactionCombinations','maxUniqueFragmentSmiles'])value[key]=Number(extract.elements[key].value);for(const key of ['resume','keepExport'])value[key]=extract.elements[key].checked;if(extract.elements.useArchitecture.checked)value.modelConfig=getTrainingConfig().modelConfig;return value;}
 extract.onsubmit=e=>{e.preventDefault();busy(true);send({type:'trainable/run',mode:'extract',config:extractionConfig()});};
 el('trainableCopyCommand').onclick=()=>send({type:'trainable/copyCommand',mode:'extract',config:extractionConfig()});
 function summary(host,stats){host.replaceChildren();for(const [label,value]of [['Input spectra',stats.total_records],['Trainable',stats.trainable],['Excluded',stats.excluded],...Object.entries(stats.exclusion_reasons||{})]){const p=document.createElement('p');p.textContent=label+': '+value;host.append(p);}}
 window.addEventListener('message',event=>{const m=event.data;if(m.type==='trainable/picked'){const targets={extractInput:[extract,'input'],extractOutput:[extract,'outputDir']},target=targets[m.target];if(target)target[0].elements[target[1]].value=m.paths[0];}
  if(m.type==='trainable/copied')el('trainableExtractStatus').textContent='Command copied.';
  if(m.type==='trainable/status'){busy(m.status==='running');el('trainableExtractStatus').textContent=m.error||m.status;}
  if(m.type==='trainable/log')el('trainableLog').textContent=m.text;
  if(m.type==='trainable/progress'){const p=el('trainableProgress');p.hidden=false;p.max=Math.max(m.total,1);p.value=m.completed;el('trainableExtractStatus').textContent=m.completed+' / '+m.total+' spectra checked · '+m.trainable+' trainable';}
  if(m.type==='trainable/result'){busy(false);const host=el('trainableExtractSummary');summary(host,m.stats,m.mode);el('trainableExtractStatus').textContent='Completed · '+m.outputDir;const button=document.createElement('button');button.textContent='Reveal output';button.onclick=()=>send({type:'library/open',path:m.outputDir});host.append(button);}
 });
}
module.exports={buildArgs,shellDisplay,attach,html,script:()=>`(${client.toString()})();`};
