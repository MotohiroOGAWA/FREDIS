const assert=require('assert/strict'),fs=require('fs'),path=require('path'),os=require('os'),Module=require('module');
const {EventEmitter}=require('events');
const {JSDOM,VirtualConsole}=require('jsdom');
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-projects-'));
const memory=()=>{const values=new Map();return {get:(key,fallback)=>values.has(key)?values.get(key):fallback,update:async(key,value)=>values.set(key,value)};};
const workspaceState=memory(),globalState=memory(),opened=[],children=[];
const mock={workspace:{workspaceFolders:[{uri:{fsPath:path.resolve(__dirname,'../..')}}],getConfiguration:()=>({get:(key,fallback)=>fallback})},window:{showInformationMessage(){},showOpenDialog:async()=>undefined,showTextDocument:async uri=>opened.push(uri.fsPath)},commands:{executeCommand:async(...args)=>opened.push(args)},env:{clipboard:{writeText:async value=>opened.push(value)}},Uri:{file:fsPath=>({fsPath})}};
const original=Module._load;
Module._load=function(name,parent,main){if(name==='vscode')return mock;if(name==='child_process')return {spawn:()=>{const child=new EventEmitter();child.pid=12345;child.unref=()=>{};child.stdout=new EventEmitter();child.stderr=new EventEmitter();child.stdin=new EventEmitter();child.stdin.end=()=>{};children.push(child);return child;}};return original.call(this,name,parent,main);};
const projects=require('../src/workbench/project'),store=projects.store,jobs=require('../src/workbench/panel');
const extension=require('../src/extension'),preset=require('./fixtures').model;
function makePanel(){const listeners=[],disposers=[],posted=[];return {posted,webview:{postMessage:value=>posted.push(JSON.parse(JSON.stringify(value))),onDidReceiveMessage:fn=>listeners.push(fn)},onDidDispose:fn=>disposers.push(fn),send:async value=>{for(const fn of listeners)await fn(value);},dispose:()=>disposers.forEach(fn=>fn())};}
(async()=>{
 let dom;
 try{
  const a=path.join(scratch,'project A'),b=path.join(scratch,'project B');fs.mkdirSync(a);fs.mkdirSync(b);
  const root=store.open(a);assert.equal(root,a);assert.equal(store.open(a),a);assert.equal(store.read(a).version,1);
  fs.mkdirSync(path.join(a,'node_modules'));fs.writeFileSync(path.join(a,'node_modules','ignored.json'),'{}');fs.writeFileSync(path.join(a,'existing-model.pt'),'model');await store.discover(a);assert(store.read(a).files.some(item=>item.path.endsWith('existing-model.pt')));assert(!store.read(a).files.some(item=>item.path.includes('node_modules')));
  const data=path.join(a,'data.msds');fs.writeFileSync(data,'original');
  const id=store.snapshot(a,'data',{input:data,outputDir:path.join(a,'out'),validationRatio:0.1},{label:'Baseline',base:a});
  const next=store.snapshot(a,'data',{input:data,outputDir:path.join(a,'out'),validationRatio:0.2},{label:'More validation',base:a});
  store.drafts(a,{data:{input:data,validationRatio:0.8}},a);
  assert.equal(store.configuration(a,id).config.validationRatio,0.1);
  assert.equal(store.readDrafts(a).data.validationRatio,0.8);
  let resource=store.describe(a).files.find(item=>item.path===data);assert(!resource.changed);
  store.pin(a,resource.id);fs.appendFileSync(data,' changed');resource=store.describe(a).files.find(item=>item.path===data);assert(resource.changed);assert(resource.pinned);
  fs.unlinkSync(data);assert(!store.describe(a).files.find(item=>item.path===data).current.exists);
  assert.throws(()=>store.configuration(a,'../../outside'),/not found/);
  const context=projects.createContext({workspaceState,globalState,storageUri:{fsPath:path.join(scratch,'storage')}}),panel=makePanel();
  projects.attach(panel,context,()=>a);jobs.attach(panel,context,()=>a,{},'home');
  try{
   await panel.send({type:'project/open',path:a});assert.equal(projects.root(context),a);assert.equal(workspaceState.get('fredis.project'),a);
   let state=panel.posted.filter(item=>item.type==='project/state').at(-1);assert.equal(state.drafts.data.validationRatio,0.8);assert(state.outputs.training.startsWith(path.join(a,'runs')));
   await panel.send({type:'project/compare',root:a,id:next});const comparison=panel.posted.at(-1);assert.equal(comparison.type,'project/comparison');assert.equal(comparison.changes[0].field,'validationRatio');assert.equal(comparison.changes[0].before,0.1);
   await panel.send({type:'project/restore',root:a,id});const restored=panel.posted.at(-1);assert.equal(restored.type,'project/restore');assert.notEqual(restored.config.outputDir,path.join(a,'out'));assert.equal(store.configuration(a,id).config.outputDir,path.join(a,'out'));
   const batchId=store.snapshot(a,'prediction',{input:'/test/input.msds',outputDir:'/old',modelPath:'/test/model.pt'},{label:'Batch'});
   await panel.send({type:'project/restore',root:a,id:batchId});assert(panel.posted.at(-1).config.outputDir.startsWith(path.join(a,'runs')));assert(!('batchOutputDir' in panel.posted.at(-1).config));
   const predictionPanel=makePanel();
   require('../src/features/spectrum-prediction/editor').attach(predictionPanel,context,()=>a,{show(){},appendLine(){}});
   try{
    const run=predictionPanel.send({type:'predictBatchRun',config:{input:data,outputDir:path.join(a,'batch'),modelPath:'/model.pt'}});
    const child=children.at(-1);assert(child);child.stdout.emit('data',Buffer.from('partial prediction output\n'));child.emit('close',3,null);await run;
    const batchConfig=store.read(a).configurations.find(item=>item.label==='Batch prediction');assert(batchConfig);
    const savedJob=JSON.parse(fs.readFileSync(path.join(a,'.fredis/jobs',batchConfig.jobId+'.json')));
    assert.equal(savedJob.type,'batch-prediction');assert.equal(savedJob.status,'completed');assert.equal(savedJob.exitCode,3);assert(savedJob.warning);assert(fs.readFileSync(savedJob.logPath,'utf8').includes('partial prediction output'));
   }finally{predictionPanel.dispose();}
   const captured=projects.capture(context);
   const observed=jobs.observeJob(captured,{type:'preparation',workflow:'data',name:'A run',config:{input:'data.msds',outputDir:'out'},base:a,outputDir:'out'});
   await panel.send({type:'project/open',root:a,path:b,drafts:{data:{input:data,validationRatio:0.3}}});assert.equal(projects.root(context),b);assert.equal(store.readDrafts(a).data.validationRatio,0.3);
   observed.log('Finished in project A\n');observed.finish({code:0});
   assert.equal(JSON.parse(fs.readFileSync(path.join(a,'.fredis/jobs',observed.job.id+'.json'))).status,'completed');
   assert.equal(store.read(b).configurations.length,0);assert.equal(projects.root(captured),a);
   await panel.send({type:'workbench/ready'});assert.equal(panel.posted.filter(item=>item.type==='workbench/jobs').at(-1).jobs.length,0);
   await panel.send({type:'project/drafts',root:a,drafts:{data:{input:'wrong'}}});assert.equal(panel.posted.at(-1).type,'project/error');assert.deepEqual(store.readDrafts(b),{});
   await panel.send({type:'project/notes',root:b,name:'Study B',notes:'Next: compare collision energies'});assert.equal(store.read(b).name,'Study B');
   await panel.send({type:'project/open',root:b,path:a});await panel.send({type:'workbench/ready'});assert.equal(panel.posted.filter(item=>item.type==='workbench/jobs').at(-1).jobs.find(job=>job.id===observed.job.id).id,observed.job.id);
   await panel.send({type:'project/file',root:a,id:resource.id,action:'copy'});assert.equal(opened.at(-1),data);
   await panel.send({type:'project/close',root:a});assert.equal(projects.root(context),null);assert(fs.existsSync(path.join(a,'.fredis/project.json')));
   await panel.send({type:'project/open',path:a});assert.equal(projects.root(projects.createContext({workspaceState})),a);
  }finally{panel.dispose();}
  // Unsupported metadata must never be overwritten.
  fs.writeFileSync(path.join(b,'.fredis/project.json'),'{"schema":"future","version":99}');
  assert.throws(()=>store.open(b),/Unsupported/);assert.equal(JSON.parse(fs.readFileSync(path.join(b,'.fredis/project.json'))).version,99);
  const messages=[],errors=[],virtualConsole=new VirtualConsole();virtualConsole.on('jsdomError',error=>errors.push(error.detail?.stack||error.message));
  dom=new JSDOM(extension.workbenchHtml(require('./fixtures').dataConfig,{modelConfig:preset}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:value=>messages.push(JSON.parse(JSON.stringify(value))),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){}disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:value=>value};}});
  const w=dom.window,d=w.document,post=value=>w.dispatchEvent(new w.MessageEvent('message',{data:JSON.parse(JSON.stringify(value))}));
  assert.deepEqual(errors,[]);assert(messages.some(item=>item.type==='project/ready'));
  const values={data:{input:'/dataset.msds',outputDir:'/previous',...require('./fixtures').dataConfig},training:{modelConfig:preset,experimentName:'saved experiment',epochs:9},prediction:{modelPath:'/model.pt',batchInput:'/batch.msds',batchOutputDir:'/batch-out',smiles:'CCO',ce:35,adductType:'[M+H]+'},reactionPatterns:{value:{reaction_pattern_set:{name:'Saved patterns',patterns:[]}},path:''}};
  const descriptor=store.describe(a);descriptor.name='<img src=x onerror=alert(1)>';
  post({type:'project/state',project:descriptor,recent:[{name:'Study B',path:b}],restore:true,drafts:values,outputs:{data:'/runs/data',training:'/runs/train',prediction:'/runs/predict'}});
  assert.equal(d.querySelector('#projectTitle img'),null);assert.equal(d.getElementById('projectTitle').textContent,descriptor.name);
  assert.equal(d.getElementById('form').elements.input.value,'/dataset.msds');assert.equal(d.getElementById('trainingForm').elements.epochs.value,'9');assert.equal(d.getElementById('predictForm').elements.batchInput.value,'/batch.msds');assert.equal(d.getElementById('reactionPatternSetName').value,'Saved patterns');
  const before=messages.filter(item=>item.type==='project/snapshot').length;
  await new Promise(resolve=>setTimeout(resolve,800));assert.equal(messages.filter(item=>item.type==='project/snapshot').length,before,'restoring does not create duplicate history');
  const input=d.getElementById('form').elements.input;input.value='/new.msds';input.dispatchEvent(new w.Event('input',{bubbles:true}));await new Promise(resolve=>setTimeout(resolve,850));
  const draft=messages.filter(item=>item.type==='project/drafts').at(-1);assert.equal(draft.root,a);assert.equal(draft.drafts.data.input,'/new.msds');assert.equal(draft.drafts.prediction.smiles,'CCO');assert.equal(draft.drafts.prediction.adductType,'[M+H]+');
  d.getElementById('projectSnapshotLabel').value='Baseline 30 eV';d.getElementById('projectSnapshot').click();assert.equal(messages.at(-1).type,'project/snapshot');assert.equal(messages.at(-1).config.input,'/new.msds');
  d.getElementById('projectNotes').value='Not yet saved';d.getElementById('projectNotes').dispatchEvent(new w.Event('input',{bubbles:true}));d.getElementById('projectSearch').focus();post({type:'project/state',project:descriptor,recent:[],restore:false});assert.equal(d.getElementById('projectNotes').value,'Not yet saved');
  const runCount=messages.filter(item=>['run','runTraining','predictBatchRun','predictSpectrum'].includes(item.type)).length;
  post({type:'project/restore',workflow:'training',config:{modelConfig:preset,epochs:17,outputDir:'/new-output'},label:'Earlier training'});
  assert.equal(d.getElementById('trainingForm').elements.epochs.value,'17');assert(!d.getElementById('trainingForm').hidden);assert.equal(messages.filter(item=>['run','runTraining','predictBatchRun','predictSpectrum'].includes(item.type)).length,runCount);
  post({type:'project/restore',workflow:'prediction',config:{input:'/normalized.msds',outputDir:'/new-batch',modelPath:'/model.pt'},label:'Batch'});assert.equal(d.getElementById('predictForm').elements.batchInput.value,'/normalized.msds');
  d.getElementById('projectClose').click();assert.equal(messages.at(-1).type,'project/close');assert.equal(messages.at(-1).drafts.training.epochs,17);
  post({type:'project/state',project:null,recent:[],restore:true});assert(d.getElementById('projectContent').hidden);assert.notEqual(d.getElementById('form').elements.input.value,'/new.msds');assert.deepEqual(errors,[]);
  await new Promise(resolve=>setTimeout(resolve,20));assert.deepEqual(errors,[]);
  console.log('Projects: durable snapshots/drafts, file changes/pins, comparison, safe restore, switching/stale messages, job isolation/completion, metadata protection and Home UI passed.');
 }finally{dom?.window.close();fs.rmSync(scratch,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
