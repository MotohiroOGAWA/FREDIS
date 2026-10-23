const assert=require('assert/strict'),fs=require('fs'),os=require('os'),path=require('path'),Module=require('module'),{JSDOM,VirtualConsole}=require('jsdom');
const root=path.resolve(__dirname,'../..'),temporary=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-training-config-'));
const source=path.join(temporary,'fragment-tree.training-config.pft'),savedFile=path.join(temporary,'saved.training-config.pft');
const fixture=require('./fixtures'),originalLoad=Module._load;
const vscode={workspace:{workspaceFolders:[{uri:{fsPath:root}}],getConfiguration:()=>({get:(key,fallback)=>fallback})},Uri:{file:fsPath=>({fsPath})},window:{showSaveDialog:async()=>({fsPath:path.join(temporary,'saved')})}};
Module._load=function(name,...args){return name==='vscode'?vscode:originalLoad.call(this,name,...args);};
const ext=require('../src/extension');
for(const name of ['./workbench/project','./features/smarts-search/editor','./features/spectrum-prediction/editor','./workbench/parameter-service','./workbench/services','./workbench/defaults','./workbench/dataset-upload','./webview-path-drop','./features/reaction-pattern-analysis/editor'])require(path.join(__dirname,'../src',name)).attach=()=>{};
const workbench=require('../src/workbench/panel');let openedPage;workbench.attach=(panel,context,root,output,page)=>{openedPage=page;};
require('../src/features/model-architecture/editor').attachWorkbench=()=>{};
require('../src/features/spectrum-viewer/workbench').configure=()=>{};
require('../src/features/spectrum-viewer/extension').createEmbeddedProvider=()=>({resolveEmbeddedEditor:async()=>{},dispose(){}});
(async()=>{
 let dom;
 try{
  const pkg=require('../package.json'),editor=pkg.contributes.customEditors.find(entry=>entry.viewType==='fredis.trainingConfigEditor');
  assert(editor.selector.some(entry=>entry.filenamePattern==='*.training-config.pft'));assert.equal(editor.priority,'default');assert(pkg.activationEvents.includes('onCustomEditor:fredis.trainingConfigEditor'));
  const config={schema:'fredis.training-config',application:'fragment-tree-training',datasetDir:'../prepared',outputDir:'run',epochs:17,batchSize:8,lr:.0002,dropout:.3,modelConfig:fixture.model,patternArchitecture:{mode:'individual',shared:{}}};
  fs.writeFileSync(source,JSON.stringify(config));
  const handlers=[],panel={webview:{cspSource:'test',onDidReceiveMessage:fn=>handlers.push(fn),postMessage(){}},onDidDispose(){}};
  const provider=new ext.TrainingConfigEditorProvider({extensionPath:path.resolve(__dirname,'..')},{appendLine(){}});
  await provider.resolveCustomEditor(await provider.openCustomDocument({fsPath:source}),panel);
  assert.equal(openedPage,'training');assert.match(panel.title,/Training/);assert(panel.webview.html.includes('trainingDocumentPending'));
  const errors=[],messages=[],vc=new VirtualConsole();vc.on('jsdomError',error=>errors.push(error.message));
  dom=new JSDOM(panel.webview.html,{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(JSON.parse(JSON.stringify(m))),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){}disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:value=>value};}});
  const w=dom.window,d=w.document,snapshot=()=>{d.getElementById('trainingCopyCommand').click();return messages.filter(m=>m.type==='copyTrainingCommand').at(-1).config;};
  assert.equal(d.getElementById('trainingForm').hidden,false);assert.equal(snapshot().epochs,17);assert.equal(snapshot().batchSize,8);assert.equal(snapshot().dropout,.3);assert.equal(snapshot().datasetDir,path.resolve(temporary,'../prepared'));assert.equal(snapshot().outputDir,path.join(temporary,'run'));assert.equal(snapshot().patternArchitecture.mode,'individual');
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'project/state',project:null,recent:[],restore:true,drafts:{training:{...config,epochs:999,datasetDir:'/unrelated'}}}}));
  assert.equal(snapshot().epochs,17);assert.equal(snapshot().datasetDir,path.resolve(temporary,'../prepared'));assert.equal(d.getElementById('trainingForm').hidden,false);
  await handlers.at(-1)({type:'saveTrainingConfig',config:snapshot()});assert(fs.existsSync(savedFile));assert.equal(JSON.parse(fs.readFileSync(savedFile)).epochs,17);
  await new Promise(resolve=>setTimeout(resolve,50));assert.deepEqual(errors,[]);
  console.log('Training configuration file association, Training-page navigation, relative paths, settings restoration, startup isolation and .pft saving passed.');
 }finally{dom?.window.close();fs.rmSync(temporary,{recursive:true,force:true});Module._load=originalLoad;}
})().catch(error=>{console.error(error);process.exitCode=1;});
