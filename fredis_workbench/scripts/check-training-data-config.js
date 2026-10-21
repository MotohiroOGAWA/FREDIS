const assert=require('assert/strict'),fs=require('fs'),os=require('os'),path=require('path'),Module=require('module'),{JSDOM,VirtualConsole}=require('jsdom');
const root=path.resolve(__dirname,'../..'),temporary=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-training-data-config-'));
const source=path.join(temporary,'fragment-tree.training-data-config.pft'),savedFile=path.join(temporary,'saved.training-data-config.pft');
const fixture=require('./fixtures'),originalLoad=Module._load;
const vscode={workspace:{workspaceFolders:[{uri:{fsPath:root}}],getConfiguration:()=>({get:(key,fallback)=>fallback})},Uri:{file:fsPath=>({fsPath})},window:{showSaveDialog:async()=>({fsPath:path.join(temporary,'saved')})}};
Module._load=function(name,...args){return name==='vscode'?vscode:originalLoad.call(this,name,...args);};
const ext=require('../src/extension');
const modules=['./workbench/project','./features/smarts-search/editor','./features/spectrum-prediction/editor','./workbench/parameter-service','./workbench/services','./workbench/defaults','./workbench/dataset-upload','./webview-path-drop','./features/reaction-pattern-analysis/editor'];
for(const name of modules){const feature=require(path.join(__dirname,'../src',name));feature.attach=()=>{};}
const workbench=require('../src/workbench/panel');let openedPage;workbench.attach=(panel,context,root,output,page)=>{openedPage=page;};
require('../src/features/model-architecture/editor').attachWorkbench=()=>{};
require('../src/features/spectrum-viewer/workbench').configure=()=>{};
require('../src/features/spectrum-viewer/extension').createEmbeddedProvider=()=>({resolveEmbeddedEditor:async()=>{},dispose(){}});
const pkg=require('../package.json');
(async()=>{
 let dom;
 try{
  const editor=pkg.contributes.customEditors.find(entry=>entry.viewType==='fredis.trainingDataConfigEditor');
  assert(editor.selector.some(entry=>entry.filenamePattern==='*.training-data-config.pft'));assert.equal(editor.priority,'default');
  assert(pkg.activationEvents.includes('onCustomEditor:fredis.trainingDataConfigEditor'));
  const config={input:'input.msds',output_dir:'prepared',validation_ratio:0.2,validation_seed:17,chemistry_config:fixture.chemistry,workbench_config:{application:'fragment-tree-data-preparation'}};
  fs.writeFileSync(source,JSON.stringify(config));
  const handlers=[],panel={webview:{cspSource:'test',onDidReceiveMessage:fn=>handlers.push(fn),postMessage(){}},onDidDispose(){}};
  const context={extensionPath:path.resolve(__dirname,'..')},provider=new ext.TrainingDataConfigEditorProvider(context,{appendLine(){}});
  const document=await provider.openCustomDocument({fsPath:source});await provider.resolveCustomEditor(document,panel);
  assert.equal(openedPage,'data');assert.match(panel.title,/Training Data/);assert(panel.webview.html.includes('preparationDocumentPending'));
  const errors=[],messages=[],vc=new VirtualConsole();vc.on('jsdomError',error=>errors.push(error.message));
  dom=new JSDOM(panel.webview.html,{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(JSON.parse(JSON.stringify(m))),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){}disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:value=>value};}});
  const w=dom.window,d=w.document;
  const snapshot=()=>{d.getElementById('copyCommand').click();return messages.filter(m=>m.type==='copyCommand').at(-1).config;};
  assert.equal(d.getElementById('form').hidden,false);
  assert.equal(snapshot().datasets[0].path,path.join(temporary,'input.msds'));
  assert.equal(snapshot().validationRatio,0.2);assert.equal(snapshot().validationSeed,17);assert.equal(snapshot().outputDir,path.join(temporary,'prepared'));
  // Initial project restoration must not replace the explicitly opened document.
  w.dispatchEvent(new w.MessageEvent('message',{data:{type:'project/state',project:null,recent:[],restore:true,drafts:{data:{...fixture.dataConfig,datasets:[{path:'/unrelated.msds'}],datasetMode:'automatic'}}}}));
  assert.equal(snapshot().datasets[0].path,path.join(temporary,'input.msds'));assert.equal(d.getElementById('form').hidden,false);
  await handlers.at(-1)({type:'saveConfig',config:snapshot()});assert(fs.existsSync(savedFile));assert.equal(JSON.parse(fs.readFileSync(savedFile)).datasets[0].path,path.join(temporary,'input.msds'));
  await new Promise(resolve=>setTimeout(resolve,50));assert.deepEqual(errors,[]);
  console.log('Training data configuration association, click-to-open Data page, JSON compatibility, restored settings, startup project isolation and .pft saving passed.');
 }finally{dom?.window.close();fs.rmSync(temporary,{recursive:true,force:true});Module._load=originalLoad;}
})().catch(error=>{console.error(error);process.exitCode=1;});
