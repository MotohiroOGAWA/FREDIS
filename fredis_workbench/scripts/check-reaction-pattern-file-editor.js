const assert=require('assert/strict'),fs=require('fs'),path=require('path'),os=require('os'),Module=require('module');
const {JSDOM,VirtualConsole}=require('jsdom');
const providers=new Map();let changed,document,currentText,allowEdit=true,applyEditCalls=0,saveDialogResult,saveDialogOptions,openDialogResult,clipboardText,informationMessages=[];
const vscode={
  window:{
    registerCustomEditorProvider:(name,provider)=>{providers.set(name,provider);return {dispose(){}};},
    showSaveDialog:async options=>{saveDialogOptions=options;return saveDialogResult;},
    showOpenDialog:async()=>openDialogResult,
    showInformationMessage:(text)=>{informationMessages.push(text);}
  },
  workspace:{
    onDidChangeTextDocument:fn=>{changed=fn;return {dispose(){}};},
    applyEdit:async edit=>{applyEditCalls++;if(!allowEdit)return false;currentText=edit.text;changed({document});return true;}
  },
  env:{clipboard:{writeText:async text=>{clipboardText=text;}}},
  Range:class{constructor(a,b){this.a=a;this.b=b;}},
  WorkspaceEdit:class{replace(uri,range,text){this.text=text;}},
  Uri:{file:p=>({fsPath:p})}
};
// A stub of the shared host module: the real one is exercised end-to-end by
// check-reaction-service.js / check-reaction-preview.py / check-reaction-pattern-ui.js.
// Here we only need to confirm the editor forwards to it correctly.
const host={
  runChemistryBackend:async(context,command,payload,owner)=>({command,payload,ownerIsPanel:!!(owner&&typeof owner.onDidDispose==='function')}),
  showElementPicker:async elements=>[...elements,'Xe'],
  readReactionPatternImport:async()=>importResult,
  safeFileStem:value=>String(value).trim().replace(/[^A-Za-z0-9_.-]+/g,'_')||'pattern',
  ensureFileSuffix:(filePath,suffix,aliases=[])=>{const accepted=[suffix,...aliases].map(v=>v.toLowerCase());if(accepted.some(e=>filePath.toLowerCase().endsWith(e)))return filePath;return filePath.replace(/\.json$/i,'')+suffix;},
  reactionPatternSetSavePath:(name,previousPath,suffix)=>{const stem=String(name||'patterns').trim().replace(/[^\w.-]+/g,'_')||'patterns';return previousPath?path.join(path.dirname(previousPath),stem+suffix):stem+suffix;}
};
let importResult;
const load=Module._load;
Module._load=function(name,parent,main){
  if(name==='vscode')return vscode;
  if(name==='./host'&&parent&&/reaction-pattern-set[\\/]editor\.js$/.test(parent.filename||''))return host;
  return load.call(this,name,parent,main);
};
const feature=require('../src/features/reaction-pattern-set/editor');Module._load=load;
feature.register({subscriptions:[]});
const manifest=require('../package.json');
for(const suffix of ['*.reaction.pft'])assert(manifest.contributes.customEditors.some(item=>item.viewType==='fredis.reactionPatternEditor'&&item.priority==='default'&&item.selector.some(selector=>selector.filenamePattern===suffix)));
assert(manifest.activationEvents.includes('onCustomEditor:fredis.reactionPatternEditor'));
assert(manifest.contributes.customEditors.some(item=>item.viewType==='fredis.reactionPatternSetEditor'&&item.selector.some(selector=>selector.filenamePattern==='*.reactionset.pft')));
assert.equal(feature.PATTERN_VIEW_TYPE,'fredis.reactionPatternEditor');assert.equal(feature.VIEW_TYPE,'fredis.reactionPatternSetEditor');

async function open(viewType,value){
 currentText=JSON.stringify(value);const messages=[];let handler,dispose;
 document={uri:{toString:()=>'/tmp/pattern.json',fsPath:'/tmp/pattern.json'},getText:()=>currentText,positionAt:value=>value};
 const panel={webview:{postMessage:message=>messages.push(message),onDidReceiveMessage:fn=>handler=fn},onDidDispose:fn=>dispose=fn};
 providers.get(viewType).resolveCustomTextEditor(document,panel);await handler({type:'ready'});
 return {panel,messages,send:message=>handler(message),dispose};
}

(async()=>{
 const original=JSON.parse(fs.readFileSync(path.join(__dirname,'../../fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/reaction_patterns/nonring_c_single_bond.reaction.pft'),'utf8'));
 const single=await open('fredis.reactionPatternEditor',original);let dom;
 try{
  const initial=single.messages.at(-1);
  assert.equal(initial.type,'reactionPatternSet');
  assert.deepEqual(initial.value,{reaction_pattern_set:{name:'',patterns:[original]}});
  assert.equal(initial.path,'/tmp/pattern.json');

  const errors=[],sent=[],console2=new VirtualConsole();console2.on('jsdomError',error=>errors.push(error.detail?.stack||error.message));
  dom=new JSDOM(single.panel.webview.html,{runScripts:'dangerously',virtualConsole:console2,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:message=>sent.push(JSON.parse(JSON.stringify(message)))});w.HTMLElement.prototype.scrollIntoView=function(){};}});
  const w=dom.window,d=w.document,post=message=>w.dispatchEvent(new w.MessageEvent('message',{data:message}));
  post(initial);
  assert.deepEqual(errors,[]);
  // Reuses the exact Workbench components: visual builder + reaction preview are really installed.
  assert(d.getElementById('visualBuilder'));assert(d.querySelector('.ce-step'));assert(d.getElementById('reactionPatternPreview'));
  assert.equal(d.querySelector('#reactionPatternApp h2').textContent,'Reaction Pattern Configuration');
  d.querySelector('[data-save-pattern="0"]').click();assert.equal(sent.at(-1).type,'saveReactionPattern');assert.equal(sent.at(-1).path,'/tmp/pattern.json');sent.pop();
  assert(d.getElementById('reactionPatternSetName').closest('label').hidden);
  assert(d.getElementById('addReactionPattern').closest('section').hidden);
  assert.equal(d.querySelectorAll('[data-remove-pattern]').length,0);
  assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,1);

  const input=d.querySelector('[data-pattern="0"][data-key="name"]');input.value='Edited <pattern>';input.dispatchEvent(new w.Event('input',{bubbles:true}));
  await new Promise(resolve=>setTimeout(resolve,260));
  assert.equal(sent.at(-1).type,'update');
  await single.send(sent.at(-1));
  let saved=JSON.parse(currentText);
  assert.equal(saved.name,'Edited <pattern>');assert.equal(saved.reactant_smarts,original.reactant_smarts);assert.equal(saved.products.length,2);assert(!('reaction_pattern_set' in saved));

  d.querySelector('[data-add-product="0"]').click();
  const productSmarts=d.querySelector('[data-pattern="0"][data-product="2"][data-key="smarts"]');productSmarts.value='[!#1:3]';productSmarts.dispatchEvent(new w.Event('input',{bubbles:true}));
  await new Promise(resolve=>setTimeout(resolve,260));
  await single.send(sent.at(-1));
  saved=JSON.parse(currentText);
  assert.equal(saved.products.length,3);assert.equal(saved.products[2].smarts,'[!#1:3]');

  // Undo / external edits are reflected back into the webview.
  currentText=`${JSON.stringify(original,null,2)}\n`;changed({document});
  assert.equal(single.messages.at(-1).type,'reactionPatternSet');
  assert.deepEqual(single.messages.at(-1).value,{reaction_pattern_set:{name:'',patterns:[original]}});

  const before=applyEditCalls;
  await single.send({type:'update',value:{reaction_pattern_set:{name:'',patterns:[]}}});
  assert.equal(single.messages.at(-1).type,'reactionPatternError');assert.deepEqual(JSON.parse(currentText),original);assert.equal(applyEditCalls,before);

  allowEdit=false;
  await single.send({type:'update',value:{reaction_pattern_set:{name:'',patterns:[{...original,name:'Rejected'}]}}});
  assert.equal(single.messages.at(-1).type,'reactionPatternError');assert.deepEqual(JSON.parse(currentText),original);allowEdit=true;

  // A no-op update (identical content) must not touch the document.
  const beforeNoop=applyEditCalls,messageCountBefore=single.messages.length;
  await single.send({type:'update',value:{reaction_pattern_set:{name:'',patterns:[original]}}});
  assert.equal(applyEditCalls,beforeNoop);assert.equal(single.messages.length,messageCountBefore);

  currentText='{"reaction_pattern_set":{"patterns":[]}}';changed({document});
  assert.equal(single.messages.at(-1).type,'reactionPatternError');assert.deepEqual(errors,[]);

  // chemistry / selectElements / copyText are forwarded to the shared host module.
  await single.send({type:'chemistry',requestId:7,command:'molecule',payload:{smiles:'C'}});
  assert.deepEqual(single.messages.at(-1),{type:'chemistryResult',requestId:7,result:{command:'molecule',payload:{smiles:'C'},ownerIsPanel:true}});
  await single.send({type:'selectElements',requestId:9,elements:['C']});
  assert.deepEqual(single.messages.at(-1),{type:'elementSelectionResult',requestId:9,elements:['C','Xe']});
  await single.send({type:'copyText',text:'hello world'});
  assert.equal(clipboardText,'hello world');
 }finally{await new Promise(resolve=>setTimeout(resolve,0));dom?.window.close();single.dispose();}

 // Save / load messages route through the shared host helpers exactly like the Workbench does.
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-reaction-pattern-editor-'));
 try{
  const single2=await open('fredis.reactionPatternEditor',original);
  saveDialogResult={fsPath:path.join(tmp,'exported_pattern')};
  await single2.send({type:'saveReactionPattern',pattern:{name:'exported',reactant_smarts:'[#6:1]',products:[]}});
  assert.equal(saveDialogOptions.defaultUri.fsPath,'/tmp/pattern.json');
  const savedPattern=JSON.parse(fs.readFileSync(path.join(tmp,'exported_pattern.reaction.pft'),'utf8'));
  assert.deepEqual(savedPattern,{name:'exported',reactant_smarts:'[#6:1]',products:[]});
  assert(informationMessages.some(text=>text.includes('exported_pattern.reaction.pft')));

  importResult={value:{name:'imported',reactant_smarts:'[#8:1]',products:[]},path:'/tmp/imported.reaction.pft'};
  await single2.send({type:'loadReactionPattern',index:0});
  assert.equal(single2.messages.at(-1).type,'reactionPatternLoaded');
  assert.deepEqual(single2.messages.at(-1).pattern,{name:'imported',reactant_smarts:'[#8:1]',products:[]});
  assert.equal(single2.messages.at(-1).path,'/tmp/imported.reaction.pft');
  assert.equal(single2.messages.at(-1).index,0);
  single2.dispose();
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}

 // The Pattern Set editor shares the same code path but keeps the full multi-pattern UI.
 const value={reaction_pattern_set:{name:'Original set',patterns:[original]}},set=await open('fredis.reactionPatternSetEditor',value);
 let setDom;
 try{
  assert.deepEqual(set.messages.at(-1),{type:'reactionPatternSet',value,path:'/tmp/pattern.json'});
  const sent2=[],setErrors=[],setConsole=new VirtualConsole();setConsole.on('jsdomError',error=>setErrors.push(error.detail?.stack||error.message));
  setDom=new JSDOM(set.panel.webview.html,{runScripts:'dangerously',virtualConsole:setConsole,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>sent2.push(JSON.parse(JSON.stringify(m)))});w.HTMLElement.prototype.scrollIntoView=function(){};}});
  const w=setDom.window,d=w.document;
  w.dispatchEvent(new w.MessageEvent('message',{data:set.messages.at(-1)}));
  assert.deepEqual(setErrors,[]);
  assert.equal(d.querySelector('#reactionPatternApp h2').textContent,'Reaction Pattern Set Configuration');
  assert(!d.getElementById('reactionPatternSetName').closest('label').hidden);
  assert(!d.getElementById('addReactionPattern').closest('section').hidden);
  assert.equal(d.querySelectorAll('[data-remove-pattern]').length,1);

  const nameInput=d.getElementById('reactionPatternSetName');nameInput.value='Renamed set';nameInput.dispatchEvent(new w.Event('input',{bubbles:true}));
  await new Promise(resolve=>setTimeout(resolve,260));
  assert(!sent2.some(message=>message.type==='update'));
  assert.deepEqual(JSON.parse(currentText),value);

  const tmp2=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-reaction-pattern-set-'));
  saveDialogResult={fsPath:path.join(tmp2,'exported_set')};
  try{
	   await set.send({type:'saveReactionPatternSet',value:{reaction_pattern_set:{name:'exported_set',patterns:[original]}},path:''});
	   assert.equal(saveDialogOptions.defaultUri.fsPath,'/tmp/exported_set.reactionset.pft');
   const savedSet=JSON.parse(fs.readFileSync(path.join(tmp2,'exported_set.reactionset.pft'),'utf8'));
   assert.deepEqual(savedSet,{reaction_pattern_set:{name:'exported_set',patterns:[original]}});
  }finally{fs.rmSync(tmp2,{recursive:true,force:true});}
 }finally{await new Promise(resolve=>setTimeout(resolve,0));setDom?.window.close();set.dispose();}
 console.log('Shared reaction-pattern editor registration, visual-builder reuse, single-pattern live sync, set draft isolation, error handling, no-op writes, chemistry/element passthrough and import/export passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
