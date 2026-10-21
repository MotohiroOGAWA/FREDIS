const fileFormat = require('../../file-formats').format('fragmenter');
const vscode=require('vscode');
const fs=require('fs');
const path=require('path');
const reactionPatternUi=require('../reaction-pattern-set/ui');
const reactionPatternVisualEditor=require('../reaction-pattern-set/visual-editor');
const reactionPatternPreview=require('../reaction-pattern-set/reaction-preview');
const reactionPatternSetEditor=require('../reaction-pattern-set/editor');
const adductUi=require('../adduct-rule-set/ui');
const adductEditor=require('../adduct-rule-set/editor');
const {commonCss,formCss}=require('../reaction-pattern-set/styles');
const host=require('../reaction-pattern-set/host');
const pathDrop=require('../../webview-path-drop');
const parameterEditor=require('../../workbench/parameter-editor');
const molEncoderFields=require('../../workbench/mol-encoder-fields');
const parameterService=require('../../workbench/parameter-service');
// .pft (not .json) so VS Code's contributed language icon actually shows, as
// for .reactionset.pft / .adductset.pft.
const VIEW_TYPE=fileFormat.viewType,FILE_SUFFIX=fileFormat.suffix;
// The file holds the fragmenter_params object itself, exactly what
// dataset preparation and Fragmenter.from_dict read.
function normalizeDocument(value){return parameterService.unpackFragmenter(JSON.parse(JSON.stringify(value)));}
// Fill fields missing from an older or hand-written file from the project preset,
// so every option is editable, as when a config is loaded in the Workbench.
function withDefaults(context,value){try{return parameterService.merge(parameterService.chemistryDefaults(host.projectRoot(context)).fragmenter_params,value);}catch{return value;}}
function savePath(previousPath){const name=previousPath?path.basename(previousPath).replace(/\.fragmenter\.(pft|json)$/i,'').replace(/\.(pft|json)$/i,'')+'_copy':'fragmenter';return host.reactionPatternSetSavePath(name,previousPath,FILE_SUFFIX);}
async function saveAs(value,previousPath){const params=normalizeDocument(value),selected=await vscode.window.showSaveDialog({filters:{'FREDIS Fragmenter Config':['pft']},defaultUri:vscode.Uri.file(savePath(previousPath))});if(!selected)return undefined;const target=host.ensureFileSuffix(selected.fsPath,FILE_SUFFIX);await fs.promises.writeFile(target,JSON.stringify(params,null,2)+'\n');vscode.window.showInformationMessage(`Saved ${path.basename(target)}`);return target;}
// Messages of the embedded Reaction Pattern Set / Adduct Rule Set editors,
// answered exactly as by their own file editors.
async function embeddedSetMessage(context,panel,message){
  if(message.type==='chemistry'){const result=await host.runChemistryBackend(context,message.command,message.payload,panel);panel.webview.postMessage({type:'chemistryResult',requestId:message.requestId,result});return true;}
  if(message.type==='selectElements'){const elements=await host.showElementPicker(message.elements);if(elements)panel.webview.postMessage({type:'elementSelectionResult',requestId:message.requestId,elements});return true;}
  if(message.type==='copyText'){await vscode.env.clipboard.writeText(String(message.text||''));return true;}
  if(message.type==='loadReactionPatternSet'||message.type==='loadReactionPattern'){const source=await host.readReactionPatternImport(context,message);if(!source)return true;if(message.type==='loadReactionPatternSet')panel.webview.postMessage({type:'reactionPatternSet',value:reactionPatternSetEditor.normalizeDocument(source.value),path:source.path});else panel.webview.postMessage({type:'reactionPatternLoaded',path:source.path,pattern:reactionPatternSetEditor.normalizePattern(source.value),index:message.index});return true;}
  if(message.type==='saveReactionPatternSet'){const value=reactionPatternSetEditor.normalizeDocument(message.value),selected=await vscode.window.showSaveDialog({filters:{'FREDIS Reaction Pattern Set':['pft']},defaultUri:vscode.Uri.file(message.path||host.reactionPatternSetSavePath(value.reaction_pattern_set.name,'',reactionPatternSetEditor.FILE_SUFFIX))});if(selected){const target=host.ensureFileSuffix(selected.fsPath,reactionPatternSetEditor.FILE_SUFFIX);await fs.promises.writeFile(target,JSON.stringify(value,null,2)+'\n');panel.webview.postMessage({type:'reactionPatternSetSaved',path:target});vscode.window.showInformationMessage(`Saved ${path.basename(target)}`);}return true;}
  if(message.type==='saveReactionPattern'){const pattern=reactionPatternSetEditor.normalizePattern(message.pattern),selected=await vscode.window.showSaveDialog({filters:{'FREDIS Reaction Pattern':['pft']},defaultUri:vscode.Uri.file(message.path||`${host.safeFileStem(pattern.name||'pattern')}.reaction.pft`)});if(selected){const target=host.ensureFileSuffix(selected.fsPath,require('../../file-formats').format('reaction').suffix);await fs.promises.writeFile(target,JSON.stringify(pattern,null,2)+'\n');vscode.window.showInformationMessage(`Saved ${path.basename(target)}`);}return true;}
  if(message.type==='loadAdductRuleSet'||message.type==='loadAdductRule'){const source=await host.readReactionPatternImport(context,message);if(!source)return true;if(message.type==='loadAdductRuleSet')panel.webview.postMessage({type:'adductRuleSet',value:adductEditor.normalizeDocument(source.value),path:source.path});else panel.webview.postMessage({type:'adductRuleLoaded',rule:adductEditor.normalizeRule(source.value),index:message.index,path:source.path});return true;}
  if(message.type==='saveAdductRuleSet'){const value=adductEditor.normalizeDocument(message.value),selected=await vscode.window.showSaveDialog({filters:{'FREDIS Fragment Ion Adduct Rule Set':['pft']},defaultUri:vscode.Uri.file(host.reactionPatternSetSavePath(value.fragment_ion_adduct_rule_set.name,message.path,adductEditor.FILE_SUFFIX))});if(selected){const target=host.ensureFileSuffix(selected.fsPath,adductEditor.FILE_SUFFIX);await fs.promises.writeFile(target,JSON.stringify(value,null,2)+'\n');panel.webview.postMessage({type:'adductRuleSetSaved',path:target});vscode.window.showInformationMessage(`Saved ${path.basename(target)}`);}return true;}
  if(message.type==='saveAdductRule'){const rule=adductEditor.normalizeRule(message.rule),selected=await vscode.window.showSaveDialog({filters:{'FREDIS Fragment Ion Adduct Rule':['pft']},defaultUri:vscode.Uri.file(host.safeFileStem(rule.name||'adduct_rule')+adductEditor.RULE_SUFFIX)});if(selected){const target=host.ensureFileSuffix(selected.fsPath,adductEditor.RULE_SUFFIX);await fs.promises.writeFile(target,JSON.stringify(rule,null,2)+'\n');vscode.window.showInformationMessage(`Saved ${path.basename(target)}`);}return true;}
  return false;
}
class Provider{
 constructor(context){this.context=context;}
 resolveCustomTextEditor(document,panel){const context=this.context;panel.webview.options={enableScripts:true};panel.webview.html=editorHtml();pathDrop.attach(panel);let applying=false;
  const current=()=>{const text=document.getText();return withDefaults(context,text.trim()?normalizeDocument(JSON.parse(text)):{});};
  const send=()=>{try{panel.webview.postMessage({type:'fragmenter',value:current(),path:document.uri.fsPath});}catch(error){panel.webview.postMessage({type:'fragmenterError',message:error.message});}};
  const subscription=vscode.workspace.onDidChangeTextDocument(event=>{if(event.document.uri.toString()===document.uri.toString()&&!applying)send();});panel.onDidDispose(()=>subscription.dispose());
  const write=async raw=>{const text=JSON.stringify(normalizeDocument(raw),null,2)+'\n';if(text===document.getText())return;const edit=new vscode.WorkspaceEdit();edit.replace(document.uri,new vscode.Range(document.positionAt(0),document.positionAt(document.getText().length)),text);applying=true;try{if(!await vscode.workspace.applyEdit(edit))throw new Error('The document could not be updated.');}finally{applying=false;}};
  panel.webview.onDidReceiveMessage(async message=>{try{
   if(message.type==='ready'){send();return;}
   if(message.type==='update'){await write(message.value);return;}
   if(message.type==='loadFragmenter'){const source=await host.readReactionPatternImport(context,message);if(!source)return;await write(withDefaults(context,normalizeDocument(source.value)));send();panel.webview.postMessage({type:'fragmenterStatus',message:'Loaded '+(path.basename(source.path)||'fragmenter config')+' into this file.'});return;}
   if(message.type==='saveFragmenterAs'){const target=await saveAs(message.value,document.uri.fsPath);if(target)panel.webview.postMessage({type:'fragmenterStatus',message:'Saved a copy as '+path.basename(target)+'.'});return;}
   await embeddedSetMessage(context,panel,message);
  }catch(error){panel.webview.postMessage({type:'fragmenterError',message:String(error.message||error)});}});
 }
}
function register(context){context.subscriptions.push(vscode.window.registerCustomEditorProvider(VIEW_TYPE,new Provider(context),{webviewOptions:{retainContextWhenHidden:true},supportsMultipleEditorsPerDocument:false}));}
// Same Fragmentation editor, Reaction Pattern Set editor and Adduct Rule Set
// editor as Data / Training Data, so this file editor cannot drift from it.
function html(){return `<div id="fragmenterApp" class="fragmenter-file-editor"><div class="page-heading"><h1>Fragmenter Configuration</h1><p id="fragmenterPath" class="muted"></p></div><p id="fragmenterError" class="danger" role="alert"></p><div id="fragmenterDrop" class="fragmenter-config-tools" title="Drop a fragmenter config JSON here"><span id="fragmenterStatus" role="status">Edits are written to this file. Save it with Ctrl+S.</span><button type="button" id="fragmenterLoad">Load</button><button type="button" id="fragmenterSaveAs">Save As…</button></div><div id="fragmenterParameterEditor" class="parameter-editor"><p class="muted">Loading…</p></div></div><div id="reactionPatternPageMount">${reactionPatternUi.html()}</div><div id="adductPageMount">${adductUi.html()}</div>`;}
function client(){
  const el=id=>document.getElementById(id),mount=el('fragmenterParameterEditor'),status=el('fragmenterStatus'),error=el('fragmenterError');
  let editor=null,filePath='',syncTimer;
  function changed(){if(!editor)return;clearTimeout(syncTimer);syncTimer=setTimeout(()=>{error.textContent='';vscode.postMessage({type:'update',value:editor.getValue().fragmenter_params});},200);}
  function editPatterns(value,apply,target){reactionPatternModel={reaction_pattern_set:value};reactionPatternPath='';target.append(el('reactionPatternApp'));el('reactionPatternApp').hidden=false;el('applyParameterPatterns').hidden=true;el('reactionPatternPath').textContent='Embedded in this fragmenter config';onReactionPatternChange=()=>{apply(reactionPatternModel.reaction_pattern_set);changed();};renderReactionPatterns();}
  function editAdducts(value,apply,target){adductModel={fragment_ion_adduct_rule_set:value};adductPath='';target.append(el('adductApp'));el('adductApp').hidden=false;el('applyParameterAdductRules').hidden=true;el('adductPath').textContent='Embedded in this fragmenter config';onAdductChange=()=>{apply(adductModel.fragment_ion_adduct_rule_set);changed();};renderAdductRules();}
  // Field edits bubble input/change; Add/Remove buttons only re-render. Clicks
  // inside the embedded set editors are covered by onReactionPatternChange/onAdductChange.
  mount.addEventListener('input',event=>{if(!event.target.closest('#reactionPatternApp,#adductApp'))changed();});
  mount.addEventListener('change',event=>{if(!event.target.closest('#reactionPatternApp,#adductApp'))changed();});
  mount.addEventListener('click',event=>{if(event.target.closest('button')&&!event.target.closest('#reactionPatternApp,#adductApp'))changed();});
  el('fragmenterLoad').onclick=()=>vscode.postMessage({type:'loadFragmenter'});
  el('fragmenterSaveAs').onclick=()=>{if(editor)vscode.postMessage({type:'saveFragmenterAs',value:editor.getValue().fragmenter_params});};
  const zone=el('fragmenterDrop');installFileDropZone(zone,{kind:'file',upload:false,onDrop:paths=>vscode.postMessage({type:'loadFragmenter',path:paths[0]}),onFiles:async files=>{const file=files[0];if(file.size>2000000)throw new Error('Fragmenter config is too large.');vscode.postMessage({type:'loadFragmenter',json:await file.text()});},onError:message=>{error.textContent=message;}});
  window.addEventListener('message',event=>{const message=event.data;
    if(message.type==='fragmenter'){error.textContent='';filePath=message.path||filePath;el('fragmenterPath').textContent=filePath;const value={fragmenter_params:message.value};if(!editor)editor=createParameterEditor(mount,value,'fragmenter',editPatterns,editAdducts);else if(JSON.stringify(editor.getValue())!==JSON.stringify(value))editor.setValue(value);}
    if(message.type==='fragmenterStatus')status.textContent=message.message;
    if(message.type==='fragmenterError'||message.type==='reactionPatternError'||message.type==='adductRuleError')error.textContent=message.message;
  });
}
function editorHtml(){return `<!doctype html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${commonCss()}${formCss()}${pathDrop.css()}${parameterEditor.css()}</style></head><body><main>${html()}<script>const vscode=acquireVsCodeApi();const molEncoderFields=${JSON.stringify(molEncoderFields)};${reactionPatternUi.state()}${adductUi.state()}const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));${pathDrop.script({host:true})}${reactionPatternUi.script()}${reactionPatternVisualEditor.script()}${reactionPatternPreview.script()}${adductUi.script()}${parameterEditor.createParameterEditor.toString()}(${client.toString()})();vscode.postMessage({type:'ready'});</script></main></body></html>`;}
module.exports={register,normalizeDocument,editorHtml,embeddedSetMessage,FILE_SUFFIX,VIEW_TYPE};
