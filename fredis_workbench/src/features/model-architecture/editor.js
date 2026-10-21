const fileFormat = require('../../file-formats').format('architecture');
const vscode=require('vscode'),fs=require('fs');
const parameters=require('../../workbench/parameter-editor');
const fields=require('../../workbench/mol-encoder-fields'),descriptors=require('../../workbench/mol-descriptors');
const {commonCss,formCss}=require('../reaction-pattern-set/styles');
const FILE_SUFFIX=fileFormat.suffix,VIEW_TYPE=fileFormat.viewType;
const KEYS=['mol_encoder_params','reaction_model_params','spectrum_model_params'];
function normalizeDocument(value){
 if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Expected a model architecture object.');
 const source=value.model_config||value,params=source;
 const model_config={};
 for(const key of KEYS)if(key in params){if(!params[key]||typeof params[key]!=='object'||Array.isArray(params[key]))throw new Error('Invalid '+key);model_config[key]=JSON.parse(JSON.stringify(params[key]));}
 if(!Object.keys(model_config).length)throw new Error('No model architecture components found.');
 return {model_config};
}
async function attachWorkbench(panel){panel.webview.onDidReceiveMessage(async m=>{if(!m.type?.startsWith('architecture/'))return;try{
 if(m.type==='architecture/save'){const value=normalizeDocument(m.value),uri=await vscode.window.showSaveDialog({defaultUri:vscode.Uri.file('model'+FILE_SUFFIX),filters:{'Model Architecture':['pft']}});if(uri){const target=uri.fsPath.endsWith(FILE_SUFFIX)?uri.fsPath:uri.fsPath+FILE_SUFFIX;await fs.promises.writeFile(target,JSON.stringify(value,null,2)+'\n');panel.webview.postMessage({type:'architecture/status',text:'Saved '+target});}}
 if(m.type==='architecture/load'){const picked=m.path?[vscode.Uri.file(m.path)]:await vscode.window.showOpenDialog({canSelectMany:false,filters:{'Model Architecture':['pft']}});if(picked?.length){const value=normalizeDocument(JSON.parse(await fs.promises.readFile(picked[0].fsPath,'utf8')));panel.webview.postMessage({type:'architecture/loaded',value:value.model_config});}}
 }catch(e){panel.webview.postMessage({type:'architecture/status',text:e.message});}});}
class Provider{
 resolveCustomTextEditor(document,panel){panel.webview.options={enableScripts:true};panel.webview.html=editorHtml();let applying=false,queue=Promise.resolve();
 const send=()=>{try{panel.webview.postMessage({type:'document',value:normalizeDocument(JSON.parse(document.getText())).model_config});}catch(e){panel.webview.postMessage({type:'error',message:e.message});}};
 const sub=vscode.workspace.onDidChangeTextDocument(e=>{if(e.document.uri.toString()===document.uri.toString()&&!applying)send();});panel.onDidDispose(()=>sub.dispose());
 panel.webview.onDidReceiveMessage(m=>{if(m.type==='ready')send();if(m.type==='update')queue=queue.then(async()=>{const text=JSON.stringify(normalizeDocument(m.value),null,2)+'\n';if(text===document.getText())return;const edit=new vscode.WorkspaceEdit();edit.replace(document.uri,new vscode.Range(document.positionAt(0),document.positionAt(document.getText().length)),text);applying=true;try{if(!await vscode.workspace.applyEdit(edit))throw new Error('Unable to update document');}finally{applying=false;}}).catch(e=>panel.webview.postMessage({type:'error',message:e.message}));});
 }
}
function editorHtml(){return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}${formCss()}${parameters.css()}</style></head><body><main><h1>Model Architecture</h1><p>Architecture only. Save with Ctrl+S.</p><p id="error" role="alert"></p><div id="editor" class="parameter-editor"></div><script>const vscode=acquireVsCodeApi(),molEncoderFields=${JSON.stringify(fields)},molDescriptors=${JSON.stringify(descriptors)};${parameters.createParameterEditor.toString()}
let editor,timer;const mount=document.getElementById('editor');function update(){clearTimeout(timer);timer=setTimeout(()=>{if(editor)vscode.postMessage({type:'update',value:editor.getValue()});},150);}for(const type of ['input','change','click'])mount.addEventListener(type,update);window.addEventListener('message',e=>{if(e.data.type==='document'){if(editor)editor.setValue(e.data.value);else editor=createParameterEditor(mount,e.data.value,'architecture');}if(e.data.type==='error')document.getElementById('error').textContent=e.data.message;});vscode.postMessage({type:'ready'});</script></main></body></html>`;}
function register(context){context.subscriptions.push(vscode.window.registerCustomEditorProvider(VIEW_TYPE,new Provider(),{webviewOptions:{retainContextWhenHidden:true},supportsMultipleEditorsPerDocument:false}));}
module.exports={register,attachWorkbench,normalizeDocument,FILE_SUFFIX,VIEW_TYPE,editorHtml};
