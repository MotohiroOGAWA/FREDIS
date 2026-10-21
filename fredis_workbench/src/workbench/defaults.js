const path=require('path');
const vscode=require('vscode');
const {merge}=require('./parameter-service');
function outputDirectory(root,workflow,now=new Date()){
  if(typeof root!=='string'||!root.trim())return '';
  return path.join(root.trim(),`${workflow}-${now.toISOString().replace(/[:.]/g,'-')}`);
}
function workflowDefaults(workflow,builtins,projectRoot,configuration=vscode.workspace.getConfiguration('fredis.workbench')){
  const overrides=configuration.get(`${workflow}Defaults`,{});
  const result=merge(builtins,overrides&&typeof overrides==='object'&&!Array.isArray(overrides)?overrides:{});
  if(workflow==='data'&&overrides?.fragmenterParams)result.modelConfig=merge(result.modelConfig||{},{fragmenter_params:overrides.fragmenterParams});
  let root=configuration.get('defaultOutputDirectory','');
  if(root&&projectRoot&&!path.isAbsolute(root))root=path.resolve(projectRoot,root);
  const field=workflow==='prediction'?'batchOutputDir':'outputDir';
  result[field]=outputDirectory(root,workflow==='data'?'preparation':workflow);
  return result;
}
function attach(panel){panel.webview.onDidReceiveMessage(async message=>{
  if(message?.type!=='defaults/save'||!['data','training','prediction','trainable'].includes(message.workflow))return;
  try{
    if(!message.config||typeof message.config!=='object'||Array.isArray(message.config))throw new Error('Invalid default configuration.');
    const config=merge({},message.config);
    for(const key of ['application','outputDir','batchOutputDir','output'])delete config[key];
    const target=vscode.workspace.workspaceFolders?.length?vscode.ConfigurationTarget.Workspace:vscode.ConfigurationTarget.Global;
    await vscode.workspace.getConfiguration('fredis.workbench').update(`${message.workflow}Defaults`,config,target);
    panel.webview.postMessage({type:'defaults/saved',workflow:message.workflow});
  }catch(error){panel.webview.postMessage({type:'defaults/error',workflow:message.workflow,error:error.message});}
});}
function client(){
  for(const workflow of ['data','training','prediction','trainable']){
    const button=document.getElementById(workflow+'SaveDefaults');
    if(!button)continue;
    button.onclick=()=>{let config;if(workflow==='data')config=getConfig();else if(workflow==='training')config=getTrainingConfig();else if(workflow==='trainable')config={fragmenterParams:window.getTrainableFragmenterConfig(),maxReactionCombinations:Number(document.getElementById('trainableExtractForm').elements.maxReactionCombinations.value),maxUniqueFragmentSmiles:Number(document.getElementById('trainableExtractForm').elements.maxUniqueFragmentSmiles.value)};else{config={};for(const input of document.getElementById('predictForm').elements)if(input.name)config[input.name]=input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value):input.value;}
      vscode.postMessage({type:'defaults/save',workflow,config});};
  }
  window.addEventListener('message',event=>{const m=event.data;if(['defaults/saved','defaults/error'].includes(m.type)){const status=document.getElementById(m.workflow+'DefaultsStatus');if(status)status.textContent=m.error||'Defaults saved. They apply when you open a new Workbench. Output directories use the default output root in Preferences.';}});
}
function html(workflow){if(workflow!=='trainable')return '';return `<div class="actions"><button type="button" id="trainableSaveDefaults">Save as Defaults</button><span id="trainableDefaultsStatus" role="status"></span></div>`;}
module.exports={outputDirectory,workflowDefaults,attach,html,script:()=>`(${client.toString()})();`};
