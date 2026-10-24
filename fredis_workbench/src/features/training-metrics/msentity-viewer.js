const fs = require('fs');
const path = require('path');
const upstream = require('../spectrum-viewer/upstream');

function assets() {
  return {js:fs.readFileSync(upstream.file('media','spectrum.js'),'utf8'),
    css:fs.readFileSync(upstream.file('media','spectrum.css'),'utf8')};
}

function comparisonPayload(record, key) {
  const row={SMILES:record.smiles||record.SMILES||'',
    'Saved assignment score (with precursor)':record.assignment_score,
    'Saved assignment score (without precursor)':record.assignment_score_without_precursor,
    'Saved cosine (with precursor)':record.cosine_similarity,
    'Saved cosine (without precursor)':record.cosine_similarity_without_precursor,
    Adduct:record.main_adduct||record.adduct,CollisionEnergy:record.collision_energy,PrecursorMZ:record.precursor_mz};
  const payload=(role,peaks)=>({title:role==='generated'?'Generated':'Observed',datasetName:'Validation '+role,
    globalIndex:0,row,columns:Object.keys(row),
    spectrum:{mz:peaks.map(p=>Number(p.mz)),intensity:peaks.map(p=>Number(p.intensity)),
      metadata_columns:['precursor','depth','assigned'],
      metadata:peaks.map(p=>({precursor:!!p.precursor,depth:p.depth??null,assigned:!!p.assigned}))}});
  const generated=record.generated_peaks || (record.generated_mz||[]).map((mz,i)=>({mz,intensity:record.generated_intensity[i]}));
  return {type:'spectrum-comparison',selectionKey:key,top:payload('generated',generated),bottom:payload('observed',record.original_peaks||[])};
}

// Serialized into both the result viewer and the Workbench. The upstream
// spectrum.js and spectrum.css remain unchanged and isolated in an iframe.
function installMsentityViewer(assets, vscode) {
  let frame=null,ready=false,pending=null,signature='',themeObserver=null;
  const nonce=document.currentScript?.nonce || '';
  let viewerState=vscode.getState()?.validationViewerState || {};
  const syncTheme=()=>{
    const root=frame?.contentDocument?.documentElement;
    if(!root)return;
    const current=getComputedStyle(document.documentElement);
    for(let i=0;i<current.length;i++) {const name=current[i];if(name.startsWith('--vscode-'))root.style.setProperty(name,current.getPropertyValue(name));}
    frame.contentDocument.body.className=document.body.className;
  };
  const send=()=>{if(ready&&pending)frame.contentWindow.postMessage(pending,'*');};
  window.addEventListener('message',event=>{
    if(!frame||event.source!==frame.contentWindow||event.data?.type!=='fredisValidationSpectrum')return;
    const message=event.data.message;
    if(message?.type==='ready'){ready=true;syncTheme();send();}
    else if(message?.type==='viewer-height' && Number.isFinite(message.height) && message.height>0){frame.style.height=Math.ceil(message.height)+'px';}
    else if(message?.type==='viewer-state'){viewerState=message.value;vscode.setState({...vscode.getState(),validationViewerState:viewerState});}
    else if(['save-image','save-peaks','copy-notification','clipboard-error','export-error'].includes(message?.type))vscode.postMessage({type:'metricsSpectrumAction',action:message});
  });
  return {update(container,payload){
    if(!frame){
      frame=document.createElement('iframe');frame.title='msentity validation spectrum comparison';frame.id='validationMsentityFrame';frame.setAttribute('scrolling','no');frame.style.cssText='display:block;width:100%;height:540px;border:0';
      container.append(frame);
      const style=assets.css+'\n.spectrum-viewer{max-width:none;padding:8px}.spectrum-header{padding-bottom:8px}.spectrum-header h1{font-size:16px}.spectrum-workspace,.spectrum-workspace.compared{grid-template-columns:minmax(0,1fr) minmax(0,40%);align-items:start}.plot-panel{width:100%}.peaks-panel{min-width:0}.peak-scroll{max-height:none;overflow:visible}.peaks-panel.compared .peak-table{min-width:0;table-layout:fixed}.peak-table td,.peak-table th{overflow-wrap:anywhere}.peaks-header{flex-wrap:wrap}html,body{min-height:0;height:auto;overflow:hidden}.metadata-panels.compared{grid-template-columns:repeat(2,minmax(0,1fr))}#spectrum-svg{display:block;width:100%;height:300px}';
      const bootstrap='const reportHeight=()=>{const app=document.getElementById("spectrum-app");if(app&&document.getElementById("spectrum-svg"))parent.postMessage({type:"fredisValidationSpectrum",message:{type:"viewer-height",height:Math.ceil(app.getBoundingClientRect().height)+4}},"*");};if(typeof ResizeObserver!=="undefined")new ResizeObserver(reportHeight).observe(document.getElementById("spectrum-app"));window.addEventListener("resize",reportHeight);new MutationObserver(()=>{const plot=document.getElementById("spectrum-svg");if(plot)plot.setAttribute("preserveAspectRatio","none");requestAnimationFrame(reportHeight);}).observe(document.documentElement,{childList:true,subtree:true});window.acquireVsCodeApi=()=>({getState:()=>window.validationViewerState,setState:value=>{window.validationViewerState=value;parent.postMessage({type:"fredisValidationSpectrum",message:{type:"viewer-state",value}},"*");},postMessage:message=>parent.postMessage({type:"fredisValidationSpectrum",message},"*")});window.validationViewerState='+JSON.stringify(viewerState).replace(/</g,'\\u003c')+';';
      frame.srcdoc='<!doctype html><html><head><meta charset="UTF-8"><style>'+style+'</style></head><body><div id="spectrum-app">Loading msentity…</div><script nonce="'+nonce+'">'+bootstrap+assets.js+'<'+ '/script></body></html>';
      frame.addEventListener('load',syncTheme);
      themeObserver=new MutationObserver(syncTheme);themeObserver.observe(document.documentElement,{attributes:true});themeObserver.observe(document.body,{attributes:true});
    }
    const next=JSON.stringify(payload);
    if(next!==signature){signature=next;pending=payload;send();}
  }};
}

async function handleAction(vscode, action) {
  if (action?.type==='save-image') {
    if(!Array.isArray(action.bytes)||!action.bytes.every(n=>Number.isInteger(n)&&n>=0&&n<=255))throw new Error('Invalid image export.');
    const name=path.basename(action.filename||'validation-spectrum.png');
    const target=await vscode.window.showSaveDialog({defaultUri:vscode.Uri.file(name),filters:name.toLowerCase().endsWith('.svg')?{SVG:['svg']}:{PNG:['png']}});
    if(target)await vscode.workspace.fs.writeFile(target,Uint8Array.from(action.bytes));
  } else if(action?.type==='save-peaks') {
    const target=await vscode.window.showSaveDialog({defaultUri:vscode.Uri.file(path.basename(action.basename||'validation-spectrum')+'.tsv'),filters:{TSV:['tsv'],CSV:['csv']}});
    if(!target)return;
    const format=target.fsPath.toLowerCase().endsWith('.csv')?'csv':'tsv';
    if(typeof action.contents?.[format]!=='string')throw new Error('Invalid peak export.');
    await vscode.workspace.fs.writeFile(target,Buffer.from(action.contents[format],'utf8'));
  } else if(action?.type==='copy-notification')vscode.window.setStatusBarMessage(action.message||'Copied peaks',3000);
  else if(['clipboard-error','export-error'].includes(action?.type))vscode.window.showErrorMessage(action.message||'Spectrum export failed.');
}

module.exports={assets,comparisonPayload,installMsentityViewer,handleAction};
