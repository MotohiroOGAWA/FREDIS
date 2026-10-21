function filteringHtml() {
  return `<section id="filteringEvaluation"><header><div><span class="eyebrow">FREDIS EVALUATION · PREPARATION</span><h1>Training Data Filtering</h1><p>Common = SpecIDs retained by every tool. Percentages use metadata SpecIDs in each category as denominator.</p></div><div class="actions"><button id="filterLoad">Load Settings</button><button id="filterSave">Save Settings</button></div></header><section class="panel"><label>Title<input id="filterTitle"></label><label>Metadata path (.msds / CSV / TSV / JSON / Parquet)<div class="path"><input id="filterMetadata"><button id="filterBrowseMetadata">Browse</button></div></label><button id="filterInspect">Load columns</button><label>SpecID column<select id="filterId"><option>SpecID</option></select></label><label>Classification column<select id="filterGroup"></select></label><h2>Tools</h2><div id="filterTools"></div><button id="filterAdd">＋ Add tool</button><div class="actions"><button id="filterRun" class="primary">Evaluate</button><button id="filterExport">Export directory (CSV + PNG + settings)</button></div></section><p id="filterStatus" role="status"></p><img id="filterChart" style="max-width:100%" hidden alt="Retention percentages by category and tool"><div style="overflow:auto"><table id="filterResults"></table><table id="filterDiagnostics"></table></div></section>`;
}
function filteringClient() {
  const get = id => document.getElementById('filter'+id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let tools = [], currentPath, pending;
  const request = () => ({title:get('Title').value,metadata:get('Metadata').value,idColumn:get('Id').value,groupColumn:get('Group').value,tools:tools.map(t=>({...t}))});
  function renderTools() {
    get('Tools').innerHTML = tools.map((t,i)=>`<div class="panel" data-tool="${i}"><label>Tool name<input data-field="name" value="${esc(t.name)}"></label><label>SpecID list path (one ID per line)<div class="path"><input data-field="path" value="${esc(t.path)}"><button data-browse="${i}">Browse</button></div></label><label>Color<input data-field="color" type="color" value="${esc(t.color)}"></label><button data-remove="${i}">Remove</button></div>`).join('');
  }
  get('Tools').oninput = e => {if(e.target.dataset.field)tools[Number(e.target.closest('[data-tool]').dataset.tool)][e.target.dataset.field]=e.target.value;};
  get('Tools').onclick = e => {
    if(e.target.dataset.remove!==undefined){tools.splice(Number(e.target.dataset.remove),1);renderTools();}
    if(e.target.dataset.browse!==undefined)vscode.postMessage({type:'filteringPick',index:Number(e.target.dataset.browse)});
  };
  get('Add').onclick=()=>{tools.push({name:'Tool '+(tools.length+1),path:'',color:'#36c5a2'});renderTools();};
  get('Add').click();
  get('BrowseMetadata').onclick=()=>vscode.postMessage({type:'filteringPick',metadata:true});
  get('Inspect').onclick=()=>vscode.postMessage({type:'filteringInspect',request:request()});
  get('Run').onclick=()=>vscode.postMessage({type:'filteringRun',request:request()});
  get('Export').onclick=()=>vscode.postMessage({type:'filteringExport',request:request()});
  get('Save').onclick=()=>vscode.postMessage({type:'saveEvaluationConfig',kind:'filtering',config:request()});
  get('Load').onclick=()=>vscode.postMessage({type:'loadEvaluationConfig',kind:'filtering'});
  window.evalFiltering={request,getPath:()=>currentPath};
  function table(id, rows) {
    if(!rows.length){get(id).innerHTML='';return;}
    const keys=Object.keys(rows[0]);
    get(id).innerHTML='<thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(typeof r[k]==='number'&&!Number.isInteger(r[k])?r[k].toFixed(2):r[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';
  }
  window.addEventListener('message',e=>{
    const m=e.data;
    if(m.type==='filteringPicked'){
      if(m.metadata){get('Metadata').value=m.path;get('Inspect').click();}
      else if(tools[m.index]){tools[m.index].path=m.path;renderTools();}return;
    }
    if(m.type==='filteringConfigLoaded'){
      pending=m.config;currentPath=m.path;get('Title').value=pending.title||'';get('Metadata').value=pending.metadata||'';tools=pending.tools||[];renderTools();get('Inspect').click();return;
    }
    if(m.type==='filteringConfigSaved'){currentPath=m.path;get('Status').textContent='Saved '+m.path;return;}
    if(m.type==='filteringColumns'){
      const old=pending||request();
      for(const [id,key] of [['Id','idColumn'],['Group','groupColumn']]){
        get(id).innerHTML=m.result.columns.map(c=>'<option>'+esc(c)+'</option>').join('');
        if(m.result.columns.includes(old[key]))get(id).value=old[key];
        else if(id==='Group')get(id).value=m.result.columns.find(c=>/class.*class|classyfire|^class$/i.test(c))||m.result.columns[0];
      }
      pending=null;get('Status').textContent=m.result.rows+' metadata records · select classification and Evaluate';return;
    }
    if(m.type==='filteringResult'){
      get('Chart').src='data:image/png;base64,'+m.result.png;get('Chart').hidden=false;table('Results',m.result.rows);table('Diagnostics',m.result.diagnostics);get('Status').textContent=m.result.outputDir?'Exported '+m.result.outputDir:m.result.sourceN+' metadata SpecIDs';return;
    }
    if(m.type==='filteringBusy'||m.type==='filteringError')get('Status').textContent=m.message;
  });
}
module.exports={filteringHtml,filteringClient};
