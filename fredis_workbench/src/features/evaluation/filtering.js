function filteringHtml() {
  return `<section id="filteringEvaluation"><header><div><span class="eyebrow">FREDIS EVALUATION · PREPARATION</span><h1>Training Data Filtering</h1><p>Common = SpecIDs retained by every tool. Percentages use metadata SpecIDs in each category as denominator.</p></div><div class="actions"><button id="filterLoad">Load Settings</button><button id="filterSave">Save Settings</button></div></header><section class="panel"><label>Title<input id="filterTitle"></label><label>Metadata path (.msds / CSV / TSV / JSON / Parquet)<div class="path"><input id="filterMetadata"><button id="filterBrowseMetadata">Browse</button></div></label><button id="filterInspect">Load columns</button><label>SpecID column<select id="filterId"><option>SpecID</option></select></label><label>Classification column<select id="filterGroup"></select></label><h2>Tools</h2><div id="filterTools"></div><button id="filterAdd">＋ Add tool</button><div class="actions"><button id="filterRun" class="primary">Evaluate</button></div></section><p id="filterStatus" role="status"></p><section class="panel" id="filterPresentation" hidden><h2>Result display</h2><p>Adjust saved results, then export the displayed rows and chart. Common always refers to all evaluated tools.</p><label>Chart title<input id="filterDisplayTitle"></label><label>Categories (Ctrl / Cmd to select multiple)<select id="filterCategories" multiple size="10"></select></label><label>Tools<select id="filterVisibleTools" multiple size="5"></select></label><label class="check"><input type="checkbox" id="filterOverall" checked>Show overall</label><label>Minimum category N<input id="filterMinN" type="number" min="0" value="0"></label><label>Order<select id="filterSort"><option value="name">Category name</option><option value="count">N descending</option><option value="retention">Retention ascending</option></select></label><div class="actions"><button id="filterApply">Apply display</button><button id="filterExportTsv">Export TSV</button><button id="filterExportPng">Export PNG</button><button id="filterExportSvg">Export SVG</button></div></section><img id="filterChart" style="max-width:100%" hidden alt="Retention percentages by category and tool"><div style="overflow:auto"><table id="filterResults"></table><table id="filterDiagnostics"></table></div></section>`;
}
function filteringClient() {
  const get = id => document.getElementById('filter'+id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let tools = [], currentPath, pending, cached = null, savedView = {};
  const view = () => cached ? ({title:get('DisplayTitle').value,categories:Array.from(get('Categories').selectedOptions,o=>o.value),tools:Array.from(get('VisibleTools').selectedOptions,o=>o.value),showOverall:get('Overall').checked,minN:Number(get('MinN').value),sort:get('Sort').value}) : savedView;
  const request = () => ({title:get('Title').value,metadata:get('Metadata').value,idColumn:get('Id').value,groupColumn:get('Group').value,tools:tools.map(t=>({...t})),view:view()});
  function renderTools() {
    get('Tools').innerHTML = tools.map((t,i)=>`<div class="panel" data-tool="${i}"><label>Tool name<input data-field="name" value="${esc(t.name)}"></label><label>SpecID list path (one ID per line)<div class="path"><input data-field="path" value="${esc(t.path)}"><button data-browse="${i}">Browse</button></div></label><label>Color<div style="display:flex;gap:8px;align-items:center"><input data-field="color" type="color" aria-label="Color palette" style="width:52px;height:36px;padding:3px;flex:none" value="${esc(t.color)}"><input data-color-hex type="text" aria-label="Color HEX" placeholder="#36c5a2" pattern="#[0-9a-fA-F]{6}" maxlength="7" style="width:130px" value="${esc(t.color)}"></div></label><button data-remove="${i}">Remove</button></div>`).join('');
  }
  get('Tools').oninput = e => {
    const row=e.target.closest('[data-tool]');
    if(!row)return;
    const tool=tools[Number(row.dataset.tool)];
    if(e.target.hasAttribute('data-color-hex')) {
      const value=e.target.value.trim();
      const valid=/^#[0-9a-fA-F]{6}$/.test(value);
      e.target.setCustomValidity(valid?'':'Enter a HEX color such as #36c5a2 (six hexadecimal digits).');
      tool.color=value;
      if(valid)row.querySelector('[data-field="color"]').value=value;
    } else if(e.target.dataset.field) {
      tool[e.target.dataset.field]=e.target.value;
      if(e.target.dataset.field==='color') {
        const hex=row.querySelector('[data-color-hex]');
        hex.value=e.target.value;
        hex.setCustomValidity('');
      }
    }
  };
  get('Tools').onclick = e => {
    if(e.target.dataset.remove!==undefined){tools.splice(Number(e.target.dataset.remove),1);renderTools();}
    if(e.target.dataset.browse!==undefined)vscode.postMessage({type:'filteringPick',index:Number(e.target.dataset.browse)});
  };
  get('Add').onclick=()=>{tools.push({name:'Tool '+(tools.length+1),path:'',color:'#36c5a2'});renderTools();};
  get('Add').click();
  get('BrowseMetadata').onclick=()=>vscode.postMessage({type:'filteringPick',metadata:true});
  get('Inspect').onclick=()=>vscode.postMessage({type:'filteringInspect',request:request()});
  get('Run').onclick=()=>{savedView=view();vscode.postMessage({type:'filteringRun',request:request()});};
  get('Apply').onclick=()=>vscode.postMessage({type:'filteringView',view:view()});
  for(const format of ['Tsv','Png','Svg'])get('Export'+format).onclick=()=>vscode.postMessage({type:'filteringExport',view:view(),format:format.toLowerCase()});
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
      pending=m.config;cached=null;savedView=pending.view||{};get('Presentation').hidden=true;get('Chart').hidden=true;table('Results',[]);table('Diagnostics',[]);currentPath=m.path;get('Title').value=pending.title||'';get('Metadata').value=pending.metadata||'';tools=pending.tools||[];renderTools();if(pending.resultPath){for(const [id,key] of [['Id','idColumn'],['Group','groupColumn']])get(id).innerHTML='<option>'+esc(pending[key]||'')+'</option>';get('Status').textContent='Loading saved results…';}else get('Inspect').click();return;
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
      cached=m.result;
      if(m.result.settingsPath)currentPath=m.result.settingsPath;
      const options=(id,values,selected)=>{get(id).innerHTML=values.map(v=>'<option value="'+esc(v)+'" '+(!selected||selected.includes(v)?'selected':'')+'>'+esc(v)+'</option>').join('');};
      options('Categories',[...new Set(cached.rows.filter(r=>r.scope==='category').map(r=>r.category))],savedView.categories);
      options('VisibleTools',cached.series.map(s=>s.name),savedView.tools);
      get('DisplayTitle').value=savedView.title||cached.title||'';
      get('Overall').checked=savedView.showOverall!==false;
      get('MinN').value=savedView.minN||0;get('Sort').value=savedView.sort||'name';
      get('Presentation').hidden=false;
      get('Chart').src='data:image/png;base64,'+cached.png;get('Chart').hidden=false;table('Results',cached.displayRows||cached.rows);table('Diagnostics',cached.diagnostics);
      get('Status').textContent=m.restored?'Opened saved evaluation':cached.outputDir?'Evaluation saved to '+cached.outputDir:cached.sourceN+' metadata SpecIDs';return;
    }
    if(m.type==='filteringPresentation'){
      get('Chart').src='data:image/png;base64,'+m.result.png;table('Results',m.result.displayRows);get('Status').textContent=m.exported?'Exported displayed results to '+m.exported:'Display updated · '+m.result.displayRows.length+' rows';return;
    }
    if(m.type==='filteringBusy'||m.type==='filteringError')get('Status').textContent=m.message;
  });
}
module.exports={filteringHtml,filteringClient};
