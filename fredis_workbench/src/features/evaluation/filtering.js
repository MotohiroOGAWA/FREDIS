function filteringHtml() {
  return `<section id="filteringEvaluation"><header><div><span class="eyebrow">FREDIS EVALUATION · PREPARATION</span><h1>Training Data Filtering</h1><p>Common = SpecIDs retained by every tool. Percentages use metadata SpecIDs in each category as denominator.</p></div><div class="actions"><button id="filterLoad">Load Settings</button><button id="filterSave">Save Settings</button></div></header><section class="panel"><label>Title<input id="filterTitle"></label><label>Metadata path (.msds / CSV / TSV / JSON / Parquet)<div class="path"><input id="filterMetadata"><button id="filterBrowseMetadata">Browse</button></div></label><button id="filterInspect">Load columns</button><label>SpecID column<select id="filterId"><option>SpecID</option></select></label><label>Classification column<select id="filterGroup"></select></label><h2>Tools</h2><div id="filterTools"></div><button id="filterAdd">＋ Add tool</button><div class="actions"><button id="filterRun" class="primary">Evaluate</button></div></section><p id="filterStatus" role="status"></p><section class="panel" id="filterPresentation" hidden><h2>Result display</h2><p>Adjust saved results, then export the displayed rows and chart. Common always refers to all evaluated tools.</p><div class="actions"><button id="filterGraphTab" aria-pressed="true">Graph</button><button id="filterListTab" aria-pressed="false">Data list</button></div><section id="filterGraphSettings"><label>Chart title<input id="filterDisplayTitle"></label><label>Common color<div style="display:flex;gap:8px"><input id="filterCommonColor" type="color" value="#666666" style="width:52px;height:36px;padding:3px"><input id="filterCommonHex" aria-label="Common HEX" value="#666666" pattern="#[0-9a-fA-F]{6}" maxlength="7" style="width:130px"></div></label><div class="size-row"><label>Group width (px)<input id="filterWidth" type="number" min="1" max="12000" step="1" value="60"></label><label>Padding between groups (px)<input id="filterPaddingWidth" type="number" min="0" max="12000" step="1" value="16"></label><label>100% bar height (px)<input id="filterHeight" type="number" min="200" max="12000" step="1" value="700"></label></div><label>Space above bars (px)<input id="filterTopPadding" type="number" min="0" max="2000" value="80"></label><p class="muted">Plot width = group width × selected groups + padding × (selected groups + 1). Height sets the 100% bar height; space above bars is added separately. Labels and legend add only the space they need. Overall counts as a group.</p><details><summary>Text and visibility</summary>${[['Title','titleSize',14,'showTitle'],['Category labels','categoryLabelSize',10,'showCategoryLabels'],['Axis ticks','tickLabelSize',10,'showTickLabels'],['Axis title','axisTitleSize',11,'showAxisTitle'],['Bar values','valueLabelSize',7,'showValues'],['Legend','legendSize',10,'showLegend']].map(([label,size,value,show])=>`<div class="size-row"><label>${label} font size<input id="filter${size}" type="number" min="1" max="96" value="${value}"></label><label class="check"><input id="filter${show}" type="checkbox" checked>Show ${label.toLowerCase()}</label></div>`).join('')}<label class="check"><input id="filtershowCounts" type="checkbox" checked>Show category N</label></details><label>Chart layout<select id="filterOrientation"><option value="vertical">Categories side by side (vertical bars)</option><option value="horizontal">Categories stacked (horizontal bars)</option></select></label></section><section id="filterListPage" hidden><p>Only checked categories appear in the graph. List filters never change checkmarks. Filters are combined with AND. Tool values are retention rates, not prediction accuracy. All-tool statistics exclude Common and include hidden tools.</p><h2>Sort priorities</h2><p>Choose any metadata column independently of the Category column. For saved evaluations, Load columns refreshes the available column names.</p><div id="filterSortRules"></div><button id="filterAddSort">＋ Add sort priority</button><h2>Filters</h2><div id="filterFilterRules"></div><button id="filterAddFilter">＋ Add filter</button><p>Click a column header to sort. Shift-click adds a priority.</p><div class="actions"><button id="filterApplyList" class="primary">Apply sort and filters</button><button id="filterSelectShown">Select all</button><button id="filterUnselectShown">Unselect all</button><span id="filterSelectionCount"></span></div><div style="overflow:auto;max-height:65vh"><table id="filterCategoryResults"></table></div><details><summary>Per-tool counts and ID diagnostics</summary><div style="overflow:auto"><table id="filterResults"></table><table id="filterDiagnostics"></table></div></details></section><p id="filterApplyError" class="error" role="alert" hidden></p><div class="actions"><button id="filterApply">Apply display</button><button id="filterExportTsv">Export TSV</button><button id="filterExportPng">Export PNG</button><button id="filterExportSvg">Export SVG</button></div></section><div id="filterZoomControls" class="actions" style="margin:10px 0;align-items:center"><button id="filterZoomOut" aria-label="Zoom out">−</button><output id="filterZoomLevel" aria-live="polite">100%</output><button id="filterZoomIn" aria-label="Zoom in">＋</button><button id="filterZoomReset">Reset to 100%</button></div><section id="filterGraphPage" style="overflow:auto;max-width:100%;max-height:75vh" tabindex="0" aria-label="Scrollable evaluation graph"><img id="filterChart" style="display:block;max-width:none;width:auto;height:auto" hidden alt="Retention percentages by category and tool"></section></section>`;
}
function filteringClient() {
  const get = id => document.getElementById('filter'+id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let tools = [], currentPath, pending, cached = null, savedView = {}, sortRules = [], filterRules = [], fields = [], resultPage = 'graph', visibleCategoryRows = [];
  let checkedCategories=new Set(), showOverall=true, zoomPercent=100;
  const view = () => cached ? ({title:get('DisplayTitle').value,categories:[...checkedCategories],tools:cached.series.map(s=>s.name),showOverall,minN:0,sortRules:sortRules.map(r=>({...r})),filters:filterRules.map(r=>({...r})),orientation:get('Orientation').value,groupWidth:Number(get('Width').value),paddingWidth:Number(get('PaddingWidth').value),...textSettings(),height:Number(get('Height').value),topPadding:Number(get('TopPadding').value),colors:{...Object.fromEntries(cached.series.map(s=>[s.name,s.color])),...savedView.colors,...Object.fromEntries(tools.map(t=>[t.name,t.color])),Common:get('CommonHex').value},page:resultPage,zoomPercent}) : savedView;
  const fontKeys=['titleSize','categoryLabelSize','tickLabelSize','axisTitleSize','valueLabelSize','legendSize'];
  const showKeys=['showTitle','showCategoryLabels','showTickLabels','showAxisTitle','showValues','showLegend','showCounts'];
  const textSettings=()=>({...Object.fromEntries(fontKeys.map(k=>[k,Number(get(k).value)])),...Object.fromEntries(showKeys.map(k=>[k,get(k).checked]))});
  let pendingSnapshot='';
  const snapshot=()=>JSON.stringify({...view(),page:undefined,zoomPercent:undefined});
  function applyState(state){
    if(state!=='Failed · retry')get('ApplyError').hidden=true;
    for(const [id,label] of [['Apply','Apply display'],['ApplyList','Apply sort and filters']]){
      get(id).textContent=label+' · '+state;
      get(id).disabled=state==='Applying…';
    }
  }
  get('Presentation').addEventListener('input',()=>{if(cached)applyState('Pending changes');});
  get('Presentation').addEventListener('click',e=>{if(e.target.closest('[data-action],#filterAddSort,#filterAddFilter'))applyState('Pending changes');});
  const request = () => ({title:get('Title').value,metadata:get('Metadata').value,idColumn:get('Id').value,groupColumn:get('Group').value,tools:tools.map(t=>({...t})),view:view()});
  function showPage(page) {
    resultPage=page;
    get('GraphPage').hidden=page!=='graph';get('GraphSettings').hidden=page!=='graph';get('ZoomControls').hidden=page!=='graph';get('ListPage').hidden=page!=='list';
    get('GraphTab').setAttribute('aria-pressed',String(page==='graph'));get('ListTab').setAttribute('aria-pressed',String(page==='list'));get('GraphTab').textContent=(page==='graph'?'✓ ':'')+'Graph';get('ListTab').textContent=(page==='list'?'✓ ':'')+'Data list';
  }
  get('GraphTab').onclick=()=>showPage('graph');get('ListTab').onclick=()=>showPage('list');
  function setZoom(percent){
    const viewport=get('GraphPage'), previous=zoomPercent;
    const centerX=(viewport.scrollLeft+viewport.clientWidth/2)/previous;
    const centerY=(viewport.scrollTop+viewport.clientHeight/2)/previous;
    zoomPercent=Math.max(10,Math.min(400,Number(percent)||100));
    get('Chart').style.zoom=String(zoomPercent/100);
    get('ZoomLevel').textContent=zoomPercent+'%';
    get('ZoomOut').disabled=zoomPercent<=10;get('ZoomIn').disabled=zoomPercent>=400;
    viewport.scrollLeft=centerX*zoomPercent-viewport.clientWidth/2;
    viewport.scrollTop=centerY*zoomPercent-viewport.clientHeight/2;
  }
  get('ZoomOut').onclick=()=>setZoom(zoomPercent-25);
  get('ZoomIn').onclick=()=>setZoom(zoomPercent+25);
  get('ZoomReset').onclick=()=>setZoom(100);

  get('CommonColor').oninput=()=>{get('CommonHex').value=get('CommonColor').value;get('CommonHex').setCustomValidity('');};
  get('CommonHex').oninput=()=>{const valid=/^#[0-9a-fA-F]{6}$/.test(get('CommonHex').value);get('CommonHex').setCustomValidity(valid?'':'Enter # followed by six hexadecimal digits.');if(valid)get('CommonColor').value=get('CommonHex').value;};
  function renderRules() {
    const options=selected=>fields.map(f=>'<option value="'+esc(f.id)+'" '+(selected===f.id?'selected':'')+'>'+esc(f.label)+'</option>').join('');
    get('SortRules').innerHTML=sortRules.map((r,i)=>'<div data-rule="'+i+'" style="display:flex;gap:6px;align-items:center"><span>'+(i+1)+'</span><select data-key="field" aria-label="Sort field">'+options(r.field)+'</select><select data-key="direction" aria-label="Sort direction"><option value="asc" '+(r.direction==='asc'?'selected':'')+'>Ascending</option><option value="desc" '+(r.direction==='desc'?'selected':'')+'>Descending</option></select><button data-action="up" aria-label="Move priority up">↑</button><button data-action="down" aria-label="Move priority down">↓</button><button data-action="remove">Remove</button></div>').join('');
    get('FilterRules').innerHTML=filterRules.map((r,i)=>'<div data-rule="'+i+'" style="display:flex;gap:6px;align-items:center"><select data-key="field" aria-label="Filter field">'+options(r.field)+'</select><select data-key="op" aria-label="Filter operator">'+['>=','<=','>','<','==','!=','contains','notContains'].map(op=>'<option value="'+esc(op)+'" '+(r.op===op?'selected':'')+'>'+esc(op)+'</option>').join('')+'</select><input data-key="value" aria-label="Filter value" value="'+esc(r.value)+'"><button data-action="remove">Remove</button></div>').join('');
  }
  for(const [id,rules] of [['SortRules',sortRules],['FilterRules',filterRules]]) {
    // Arrays are replaced when settings load, so resolve them for each event.
    const current=()=>id==='SortRules'?sortRules:filterRules;
    get(id).oninput=e=>{const row=e.target.closest('[data-rule]');if(row&&e.target.dataset.key)current()[Number(row.dataset.rule)][e.target.dataset.key]=e.target.value;};
    get(id).onclick=e=>{const row=e.target.closest('[data-rule]'),action=e.target.dataset.action;if(!row||!action)return;const items=current(),i=Number(row.dataset.rule);if(action==='remove')items.splice(i,1);else{const to=i+(action==='up'?-1:1);if(to>=0&&to<items.length)[items[i],items[to]]=[items[to],items[i]];}renderRules();};
  }
  get('AddSort').onclick=()=>{sortRules.push({field:'category',direction:'asc'});renderRules();};
  get('AddFilter').onclick=()=>{filterRules.push({field:'totalN',op:'>=',value:0});renderRules();};
  get('CategoryResults').onclick=e=>{const header=e.target.closest('[data-sort-field]');if(!header)return;const field=header.dataset.sortField,existing=sortRules.find(r=>r.field===field),direction=existing?.direction==='asc'?'desc':'asc';if(e.shiftKey){if(existing)existing.direction=direction;else sortRules.push({field,direction});}else sortRules=[{field,direction}];renderRules();get('Apply').click();};
  let previewTimer;
  function queuePreview(){clearTimeout(previewTimer);previewTimer=setTimeout(()=>get('Apply').click(),200);}
  function updateSelectionCount(){
    const count=new Set(cached.rows.filter(r=>r.scope==='category').map(r=>r.category)).size;
    get('SelectionCount').textContent=checkedCategories.size+' / '+count+' categories checked';
  }
  function setChecked(scope,category,checked){
    if(scope==='overall')showOverall=checked;
    else if(checked)checkedCategories.add(category);else checkedCategories.delete(category);
  }
  function selectShown(checked){
    for(const row of visibleCategoryRows)setChecked(row.scope,row.category,checked);
    categoryTable(visibleCategoryRows);get('Apply').click();
  }
  get('SelectShown').onclick=()=>selectShown(true);get('UnselectShown').onclick=()=>selectShown(false);
  get('CategoryResults').onchange=e=>{if(!e.target.hasAttribute('data-category-check'))return;const row=visibleCategoryRows[Number(e.target.dataset.categoryCheck)];if(row){setChecked(row.scope,row.category,e.target.checked);updateSelectionCount();queuePreview();}};
  function categoryTable(rows) {
    visibleCategoryRows=rows;
    const visible=new Set(cached.series.map(s=>s.name));
    const checked=checkedCategories;
    const metadataFields=new Set([...sortRules,...filterRules].map(r=>r.field).filter(id=>id.startsWith('metadata:')));
    const columns=[{id:'category',label:'Category'},{id:'totalN',label:'Original N'},{id:'distribution',label:'Distribution'},{id:'scope',label:'Scope'},...fields.filter(f=>!['category','totalN'].includes(f.id)&&f.cached!==false&&(!f.id.startsWith('metadata:')||metadataFields.has(f.id))&&(!f.id.startsWith('tool:')||[...visible].some(name=>['retainedPercent','retainedN','excludedN'].some(metric=>f.id==='tool:'+name+':'+metric))))];
    const colors=view().colors;
    function distribution(row){
      const series=cached.series;
      const width=100/Math.max(1,series.length);
      return '<svg width="170" height="64" viewBox="0 0 140 54" role="img" aria-label="Tool retention distribution including Common; scale 0 to 100 percent">'+series.map((s,i)=>{const value=Number(row['tool:'+s.name+':retainedPercent'])||0;return '<rect x="'+(30+i*width)+'" y="'+(36-value*.3)+'" width="'+Math.max(1,width-1)+'" height="'+(value*.3)+'" fill="'+esc(colors[s.name]||s.color)+'"><title>'+esc(s.name)+': '+value.toFixed(2)+'%</title></rect><text x="'+(30+(i+.5)*width)+'" y="45" text-anchor="middle" font-size="5" fill="currentColor">'+esc(s.name)+'</text>';}).join('')+'<path d="M30 6H130 M30 36H130" stroke="currentColor" stroke-width=".6" fill="none"/><text x="27" y="8" text-anchor="end" font-size="7" fill="currentColor">100%</text><text x="27" y="38" text-anchor="end" font-size="7" fill="currentColor">0%</text></svg>';

    }
    get('CategoryResults').innerHTML='<thead><tr><th>Show</th>'+columns.map(f=>'<th><button '+(['scope','distribution'].includes(f.id)?'':'data-sort-field="'+esc(f.id)+'"')+'>'+esc(f.label)+'</button></th>').join('')+'</tr></thead><tbody>'+rows.map((row,i)=>'<tr><td><input type="checkbox" data-category-check="'+i+'" aria-label="Show '+esc(row.category)+'" '+((row.scope==='overall'?showOverall:checked.has(row.category))?'checked':'')+' style="width:auto;margin:0"></td>'+columns.map(f=>{if(f.id==='distribution')return '<td>'+distribution(row)+'</td>';const v=row[f.id];return '<td>'+esc(Array.isArray(v)?v.join(' | '):typeof v==='number'&&!Number.isInteger(v)?v.toFixed(2):v??'—')+'</td>';}).join('')+'</tr>').join('')+'</tbody>';
    updateSelectionCount();
  }
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
  get('ApplyList').onclick=()=>get('Apply').click();
  get('Apply').onclick=()=>{clearTimeout(previewTimer);pendingSnapshot=snapshot();applyState('Applying…');vscode.postMessage({type:'filteringView',view:view(),metadataConfig:request()});};
  for(const format of ['Tsv','Png','Svg'])get('Export'+format).onclick=()=>vscode.postMessage({type:'filteringExport',view:view(),metadataConfig:request(),format:format.toLowerCase()});
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
      if(cached){
        cached.availableMetadataColumns=m.result.columns.filter(c=>c!==get('Id').value);
        fields=fields.filter(f=>!f.id.startsWith('metadata:'));
        fields.splice(2,0,...cached.availableMetadataColumns.map(c=>({id:'metadata:'+c,label:'Metadata · '+c,type:'text',cached:(cached.metadataColumns||[]).includes(c)})));
        renderRules();
      }
      pending=null;get('Status').textContent=m.result.rows+' metadata records · select classification and Evaluate';return;
    }
    if(m.type==='filteringResult'){
      cached=m.result;
      if(m.result.settingsPath)currentPath=m.result.settingsPath;
      checkedCategories=new Set(cached.rows.filter(r=>r.scope==='category'&&(!savedView.categories||savedView.categories.includes(r.category))).map(r=>r.category));
      get('DisplayTitle').value=savedView.title||cached.title||'';
      showOverall=savedView.showOverall!==false;setZoom(savedView.zoomPercent??100);
      get('Width').value=savedView.groupWidth??60;get('PaddingWidth').value=savedView.paddingWidth??16;get('Height').value=savedView.height??700;get('TopPadding').value=savedView.topPadding??80;
      const defaults=[14,10,10,11,7,10];fontKeys.forEach((key,i)=>get(key).value=savedView[key]??defaults[i]);showKeys.forEach(key=>get(key).checked=savedView[key]!==false);
      get('Orientation').value=savedView.orientation||'vertical';get('CommonColor').value=get('CommonHex').value=savedView.colors?.Common||'#666666';
      fields=cached.fields||[{id:'category',label:'Category',type:'text'},{id:'totalN',label:'Original N',type:'number'}];
      sortRules=savedView.sortRules||[{field:savedView.sort==='count'?'totalN':savedView.sort==='retention'?'stats:min':'category',direction:savedView.sort==='count'?'desc':'asc'}];filterRules=savedView.filters||[];renderRules();
      categoryTable(cached.listRows||cached.categoryRows||[]);showPage(savedView.page||'graph');applyState('Applied');
      get('Presentation').hidden=false;
      get('Chart').src='data:image/png;base64,'+cached.png;get('Chart').hidden=false;table('Results',cached.listToolRows||cached.displayRows||cached.rows);table('Diagnostics',cached.diagnostics);
      get('Status').textContent=m.restored?'Opened saved evaluation':cached.outputDir?'Evaluation saved to '+cached.outputDir:cached.sourceN+' metadata SpecIDs';return;
    }
    if(m.type==='filteringPresentation'){
      applyState(!pendingSnapshot||snapshot()===pendingSnapshot?'Applied':'Pending changes');
      get('Chart').src='data:image/png;base64,'+m.result.png;table('Results',m.result.listToolRows||m.result.displayRows);fields=m.result.fields||fields;if(m.result.updatedResult)cached={...cached,...m.result.updatedResult};renderRules();categoryTable(m.result.listRows||m.result.categoryRows||[]);get('Status').textContent=m.exported?'Exported displayed results to '+m.exported:'Display updated · '+m.result.displayRows.length+' rows';return;
    }
    if(m.type==='filteringBusy'||m.type==='filteringError')get('Status').textContent=m.message;
    if(m.type==='filteringError'){applyState('Failed · retry');get('ApplyError').textContent=m.message;get('ApplyError').hidden=false;}
  });
}
module.exports={filteringHtml,filteringClient};
