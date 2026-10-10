const vscode=require('vscode');
const path=require('path');
const fs=require('fs');
const {spawn}=require('child_process');
const reactionPatternDocuments=require('../reaction-pattern-set/editor');
const parameterService=require('../../workbench/parameter-service');
const resultsView=require('./results-view');
const {parseConfiguration,saveConfiguration,loadConfiguration}=require('../../configuration');

let running=null;

function html(){return `<div id="reactionPatternAnalysisApp" data-app-panel="reactionPatternAnalysis" hidden class="preparation-page reaction-pattern-analysis-page">
<section><div class="section-title"><div><h2>Reaction Pattern Analysis</h2><p class="muted">Compare peak assignment coverage along a tree of reaction pattern sets.</p></div><div class="actions"><button type="button" id="rpaLoadConfig" class="config-drop">Load Configuration<br><small>Drop a reaction pattern analysis config or click to Browse</small></button><button type="button" id="rpaSaveConfig">Save Configuration</button><button type="button" data-rpa-tab="input" class="active">Input / Run</button><button type="button" data-rpa-tab="results">Results</button></div></div></section>
<div data-rpa-panel="input">
	<section><h3>Datasets</h3><p class="muted">Add raw datasets, prepared/mapped fragment data, or a previous Reaction Pattern Analysis result. Expand a row to configure that source.</p><div id="rpaDatasetDrop" class="rpa-drop-zone" tabindex="0">Drop sources here or click to browse</div><div id="rpaSourceRows" class="table-scroll"></div></section>
	<section><h3>Fragmenter</h3><label>Fragmenter or chemistry configuration<div class="path"><input id="rpaParams" data-path-kind="file"><button type="button" id="rpaPickParams">Browse</button><button type="button" id="rpaLoadPatterns">Apply</button><button type="button" id="rpaImportFragmenterPatterns" title="Add the reaction patterns of this configuration as a pattern set">Add its patterns as a set</button></div></label><p class="muted">Fragmenter settings (mass tolerance, adduct rules, search depth) come from this configuration. Its reaction patterns are not used unless you add them as a pattern set.</p><details><summary>Fragmenter and assignment options</summary><div class="grid"><label>Max reaction count<input id="rpaMaxReactions" type="number" min="1" value="2"></label><label>Mass tolerance<input id="rpaMassTolerance" value="20 ppm"></label><label>Max unique fragment SMILES<input id="rpaMaxFragments" type="number" min="-1" value="-1"></label><label>Max reaction combinations<input id="rpaMaxCombinations" type="number" min="-1" value="-1"></label><label>Minimum relative intensity<input id="rpaMinIntensity" type="number" min="0" max="1" step="any" value="0"></label></div><p id="rpaReactionHint" class="muted"></p></details></section>
	<section><h3>Pattern Sets</h3><p class="muted">Every analysis unit is a reaction pattern set. A single reaction pattern is added as a one-pattern set named after the pattern. One set can be placed at any number of tree nodes.</p><div id="rpaPatternDrop" class="rpa-drop-zone"><div class="path"><input id="rpaPatternSource" data-path-kind="file" placeholder="Pattern or pattern-set file"><button type="button" id="rpaBrowsePatternSource">Browse</button><button type="button" id="rpaApplyPatternSource">Add</button></div><span>or drop pattern / pattern-set files here</span></div><div class="actions"><button type="button" id="rpaNewPatternSet">+ New empty set</button></div><div id="rpaPatternSets" class="table-scroll"></div><div id="rpaPatternSetEditor" class="rpa-set-editor" hidden></div></section>
	<section><h3>Analysis Tree</h3><p class="muted">Each node analyses the union of the pattern sets on its path from its root (A → B → C analyses A, A + B and A + B + C). Each edge compares the parent condition with the parent condition + the child set; a root reports absolute coverage only. Several roots form independent series.</p><div class="actions"><label>Pattern set<select id="rpaRootSet"></select></label><button type="button" id="rpaAddRoot">+ Add Root</button></div><div id="rpaTree" class="rpa-tree"></div><p id="rpaTreeStatus" class="muted"></p></section>
	<section><h3>Execution</h3><div class="grid"><label>Output directory<div class="path"><input id="rpaOutput" data-path-kind="folder"><button type="button" id="rpaPickOutput">Browse</button></div></label><label>Processes<input id="rpaWorkers" type="number" min="1" value="1"></label><label>Worker chunk size<input id="rpaChunk" type="number" min="1" value="1"></label></div><label class="check"><input id="rpaExtend" type="checkbox">Add new nodes to the existing result in the output directory (calculated conditions are reused)</label><label class="check"><input id="rpaOverwrite" type="checkbox">Overwrite output</label><div class="actions"><button type="button" id="rpaCopyCommand">Copy command</button><button type="button" id="rpaRun" class="primary">Run analysis</button><button type="button" id="rpaStop" class="danger" disabled>Stop</button></div><p id="rpaRunStatus" class="status idle">Ready.</p><pre id="rpaLogs" class="analysis-log"></pre></section>
</div>
<div data-rpa-panel="results" hidden>
<section id="rpaResultDrop" class="rpa-drop-zone"><h3>Generated Result</h3><label>Result<div class="path"><input id="rpaResultPath" placeholder="result.reactionpatternanalysis.pft"><button type="button" id="rpaPickResult">Open</button></div></label><p id="rpaResultInfo" class="muted">Open or drop a result.reactionpatternanalysis.pft.</p><div id="rpaCalculationProgress" class="rpa-calculation-progress" hidden><div><span id="rpaProgressText">Preparing result…</span><span id="rpaProgressCount"></span></div><progress id="rpaProgress" max="100"></progress></div></section>
${resultsView.html()}
</div></div>`;}

function css(){return resultsView.css()+`.reaction-pattern-analysis-page textarea{width:100%;box-sizing:border-box}.analysis-log{max-height:13rem;overflow:auto;white-space:pre-wrap}.rpa-result-tabs{overflow-x:auto}.rpa-result-tabs button.active,[data-rpa-tab].active{border-color:var(--vscode-focusBorder);background:var(--vscode-button-secondaryBackground)}#rpaDistribution svg,#rpaCumulative svg{width:100%;height:auto;max-height:320px}.rpa-bar{fill:var(--vscode-charts-blue,#3794ff)}.rpa-axis{stroke:var(--vscode-foreground);opacity:.5}.rpa-line{fill:none;stroke:var(--vscode-charts-orange,#d18616);stroke-width:2}.reaction-pattern-analysis-page td code{white-space:nowrap}.rpa-drop-zone{border:1px dashed var(--vscode-panel-border);border-radius:6px;padding:12px;margin:8px 0}.rpa-drop-zone.path-drop-active{border-color:var(--vscode-focusBorder);background:var(--vscode-list-hoverBackground)}.rpa-source-detail{padding:10px;background:var(--vscode-editor-inactiveSelectionBackground)}.rpa-source-detail .grid select{width:100%}.rpa-tree ul{list-style:none;margin:0;padding-left:1.4rem;border-left:1px dashed var(--vscode-panel-border)}.rpa-tree>ul{padding-left:0;border-left:0}.rpa-tree-node{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.5rem;padding:.3rem .5rem;margin:.25rem 0;border:1px solid var(--vscode-panel-border);border-radius:6px}.rpa-tree-label small{margin-left:.5rem}.rpa-tree-actions{display:flex;flex-wrap:wrap;gap:.25rem;align-items:center}.rpa-badge{font-size:.75em;padding:0 .45em;border-radius:999px;background:var(--vscode-badge-background);color:var(--vscode-badge-foreground)}.rpa-set-editor{margin-top:10px;padding:10px;border:1px solid var(--vscode-focusBorder);border-radius:6px}.rpa-calculation-progress{margin-top:12px;padding:10px;border:1px solid var(--vscode-panel-border);border-radius:6px}.rpa-calculation-progress>div{display:flex;justify-content:space-between;gap:12px;margin-bottom:6px}.rpa-calculation-progress progress{display:block;width:100%;height:8px;accent-color:var(--vscode-progressBar-background)}`;}

function script(options={}){return `(${client.toString()})();`+resultsView.script(options);}

function client(){
	 const el=id=>document.getElementById(id),esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let patternSets=[],treeNodes=[],computedNodes=new Set(),editingSet=null,analysisSources=[];
 const post=value=>vscode.postMessage(value);
 function tab(name){document.querySelectorAll('[data-rpa-tab]').forEach(x=>x.classList.toggle('active',x.dataset.rpaTab===name));document.querySelectorAll('[data-rpa-panel]').forEach(x=>x.hidden=x.dataset.rpaPanel!==name);}
 document.querySelectorAll('[data-rpa-tab]').forEach(x=>x.onclick=()=>tab(x.dataset.rpaTab));
 function resultTab(name){document.querySelectorAll('[data-rpa-result-tab]').forEach(x=>x.classList.toggle('active',x.dataset.rpaResultTab===name));document.querySelectorAll('[data-rpa-result-panel]').forEach(x=>x.hidden=x.dataset.rpaResultPanel!==name);}
 document.querySelectorAll('[data-rpa-result-tab]').forEach(x=>x.onclick=()=>resultTab(x.dataset.rpaResultTab));
	 function treeConfig(){return {patternSets:patternSets.map(({id,name,patterns,source})=>({id,name,patterns,...(source?{source}:{})})),nodes:treeNodes.map(({id,patternSet,parent})=>({id,patternSet,parent:parent||null}))};}
	 function config(){return {sources:analysisSources.map(({info,expanded,...row})=>row),params:el('rpaParams').value.trim(),outputDir:el('rpaOutput').value.trim(),tree:treeConfig(),maxReactionCount:Number(el('rpaMaxReactions').value),massTolerance:el('rpaMassTolerance').value.trim(),maxUniqueFragmentSmiles:Number(el('rpaMaxFragments').value),maxReactionCombinations:Number(el('rpaMaxCombinations').value),minimumRelativeIntensity:Number(el('rpaMinIntensity').value),numWorkers:Math.max(1,Number(el('rpaWorkers').value)||1),chunkSize:Number(el('rpaChunk').value),overwrite:el('rpaOverwrite').checked,extend:el('rpaExtend').checked};}
	 function applyConfig(value,sources,extra={}){
	  const fields={params:'rpaParams',outputDir:'rpaOutput',maxReactionCount:'rpaMaxReactions',massTolerance:'rpaMassTolerance',maxUniqueFragmentSmiles:'rpaMaxFragments',maxReactionCombinations:'rpaMaxCombinations',minimumRelativeIntensity:'rpaMinIntensity',numWorkers:'rpaWorkers',chunkSize:'rpaChunk'};
	  for(const [key,id]of Object.entries(fields))if(value[key]!==undefined)el(id).value=value[key];
	  for(const [key,id]of [['overwrite','rpaOverwrite'],['extend','rpaExtend']])if(typeof value[key]==='boolean')el(id).checked=value[key];
	  const tree=value.tree||{patternSets:[],nodes:[]};
	  patternSets=(tree.patternSets||[]).map(item=>({id:String(item.id),name:String(item.name||item.id),patterns:(item.patterns||[]).map(pattern=>({...pattern})),source:item.source||''}));
	  treeNodes=(tree.nodes||[]).map(item=>({id:String(item.id),patternSet:String(item.patternSet),parent:item.parent||null}));
	  computedNodes=new Set(extra.computedNodes||[]);if(typeof extra.extend==='boolean')el('rpaExtend').checked=extra.extend;editingSet=null;
	  analysisSources=(value.sources||[]).map((row,index)=>{const info=(sources||[])[index]||{};return {...row,path:info.path||row.path,kind:info.kind||row.kind,info,expanded:false};});
	  renderPatternSets();renderTree();renderSources();updateReactionHint();
	 }
	 const setName=id=>patternSets.find(item=>item.id===id)?.name||id;
	 const nodeUses=id=>treeNodes.filter(node=>node.patternSet===id).length;
	 function uniqueId(base,taken){base=String(base||'set').trim()||'set';let value=base,index=2;while(taken.includes(value))value=base+'-'+index++;return value;}
	 function nodePath(node){const path=[];for(let current=node;current;current=treeNodes.find(item=>item.id===current.parent))path.unshift(current);return path;}
	 function conditionLabel(node){return [...new Set(nodePath(node).map(item=>setName(item.patternSet)))].join(' + ');}
	 // A child whose patterns are all already on its path compares a condition with itself.
	 function addsNothing(node){if(!node.parent)return false;const before=new Set(nodePath(treeNodes.find(item=>item.id===node.parent)).flatMap(item=>(patternSets.find(set=>set.id===item.patternSet)?.patterns||[]).map(pattern=>pattern.name)));return (patternSets.find(set=>set.id===node.patternSet)?.patterns||[]).every(pattern=>before.has(pattern.name));}
	 function setOptions(selected){return patternSets.map(item=>'<option value="'+esc(item.id)+'" '+(item.id===selected?'selected':'')+'>'+esc(item.name)+' ('+item.patterns.length+')</option>').join('');}
	 function fileName(value){return String(value||'').split(/[\\/]/).pop();}
	 function renderPatternSets(){
	  el('rpaPatternSets').innerHTML=patternSets.length?'<table><thead><tr><th>Pattern set</th><th>Patterns</th><th>Tree nodes</th><th></th></tr></thead><tbody>'+patternSets.map(item=>'<tr'+(item.id===editingSet?' class="selected"':'')+'><td>'+esc(item.name)+(item.source?' <small class="muted" title="'+esc(item.source)+'">'+esc(fileName(item.source))+'</small>':'')+'</td><td>'+item.patterns.length+'</td><td>'+nodeUses(item.id)+'</td><td><button type="button" data-rpa-set-edit="'+esc(item.id)+'">Edit</button><button type="button" data-rpa-set-delete="'+esc(item.id)+'" '+(nodeUses(item.id)?'disabled title="Used by tree nodes"':'')+'>Delete</button></td></tr>').join('')+'</tbody></table>':'<p class="muted">No pattern sets. Add a reaction pattern or pattern-set file.</p>';
	  const root=el('rpaRootSet'),old=root.value;root.innerHTML=setOptions(old);root.disabled=el('rpaAddRoot').disabled=!patternSets.length;
	  renderSetEditor();
	 }
	 function renderSetEditor(){
	  const target=el('rpaPatternSetEditor'),item=patternSets.find(value=>value.id===editingSet);
	  if(!item){target.hidden=true;target.innerHTML='';return;}
	  const computed=treeNodes.some(node=>node.patternSet===item.id&&computedNodes.has(node.id));
	  target.hidden=false;
	  target.innerHTML='<div class="section-title"><h4>Edit PatternSet</h4><button type="button" data-rpa-set-close>Close</button></div><label>Name<input id="rpaSetName" value="'+esc(item.name)+'"></label>'+(computed?'<p class="status warning">This set is used by calculated nodes. Changing its patterns creates new conditions that are calculated on the next run; renaming does not.</p>':'')+'<table><thead><tr><th>Reaction pattern</th><th>Reactant SMARTS</th><th>Products</th><th></th></tr></thead><tbody>'+(item.patterns.length?item.patterns.map((pattern,index)=>'<tr><td>'+esc(pattern.name)+'</td><td><code>'+esc(pattern.reactant_smarts)+'</code></td><td>'+esc((pattern.products||[]).length)+'</td><td><button type="button" data-rpa-pattern-remove="'+index+'">Remove</button></td></tr>').join(''):'<tr><td colspan="4" class="muted">No reaction pattern in this set.</td></tr>')+'</tbody></table><div id="rpaSetDrop" class="rpa-drop-zone"><div class="actions"><button type="button" id="rpaSetBrowse">Add patterns from file…</button><span class="muted">or drop pattern / pattern-set files here to add their patterns to this set</span></div></div>';
	  installFileDropZone(el('rpaSetDrop'),{kind:'file',multiple:true,onDrop:paths=>post({type:'reactionPatternAnalysis/applyPatternFiles',paths,target:item.id}),onError:dropError});
	 }
	 function renderTree(){
	  const children=parent=>treeNodes.filter(node=>(node.parent||null)===parent);
	  const branch=parent=>{const rows=children(parent);return rows.length?'<ul>'+rows.map(node=>'<li data-rpa-node="'+esc(node.id)+'"><div class="rpa-tree-node"><span class="rpa-tree-label"><b>'+esc(setName(node.patternSet))+'</b>'+(computedNodes.has(node.id)?' <span class="rpa-badge" title="Calculated in the existing result">calculated</span>':'')+(addsNothing(node)?' <span class="status warning" title="Every pattern of this set is already in the parent condition">adds no new pattern</span>':'')+' <small class="muted">'+(node.parent?esc(conditionLabel(treeNodes.find(item=>item.id===node.parent)))+' → +'+esc(setName(node.patternSet)):'root')+'</small></span><span class="rpa-tree-actions"><select data-rpa-child-set="'+esc(node.id)+'" aria-label="Pattern set for a child of '+esc(setName(node.patternSet))+'">'+setOptions('')+'</select><button type="button" data-rpa-add-child="'+esc(node.id)+'">+ Add Child</button><button type="button" data-rpa-edit-set="'+esc(node.patternSet)+'">Edit PatternSet</button><button type="button" data-rpa-delete-node="'+esc(node.id)+'">Delete Node</button></span></div>'+branch(node.id)+'</li>').join('')+'</ul>':'';};
	  el('rpaTree').innerHTML=treeNodes.length?branch(null):'<p class="muted">No nodes. Add a root pattern set.</p>';
	  el('rpaTreeStatus').textContent=treeNodes.length?treeNodes.filter(node=>!node.parent).length+' root(s) · '+treeNodes.filter(node=>node.parent).length+' edge(s)'+(computedNodes.size?' · '+treeNodes.filter(node=>!computedNodes.has(node.id)).length+' new node(s)':''):'';
	 }
	 function addNode(patternSet,parent){if(!patternSets.some(item=>item.id===patternSet))return;const ids=treeNodes.map(node=>node.id);let index=treeNodes.length+1,id='n'+index;while(ids.includes(id))id='n'+(++index);treeNodes.push({id,patternSet,parent:parent||null});renderTree();renderPatternSets();}
	 function deleteNode(id){const remove=new Set([id]);let grew=true;while(grew){grew=false;for(const node of treeNodes)if(node.parent&&remove.has(node.parent)&&!remove.has(node.id)){remove.add(node.id);grew=true;}}treeNodes=treeNodes.filter(node=>!remove.has(node.id));renderTree();renderPatternSets();}
	 function addPatternSets(sets,target){
	  const messages=[];
	  for(const incoming of sets||[]){
	   const patterns=(incoming.patterns||[]).map(pattern=>{const value={...pattern};delete value.pattern_id;return value;});
	   if(target){const item=patternSets.find(value=>value.id===target);if(!item)continue;for(const pattern of patterns){const existing=item.patterns.find(value=>value.name===pattern.name);if(existing){if(JSON.stringify(existing)!==JSON.stringify(pattern))messages.push('Different patterns share the name '+pattern.name+'.');continue;}item.patterns.push(pattern);}continue;}
	   const same=patternSets.find(value=>value.name===incoming.name&&JSON.stringify(value.patterns)===JSON.stringify(patterns));if(same)continue;
	   const id=uniqueId(incoming.id||incoming.name,patternSets.map(value=>value.id));patternSets.push({id,name:id===incoming.name||!incoming.name?id:incoming.name+' ('+id+')',patterns,source:incoming.source||''});
	  }
	  if(messages.length){el('rpaRunStatus').textContent=messages.join(' ');el('rpaRunStatus').className='status error';}
	  renderPatternSets();renderTree();
	 }
	 function options(values,selected){return values.map(value=>'<option value="'+esc(value)+'" '+(value===selected?'selected':'')+'>'+esc(value)+'</option>').join('');}
	 function sourceColumns(value){const columns=value.columns||[],pick=(preferred,fragments)=>columns.includes(preferred)?preferred:(columns.find(column=>fragments.some(part=>column.toLowerCase().includes(part)))||columns[0]||preferred);return {smiles:pick('SMILES',['smiles']),adduct:pick('AdductType',['adduct','precursortype']),collision_energy:pick('CollisionEnergy',['collision','energy']),precursor_mz:pick('PrecursorMZ',['precursormz','precursor_mz','parentmass'])};}
	 function updateReactionHint(){const limit=Math.max(1,Number(el('rpaMaxReactions').value)||1);el('rpaReactionHint').textContent='A child pattern set extends its parent condition\'s histories at depths 0–'+(limit-1)+'; resulting histories are limited to '+limit+' reactions.';}el('rpaMaxReactions').oninput=updateReactionHint;
	 function renderSources(){el('rpaSourceRows').innerHTML=analysisSources.length?'<table><thead><tr><th></th><th>Type</th><th>Source</th><th>Records</th><th>Information</th><th></th></tr></thead><tbody>'+analysisSources.map((row,index)=>{const value=row.info||{},columns=value.columns||[],mapping=row.columns||{},conditionRows=value.conditions||[];const detail=row.expanded?'<tr><td colspan="6"><div class="rpa-source-detail">'+(row.kind==='dataset'?'<div class="grid">'+[['smiles','SMILES'],['adduct','Adduct'],['collision_energy','Collision energy'],['precursor_mz','Precursor m/z']].map(([key,label])=>'<label>'+label+'<select data-rpa-source-column="'+key+'" data-rpa-source-index="'+index+'">'+options(columns,mapping[key])+'</select></label>').join('')+'</div>':row.kind==='analysis'?'<p class="muted">The datasets and column mappings saved in this result are analysed again.</p>':'<p class="muted">The original dataset and record subset recorded by preparation are analysed.</p>')+'</div></td></tr>':'';return '<tr><td><button data-rpa-source-toggle="'+index+'">'+(row.expanded?'▾':'▸')+'</button></td><td>'+esc(row.kind||'checking')+'</td><td title="'+esc(row.path)+'">'+esc(value.label||row.path)+'</td><td>'+esc(value.records??'—')+'</td><td>'+esc(value.molecules!=null?value.molecules+' molecules':value.datasets!=null?value.datasets+' datasets':(columns.length?columns.length+' columns':'—'))+'</td><td><button data-rpa-source-inspect="'+index+'">Inspect</button><button data-rpa-source-remove="'+index+'">Remove</button></td></tr>'+detail;}).join('')+'</tbody></table>':'<p class="muted">No sources added.</p>';}
 function calculationProgress(value={}){const container=el('rpaCalculationProgress'),bar=el('rpaProgress'),current=Number(value.current),total=Number(value.total),percent=Number(value.percent);container.hidden=false;el('rpaProgressText').textContent=value.message||'Calculating…';el('rpaProgressCount').textContent=Number.isFinite(current)&&Number.isFinite(total)&&total>0?current.toLocaleString()+' / '+total.toLocaleString():Number.isFinite(percent)?Math.round(percent)+'%':'';if(value.indeterminate||!Number.isFinite(percent))bar.removeAttribute('value');else bar.value=Math.max(0,Math.min(100,percent));}
	 el('rpaSaveConfig').onclick=()=>post({type:'reactionPatternAnalysis/saveConfig',config:config()});el('rpaLoadConfig').onclick=()=>post({type:'reactionPatternAnalysis/loadConfig'});
	 const dropError=message=>{el('rpaRunStatus').textContent=message;el('rpaRunStatus').className='status error';};
	 const configDrop=el('rpaLoadConfig');installFileDropZone(configDrop,{kind:'file',upload:false,onDrop:paths=>post({type:isResultPath(paths[0])?'reactionPatternAnalysis/openResult':'reactionPatternAnalysis/loadConfigFile',path:paths[0]}),onFiles:async files=>{const file=files[0];if(isResultPath(file.name))throw new Error('Reaction Pattern Analysis result must be dropped from VS Code Explorer so its companion compounds/ and details/ directories remain accessible.');if(file.size>2000000)throw new Error('Configuration exceeds 2 MB.');post({type:'reactionPatternAnalysis/loadConfigJSON',json:await file.text(),name:file.name});},onError:dropError});
	 const isResultPath=value=>String(value||'').toLowerCase().endsWith('.reactionpatternanalysis.pft');
	 const resultDrop=el('rpaResultDrop');installFileDropZone(resultDrop,{kind:'file',requirePath:true,requirePathMessage:'Reaction Pattern Analysis result must be dropped from VS Code Explorer so its companion compounds/ and details/ directories remain accessible.',onDrop:paths=>post({type:'reactionPatternAnalysis/openResult',path:paths[0]}),onError:message=>{el('rpaResultInfo').textContent=message;}});
	 el('rpaDatasetDrop').onclick=()=>post({type:'reactionPatternAnalysis/pickInputs'});el('rpaPickParams').onclick=()=>post({type:'reactionPatternAnalysis/pickParams'});el('rpaPickOutput').onclick=()=>post({type:'reactionPatternAnalysis/pickOutput'});el('rpaPickResult').onclick=()=>post({type:'reactionPatternAnalysis/pickResult'});el('rpaLoadPatterns').onclick=()=>post({type:'reactionPatternAnalysis/loadPatterns',path:el('rpaParams').value});el('rpaBrowsePatternSource').onclick=()=>post({type:'reactionPatternAnalysis/pickPatternFiles'});el('rpaApplyPatternSource').onclick=()=>{const path=el('rpaPatternSource').value.trim();if(path)post({type:'reactionPatternAnalysis/applyPatternFiles',paths:[path]});};el('rpaCopyCommand').onclick=()=>post({type:'reactionPatternAnalysis/copy',config:config()});el('rpaRun').onclick=()=>post({type:'reactionPatternAnalysis/run',config:config()});el('rpaStop').onclick=()=>post({type:'reactionPatternAnalysis/stop'});
	 for(const [zone,type]of [[el('rpaDatasetDrop'),'sources'],[el('rpaPatternDrop'),'patterns']])installFileDropZone(zone,{kind:'file',multiple:true,onDrop:paths=>post({type:type==='sources'?'reactionPatternAnalysis/addSources':'reactionPatternAnalysis/applyPatternFiles',paths}),onError:dropError});
	 // Browse applies the picked Fragmenter / pattern files immediately, so drops on these fields do too.
	 for(const element of [el('rpaParams'),el('rpaPickParams')])installPathInputDrop(el('rpaParams'),{element,kind:'file',upload:false,onDrop:()=>el('rpaLoadPatterns').click(),onError:dropError});
	 for(const element of [el('rpaPatternSource'),el('rpaBrowsePatternSource')])installFileDropZone(element,{kind:'file',multiple:true,onDrop:paths=>{applyPickedPath(el('rpaPatternSource'),paths.at(-1));post({type:'reactionPatternAnalysis/applyPatternFiles',paths});},onError:dropError});
	 el('rpaSourceRows').onclick=event=>{const button=event.target.closest('button');if(!button)return;for(const key of ['sourceToggle','sourceInspect','sourceRemove'])if(button.dataset['ca'+key[0].toUpperCase()+key.slice(1)]!==undefined){const index=Number(button.dataset['ca'+key[0].toUpperCase()+key.slice(1)]);if(key==='sourceRemove')analysisSources.splice(index,1);else if(key==='sourceToggle')analysisSources[index].expanded=!analysisSources[index].expanded;else post({type:'reactionPatternAnalysis/inspectSource',index,path:analysisSources[index].path,kind:analysisSources[index].kind});renderSources();return;}};
	 el('rpaSourceRows').onchange=event=>{const index=Number(event.target.dataset.rpaSourceIndex??event.target.dataset.rpaSourceCondition);if(event.target.dataset.rpaSourceColumn){analysisSources[index].columns[event.target.dataset.rpaSourceColumn]=event.target.value;}else if(event.target.dataset.rpaSourceCondition!==undefined)analysisSources[index].condition=event.target.value;};
 el('rpaPatternSets').onclick=event=>{const button=event.target.closest('button');if(!button)return;if(button.dataset.rpaSetEdit!==undefined){editingSet=editingSet===button.dataset.rpaSetEdit?null:button.dataset.rpaSetEdit;renderPatternSets();}else if(button.dataset.rpaSetDelete!==undefined&&!nodeUses(button.dataset.rpaSetDelete)){patternSets=patternSets.filter(item=>item.id!==button.dataset.rpaSetDelete);if(editingSet===button.dataset.rpaSetDelete)editingSet=null;renderPatternSets();renderTree();}};
 el('rpaPatternSetEditor').onclick=event=>{const button=event.target.closest('button');if(!button)return;const item=patternSets.find(value=>value.id===editingSet);if(!item)return;if(button.dataset.rpaSetClose!==undefined){editingSet=null;renderPatternSets();}else if(button.dataset.rpaPatternRemove!==undefined){item.patterns.splice(Number(button.dataset.rpaPatternRemove),1);renderPatternSets();renderTree();}else if(button.id==='rpaSetBrowse')post({type:'reactionPatternAnalysis/pickPatternFiles',target:item.id});};
 el('rpaPatternSetEditor').onchange=event=>{if(event.target.id!=='rpaSetName')return;const item=patternSets.find(value=>value.id===editingSet);const name=event.target.value.trim();if(item&&name){item.name=name;renderPatternSets();renderTree();}};
 el('rpaTree').onclick=event=>{const button=event.target.closest('button');if(!button)return;if(button.dataset.rpaAddChild!==undefined){const select=[...el('rpaTree').querySelectorAll('[data-rpa-child-set]')].find(item=>item.dataset.rpaChildSet===button.dataset.rpaAddChild);addNode(select?.value,button.dataset.rpaAddChild);}else if(button.dataset.rpaDeleteNode!==undefined)deleteNode(button.dataset.rpaDeleteNode);else if(button.dataset.rpaEditSet!==undefined){editingSet=button.dataset.rpaEditSet;renderPatternSets();el('rpaPatternSetEditor').scrollIntoView?.({behavior:'smooth',block:'nearest'});}};
 el('rpaAddRoot').onclick=()=>addNode(el('rpaRootSet').value,null);
 el('rpaNewPatternSet').onclick=()=>{const id=uniqueId('new_set',patternSets.map(item=>item.id));patternSets.push({id,name:id,patterns:[],source:''});editingSet=id;renderPatternSets();};
 el('rpaImportFragmenterPatterns').onclick=()=>{const file=el('rpaParams').value.trim();if(file)post({type:'reactionPatternAnalysis/fragmenterPatternSet',path:file});};
	 window.addEventListener('message',event=>{const m=event.data;if(m.type==='reactionPatternAnalysis/patternSets')addPatternSets(m.sets,m.target);if(m.type==='reactionPatternAnalysis/defaultsLoaded'&&m.defaults){el('rpaMaxReactions').value=m.defaults.maxReactionCount;el('rpaMassTolerance').value=m.defaults.massTolerance;updateReactionHint();}});
	 window.addEventListener('message',event=>{const m=event.data;if(m.type==='reactionPatternAnalysis/configLoaded'){applyConfig(m.config,m.sources,m);el('rpaRunStatus').textContent='Loaded '+(m.path||'configuration')+'.';el('rpaRunStatus').className='status idle';}if(m.type==='reactionPatternAnalysis/sourceList'){for(const value of m.sources||[]){let row=analysisSources.find(item=>item.path===value.path);if(row)Object.assign(row,{kind:value.kind,info:value});else{row={path:value.path,kind:value.kind,columns:sourceColumns(value),condition:value.conditions?.[0]?.key,info:value,expanded:false};analysisSources.push(row);}}renderSources();}if(m.type==='reactionPatternAnalysis/sourceInfo'){const row=analysisSources[m.index];if(row){row.kind=m.source.kind;row.info=m.source;row.expanded=true;renderSources();}}if(m.type==='reactionPatternAnalysis/picked')el(m.field).value=m.path;if(m.type==='reactionPatternAnalysis/status'){el('rpaRunStatus').textContent=m.text;el('rpaRunStatus').className='status '+m.status;el('rpaRun').disabled=m.status==='running';el('rpaStop').disabled=m.status!=='running';if(m.result){el('rpaResultPath').value=m.result;tab('results');}}if(m.type==='reactionPatternAnalysis/log')el('rpaLogs').textContent=(el('rpaLogs').textContent+m.text).slice(-30000);if(m.type==='reactionPatternAnalysis/progress')calculationProgress(m);if(m.type==='reactionPatternAnalysis/resultOpened'){tab('results');calculationProgress({message:'Loaded result summaries.',percent:100});}if(m.type==='reactionPatternAnalysis/error'){el('rpaRunStatus').textContent=m.error;el('rpaRunStatus').className='status error';}if(m.type==='reactionPatternAnalysis/result')calculationProgress({message:'Result updated.',percent:100});});
 window.addEventListener('reactionPatternAnalysis/open',()=>{if(!el('rpaParams').value)post({type:'reactionPatternAnalysis/defaults'});});
	 renderPatternSets();renderTree();renderSources();updateReactionHint();post({type:'reactionPatternAnalysis/ready'});
}

function shellDisplay(parts){return parts.map(value=>/^[A-Za-z0-9_./:=+-]+$/.test(String(value))?String(value):`'${String(value).replace(/'/g,"'\\''")}'`).join(' ');}
function args(config){const result=['-m','fredis.cli','analysis','reaction-pattern'];for(const source of config.sources||[])result.push('--source-spec',JSON.stringify(source));result.push('--output-dir',config.outputDir);if(config.params)result.push('--params',config.params);if(config.tree)result.push('--tree-config-json',JSON.stringify(config.tree));const pairs={maxReactionCount:'--max-reaction-count',massTolerance:'--mass-tolerance',maxUniqueFragmentSmiles:'--max-unique-fragment-smiles',maxReactionCombinations:'--max-reaction-combinations',minimumRelativeIntensity:'--minimum-relative-intensity',numWorkers:'--num-workers',chunkSize:'--chunk-size'};for(const [key,flag]of Object.entries(pairs))if(config[key]!==undefined)result.push(flag,String(config[key]));if(config.extend)result.push('--extend','1');else result.push('--overwrite',config.overwrite?'1':'0');return result;}
function validateTree(tree){const nodes=tree?.nodes||[],sets=new Map((tree?.patternSets||[]).map(item=>[item.id,item]));if(!nodes.length)throw new Error('Add at least one node to the analysis tree.');for(const node of nodes){const set=sets.get(node.patternSet);if(!set)throw new Error('Tree node '+node.id+' uses an unknown pattern set.');if(!(set.patterns||[]).length)throw new Error('Pattern set '+set.name+' has no reaction pattern.');}}
function loadConfig(file){const value=JSON.parse(fs.readFileSync(file,'utf8'));const fragmenter=value.fragmenter_params||value;const builder=fragmenter.fragment_ion_tree_builder||fragmenter;return {patterns:builder.reaction_pattern_set?.patterns||[],defaults:{maxReactionCount:builder.max_reaction_count??2,massTolerance:fragmenter.mass_tolerance??'20 ppm'}};}
let queryBackend=null,queryBackendKey='',queryBuffer='',querySequence=0,queryPending=new Map(),queryErrors='';
function stopQueryBackend(error){const child=queryBackend;queryBackend=null;queryBackendKey='';if(child&&!child.killed)child.kill();for(const pending of queryPending.values())pending.reject(error||new Error(queryErrors||'Reaction Pattern Analysis result reader stopped.'));queryPending.clear();queryBuffer='';queryErrors='';}
function query(context,root,request,onProgress){return new Promise((resolve,reject)=>{const python=vscode.workspace.getConfiguration('fredis').get('pythonPath','python'),key=python+'\0'+root;if(!queryBackend||queryBackendKey!==key){if(queryBackend)stopQueryBackend();const child=spawn(python,[path.join(context.extensionPath,'src/features/reaction-pattern-analysis/backend.py'),'--server'],{cwd:root,env:{...process.env,PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)}});queryBackend=child;queryBackendKey=key;child.stdout.on('data',chunk=>{queryBuffer+=chunk.toString();for(;;){const end=queryBuffer.indexOf('\n');if(end<0)break;const line=queryBuffer.slice(0,end);queryBuffer=queryBuffer.slice(end+1);if(!line.trim())continue;let response;try{response=JSON.parse(line);}catch{queryErrors=(queryErrors+line+'\n').slice(-30000);continue;}try{const pending=queryPending.get(response.requestId);if(!pending)continue;if(response.progress){pending.onProgress?.(response.progress);continue;}queryPending.delete(response.requestId);response.ok?pending.resolve(response.result):pending.reject(new Error(response.error));}catch(error){stopQueryBackend(new Error(queryErrors||error.message));}}});child.stderr.on('data',chunk=>queryErrors=(queryErrors+chunk.toString()).slice(-30000));child.on('error',error=>{if(queryBackend===child)stopQueryBackend(error);});child.on('close',()=>{if(queryBackend===child)stopQueryBackend(new Error(queryErrors||'Reaction Pattern Analysis result reader exited.'));});}const requestId=++querySequence;queryPending.set(requestId,{resolve,reject,onProgress});queryBackend.stdin.write(JSON.stringify({...request,requestId})+'\n');});}
// A pattern-set file is one set; a single pattern is a one-pattern set named after it.
function patternSetFromFile(file){const value=JSON.parse(fs.readFileSync(file,'utf8'));const clean=pattern=>{const copy=JSON.parse(JSON.stringify(pattern));delete copy.pattern_id;return copy;};if(value&&typeof value==='object'&&value.reaction_pattern_set){const set=reactionPatternDocuments.normalizeDocument(value).reaction_pattern_set;const name=String(set.name||'').trim()||path.basename(file).split('.')[0];return {id:name,name,patterns:set.patterns.map(clean),source:file};}const pattern=clean(reactionPatternDocuments.normalizePattern(value));return {id:pattern.name,name:pattern.name,patterns:[pattern],source:file};}
function fragmenterPatternSet(file){const loaded=loadConfig(file);const name=path.basename(file).replace(/\.(json|pft)$/i,'');return {id:name,name,patterns:loaded.patterns.map(pattern=>{const copy=JSON.parse(JSON.stringify(pattern));delete copy.pattern_id;return copy;}),source:file};}
function parseAnalysisConfiguration(value){return parseConfiguration(value,'reaction-pattern-analysis');}

const RESULT_NAME='result.reactionpatternanalysis.pft',CHEMISTRY_CONFIG_NAME='reaction-pattern-analysis-chemistry-config.json';
// HDF5 results are binary: never JSON.parse them. A known suffix is enough;
// other HDF5 files (e.g. .msds datasets) are confirmed by the backend schema check.
async function isResultFile(context,root,file){
 const target=path.resolve(root,String(file||''));
 if(resultsView.isResultPath(target))return true;
 try{const handle=await fs.promises.open(target,'r');try{const buffer=Buffer.alloc(8);await handle.read(buffer,0,8,0);if(!buffer.equals(Buffer.from([0x89,0x48,0x44,0x46,0x0d,0x0a,0x1a,0x0a])))return false;}finally{await handle.close();}}catch{return false;}
 return !!(await query(context,root,{action:'detect',path:target})).isResult;
}
// Restore Input / Run from the HDF5 /configuration written by the run.
function resultConfiguration(resultPath,info){
 const configuration=info.configuration||{},config={...(configuration.run||{})};
 config.tree=configuration.tree||config.tree||{patternSets:[],nodes:[]};
 config.outputDir=path.dirname(resultPath);config.overwrite=false;config.extend=true;
 const savedChemistry=path.join(path.dirname(resultPath),CHEMISTRY_CONFIG_NAME);
 if((!config.params||!fs.existsSync(config.params))&&fs.existsSync(savedChemistry))config.params=savedChemistry;
 const datasets=info.datasets||[];
 const sources=(config.sources||[]).map(source=>{const rows=datasets.filter(row=>path.resolve(String(row.source_path||''))===path.resolve(String(source.path||'')));return {path:source.path,kind:source.kind||'dataset',label:rows[0]?.label||path.basename(String(source.path||'')),records:rows.length?rows.reduce((sum,row)=>sum+Number(row.valid_record_count||0),0):undefined,columns:Object.values(source.columns||{}),conditions:[]};});
 return {config,sources,computedNodes:(config.tree.nodes||[]).map(node=>node.id),extend:true};
}
async function openResult(context,root,file,post){
 const resultPath=path.resolve(root,String(file||''));
 const info=await query(context,root,{path:resultPath,action:'info'},progress=>post({type:'reactionPatternAnalysis/progress',...progress}));
 // Only A (HDF5 summaries) is read here; B/C load when a compound/peak is selected.
 post({type:'reactionPatternAnalysis/configLoaded',path:resultPath,...resultConfiguration(resultPath,info)});
 post({type:'reactionPatternAnalysis/resultOpened',path:resultPath,info});
}
function savedResultPath(root,config,file){
 const candidate=path.join(path.resolve(root,config.outputDir||path.dirname(file)),RESULT_NAME);
 return fs.existsSync(candidate)?candidate:null;
}

async function hydrateAnalysisConfiguration(context,root,config){
 const sources=[];for(const source of config.sources||[])sources.push(await query(context,root,{action:'inspectSource',path:source.path,kind:source.kind||'auto'}));
 return {config:{...config,tree:config.tree||{patternSets:[],nodes:[]}},sources};
}

function attach(panel,context,output,projectRoot,initialDocument=null){
 const post=value=>panel.webview.postMessage(value);
 panel.webview.onDidReceiveMessage(async message=>{
  if(!message?.type?.startsWith('reactionPatternAnalysis/'))return;
	  try{
	   const root=projectRoot(context),python=vscode.workspace.getConfiguration('fredis').get('pythonPath','python');
	   if(message.type==='reactionPatternAnalysis/ready'){
	    if(!initialDocument)return;
	    const file=initialDocument;initialDocument=null;
	    if(await isResultFile(context,root,file)){await openResult(context,root,file,post);return;}
	    const config=parseAnalysisConfiguration(JSON.parse(await fs.promises.readFile(file,'utf8')));
	    const savedResult=savedResultPath(root,config,file);
	    if(savedResult&&await isResultFile(context,root,savedResult)){post({type:'reactionPatternAnalysis/status',status:'completed',text:'Loaded saved Reaction Pattern Analysis result.',result:savedResult,opened:true});await openResult(context,root,savedResult,post);}
	    else post({type:'reactionPatternAnalysis/configLoaded',path:file,...await hydrateAnalysisConfiguration(context,root,config)});
	   }else if(message.type==='reactionPatternAnalysis/copySvg'){
	    const svg=String(message.svg||'');if(!/^<svg\b/.test(svg)||svg.length>5000000)throw new Error('Invalid or oversized SVG.');
	    await vscode.env.clipboard.writeText(svg);post({type:'reactionPatternAnalysis/svgExported',mode:'copy'});
	   }else if(message.type==='reactionPatternAnalysis/saveSvg'){
	    const svg=String(message.svg||'');if(!/^<svg\b/.test(svg)||svg.length>5000000)throw new Error('Invalid or oversized SVG.');
	    const name=String(message.name||'reaction-pattern-analysis-frequency.svg').replace(/[^A-Za-z0-9_.-]+/g,'_');
	    const selected=await vscode.window.showSaveDialog({filters:{'SVG image':['svg']},defaultUri:vscode.Uri.file(name.endsWith('.svg')?name:name+'.svg')});
	    if(selected){const target=selected.fsPath.toLowerCase().endsWith('.svg')?selected.fsPath:selected.fsPath+'.svg';await fs.promises.writeFile(target,svg);post({type:'reactionPatternAnalysis/svgExported',mode:'save',path:target});}
	   }else if(message.type==='reactionPatternAnalysis/saveConfig'){
	    const defaultName=path.join(message.config.outputDir||root,'config.reaction-pattern-analysis.pft');
	    const saved=await saveConfiguration('reaction-pattern-analysis',message.config,defaultName);
	    if(saved)post({type:'reactionPatternAnalysis/status',status:'idle',text:'Configuration saved to '+saved+'.'});
	   }else if(['reactionPatternAnalysis/loadConfig','reactionPatternAnalysis/loadConfigFile','reactionPatternAnalysis/loadConfigJSON'].includes(message.type)){
	    let loaded;
	    if(message.type==='reactionPatternAnalysis/loadConfig')loaded=await loadConfiguration('reaction-pattern-analysis');
	    else if(message.type==='reactionPatternAnalysis/loadConfigFile'){
	     const file=path.resolve(root,message.path);
	     if(await isResultFile(context,root,file)){await openResult(context,root,file,post);return;}
	     loaded={path:file,config:parseAnalysisConfiguration(JSON.parse(await fs.promises.readFile(file,'utf8')))};
	    }else{
	     if(typeof message.json!=='string'||message.json.length>2000000)throw new Error('Configuration exceeds 2 MB or is invalid.');
	     loaded={path:message.name||'Dropped configuration',config:parseAnalysisConfiguration(JSON.parse(message.json))};
	    }
	    if(loaded)post({type:'reactionPatternAnalysis/configLoaded',path:loaded.path,...await hydrateAnalysisConfiguration(context,root,loaded.config)});
	   }else if(message.type==='reactionPatternAnalysis/defaults'){
    // Only Fragmenter settings: an analysis starts without any reaction pattern set.
    const file=parameterService.defaultFragmenterPath(root);
    post({type:'reactionPatternAnalysis/picked',field:'rpaParams',path:file});
    post({type:'reactionPatternAnalysis/defaultsLoaded',defaults:loadConfig(file).defaults});
	   }else if(message.type==='reactionPatternAnalysis/pickInputs'){
	    const selected=await vscode.window.showOpenDialog({canSelectMany:true,canSelectFiles:true,canSelectFolders:true,filters:{'Datasets and reusable results':['msds','msp','mgf','tsv','csv','parquet','pt','json','pft']}});
	    if(selected){const sources=[];for(const item of selected)sources.push(await query(context,root,{action:'inspectSource',path:item.fsPath,kind:'auto'}));post({type:'reactionPatternAnalysis/sourceList',sources});}
	   }else if(message.type==='reactionPatternAnalysis/addSources'){
	    const sources=[];for(const item of message.paths||[])sources.push(await query(context,root,{action:'inspectSource',path:item,kind:'auto'}));post({type:'reactionPatternAnalysis/sourceList',sources});
	   }else if(message.type==='reactionPatternAnalysis/inspectSource'){
	    post({type:'reactionPatternAnalysis/sourceInfo',index:message.index,source:await query(context,root,{action:'inspectSource',path:message.path,kind:message.kind||'auto'})});
   }else if(message.type==='reactionPatternAnalysis/pickParams'){
    const selected=await vscode.window.showOpenDialog({canSelectMany:false,filters:{'FREDIS configuration':['json','pft']}});
    if(selected?.[0]){post({type:'reactionPatternAnalysis/picked',field:'rpaParams',path:selected[0].fsPath});post({type:'reactionPatternAnalysis/defaultsLoaded',defaults:loadConfig(selected[0].fsPath).defaults});}
	   }else if(message.type==='reactionPatternAnalysis/loadPatterns'){
	    const file=path.resolve(root,message.path);post({type:'reactionPatternAnalysis/defaultsLoaded',defaults:loadConfig(file).defaults});
	   }else if(message.type==='reactionPatternAnalysis/fragmenterPatternSet'){
	    const file=path.resolve(root,message.path),set=fragmenterPatternSet(file);if(!set.patterns.length)throw new Error('This configuration has no reaction patterns.');post({type:'reactionPatternAnalysis/patternSets',sets:[set]});
	   }else if(message.type==='reactionPatternAnalysis/pickPatternFiles'){
	    const selected=await vscode.window.showOpenDialog({canSelectMany:true,filters:{'FREDIS reaction pattern files':['pft']}});
	    if(selected?.length){const files=selected.map(x=>x.fsPath);post({type:'reactionPatternAnalysis/patternSets',sets:files.map(patternSetFromFile),target:message.target||null});if(!message.target)post({type:'reactionPatternAnalysis/picked',field:'rpaPatternSource',path:files.at(-1)});}
	   }else if(message.type==='reactionPatternAnalysis/applyPatternFiles'){
	    const files=(message.paths||[]).map(file=>path.resolve(root,file));post({type:'reactionPatternAnalysis/patternSets',sets:files.map(patternSetFromFile),target:message.target||null});
   }else if(message.type==='reactionPatternAnalysis/pickOutput'){
    const selected=await vscode.window.showOpenDialog({canSelectMany:false,canSelectFiles:false,canSelectFolders:true});if(selected?.[0])post({type:'reactionPatternAnalysis/picked',field:'rpaOutput',path:selected[0].fsPath});
	   }else if(message.type==='reactionPatternAnalysis/pickResult'){
	    const selected=await vscode.window.showOpenDialog({canSelectMany:false,filters:{'Reaction Pattern Analysis result':['pft']}});if(selected?.[0]){if(!await isResultFile(context,root,selected[0].fsPath))throw new Error('Not a Reaction Pattern Analysis result: '+selected[0].fsPath);await openResult(context,root,selected[0].fsPath,post);}
   }else if(message.type==='reactionPatternAnalysis/openResult'){
    await openResult(context,root,message.path,post);
   }else if(message.type==='reactionPatternAnalysis/copy'){
    validateTree(message.config.tree);const command=shellDisplay([python,...args(message.config)]);await vscode.env.clipboard.writeText(command);post({type:'reactionPatternAnalysis/status',status:'idle',text:'CLI command copied.'});
	   }else if(message.type==='reactionPatternAnalysis/query'){
	    post({type:'reactionPatternAnalysis/result',action:message.request.action,result:await query(context,root,message.request,progress=>post({type:'reactionPatternAnalysis/progress',action:message.request.action,...progress}))});
   }else if(message.type==='reactionPatternAnalysis/stop'){
    if(running){running.fredisCancelled=true;running.kill('SIGTERM');}
   }else if(message.type==='reactionPatternAnalysis/run'){
    if(running)throw new Error('A Reaction Pattern Analysis is already running.');
    const config=message.config;const target=path.resolve(root,config.outputDir||'');
	    if(!(config.sources||[]).length||!config.outputDir||!config.params)throw new Error('Sources, output directory and a Fragmenter configuration are required.');
	    validateTree(config.tree);
	    const hasResult=fs.existsSync(path.join(target,RESULT_NAME));
	    if(config.extend&&!hasResult)throw new Error('No Reaction Pattern Analysis result to add nodes to in '+target+'. Clear "Add new nodes to the existing result" to start a new analysis.');
	    if(!config.extend&&!config.overwrite&&fs.existsSync(target)&&fs.statSync(target).isDirectory()&&fs.readdirSync(target).length){const buttons=hasResult?['Add new nodes to the existing result','Delete directory and overwrite']:['Delete directory and overwrite'];const choice=await vscode.window.showWarningMessage(`Output directory is not empty: ${target}\n`+(hasResult?'It contains a Reaction Pattern Analysis result. Add new nodes to it (calculated conditions are reused), or delete the entire directory and recalculate everything?':'Overwrite deletes the entire directory and all of its contents. Continue?'),{modal:true},...buttons);if(!choice){post({type:'reactionPatternAnalysis/status',status:'idle',text:'Analysis cancelled. The output directory was not changed.'});return;}if(choice==='Add new nodes to the existing result')config.extend=true;else config.overwrite=true;}
    const command=args(config);
    post({type:'reactionPatternAnalysis/status',status:'running',text:config.extend?'Calculating new tree conditions…':'Generating fragment assignments…'});
    const child=spawn(python,command,{cwd:root,env:{...process.env,PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)}});running=child;output.appendLine('$ '+shellDisplay([python,...command]));let stdout='';
    child.stdout.on('data',chunk=>{stdout+=chunk;post({type:'reactionPatternAnalysis/log',text:chunk.toString()});});child.stderr.on('data',chunk=>{output.append(chunk.toString());post({type:'reactionPatternAnalysis/log',text:chunk.toString()});});child.on('error',error=>{post({type:'reactionPatternAnalysis/error',error:error.message});if(running===child)running=null;});child.on('close',code=>{const cancelled=!!child.fredisCancelled;if(running===child)running=null;if(code===0){let result=path.join(path.resolve(root,config.outputDir),RESULT_NAME);try{const last=JSON.parse(stdout.trim().split(/\r?\n/).at(-1));result=last.result||result;}catch{}post({type:'reactionPatternAnalysis/status',status:'completed',text:'Analysis completed.',result,opened:true});openResult(context,root,result,post).catch(error=>post({type:'reactionPatternAnalysis/error',error:error.message}));}else post({type:'reactionPatternAnalysis/status',status:cancelled?'idle':'error',text:cancelled?'Analysis stopped.':'Analysis failed with exit code '+code+'.'});});
   }
  }catch(error){post({type:'reactionPatternAnalysis/error',error:error.message});}
 });
 panel.onDidDispose(()=>{if(running){running.fredisCancelled=true;running.kill('SIGTERM');running=null;}if(queryBackend)stopQueryBackend();});
}

module.exports={html,css,script,attach,args,validateTree,patternSetFromFile,shellDisplay,parseAnalysisConfiguration,hydrateAnalysisConfiguration,resultConfiguration,isResultFile,openResult,savedResultPath};
