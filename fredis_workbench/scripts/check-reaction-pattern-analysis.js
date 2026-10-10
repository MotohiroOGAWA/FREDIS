const assert=require('assert'),fs=require('fs'),os=require('os'),path=require('path'),Module=require('module'),load=Module._load;
Module._load=function(name,...args){return name==='vscode'?{workspace:{getConfiguration:()=>({get:(key,fallback)=>fallback})}}:load.call(this,name,...args);};
const {JSDOM}=require('jsdom');
const editor=require('../src/features/reaction-pattern-analysis/editor');
const resultsView=require('../src/features/reaction-pattern-analysis/results-view');
const configuration=require('../src/configuration');
const manifest=require('../package.json');

const markup=editor.html(),source=editor.script();
for(const text of ['Reaction Pattern Analysis','Input / Run','Results','Pattern Sets','Analysis Tree','+ Add Root','+ New empty set','Data Filter','Reaction Combination Analysis','Copy command','Add new nodes to the existing result'])assert(markup.includes(text),text);
for(const text of ['Overall Summary','Tree Edges','Compound Summary','Compound / Peak View','Peak / Node Detail','Δ Peak Coverage (mean)','Δ Intensity Coverage (mean)','+ Add Child','Delete Node','Edit PatternSet'])assert(markup.includes(text)||source.includes(text),text);
for(const field of ['rpaResults','rpaResultDrop','rpaOverall','rpaResultTree','rpaEdgeTable','rpaEdgeDetail','rpaCompoundTable','rpaCompoundNodes','rpaDistributionDetail','rpaPeakTable','rpaAssignments'])assert(markup.includes(`id="${field}"`),field);
for(const field of ['rpaLoadConfig','rpaSaveConfig','rpaDatasetDrop','rpaSourceRows','rpaPatternSets','rpaPatternSetEditor','rpaTree','rpaRootSet','rpaAddRoot','rpaPatternDrop','rpaPatternSource','rpaWorkers','rpaExtend','rpaResultPath','rpaCalculationProgress','rpaProgress','rpaProgressText','rpaProgressCount'])assert(markup.includes(`id="${field}"`),field);
const cliConfig={sources:[{path:'a.msds',kind:'dataset',columns:{smiles:'SMILES'}}],outputDir:'out',params:'p.json',tree:{patternSets:[{id:'a',name:'a',patterns:[{name:'a',reactant_smarts:'C'}]}],nodes:[{id:'n1',patternSet:'a',parent:null}]},maxReactionCount:2,massTolerance:'20 ppm',maxUniqueFragmentSmiles:-1,maxReactionCombinations:-1,minimumRelativeIntensity:0,numWorkers:2,chunkSize:1,overwrite:true};
const cli=editor.args(cliConfig);
for(const flag of ['--source-spec','--tree-config-json','--num-workers','--max-reaction-count','--max-unique-fragment-smiles','--max-reaction-combinations','--overwrite'])assert(cli.includes(flag),flag);
// The tree travels inside the command: no external file is needed wherever the command runs.
assert.deepEqual(JSON.parse(cli[cli.indexOf('--tree-config-json')+1]),cliConfig.tree);
const extendCli=editor.args({...cliConfig,extend:true});assert(extendCli.includes('--extend')&&!extendCli.includes('--overwrite'));
assert(source.includes("reactionPatternAnalysis/query"));
assert(source.includes("reactionPatternAnalysis/ready"));
assert(String(editor.attach).includes('stopQueryBackend'));
assert(String(editor.attach).includes('config.reaction-pattern-analysis.pft'));
assert(String(editor.attach).includes('reactionPatternAnalysis/progress'));
assert(editor.hydrateAnalysisConfiguration.toString().includes('inspectSource'));
assert(String(editor.attach).includes('Delete directory and overwrite'));
assert.deepEqual(editor.parseAnalysisConfiguration({schema:'fredis.workbench.configuration',schemaVersion:1,kind:'reaction-pattern-analysis',config:{numWorkers:4}}),{numWorkers:4});
const resultEditor=manifest.contributes.customEditors.find(item=>item.viewType==='fredis.reactionPatternAnalysisResultEditor');
assert(resultEditor&&resultEditor.priority==='default'&&resultEditor.selector.some(item=>item.filenamePattern==='*.reactionpatternanalysis.pft'));
assert(manifest.activationEvents.includes('onCustomEditor:fredis.reactionPatternAnalysisResultEditor'));
assert.equal(manifest.contributes.configuration.properties['fredis.workbench.reactionPatternAnalysisPageSize'].default,20);
assert.equal(manifest.contributes.configuration.properties['fredis.workbench.reactionPatternAnalysisPeakPageSize'].default,20);
assert.equal(manifest.contributes.configuration.properties['fredis.workbench.reactionPatternAnalysisSmilesColumnWidth'].default,240);
assert(resultsView.isResultPath('/x/result.reactionpatternanalysis.pft')&&!resultsView.isResultPath('/x/config.reaction-pattern-analysis.pft'));
const configEditor=manifest.contributes.customEditors.find(item=>item.viewType==='fredis.reactionPatternAnalysisConfigEditor');
assert(configEditor&&configEditor.priority==='default');
assert(configEditor.selector.some(item=>item.filenamePattern==='*.reaction-pattern-analysis.pft'));
assert.equal(configuration.configurationSuffix('config.reaction-pattern-analysis.pft'),'.reaction-pattern-analysis.pft');
assert.equal(configuration.configurationSuffix('other.json'),'.json');
const messages=[];
const dom=new JSDOM(editor.html()+'<script>const vscode=acquireVsCodeApi();'+require('../src/webview-path-drop').script()+editor.script()+'</script>',{runScripts:'dangerously',beforeParse(window){window.acquireVsCodeApi=()=>({postMessage:value=>messages.push(value)});window.validateDroppedPaths=paths=>paths;window.HTMLElement.prototype.scrollIntoView=function(){};}});
(async()=>{
const window=dom.window,document=window.document,el=id=>document.getElementById(id);
const post=data=>window.dispatchEvent(new window.MessageEvent('message',{data}));
assert.equal(messages.shift().type,'reactionPatternAnalysis/ready');
// The analysis starts without any pattern set or node.
assert(el('rpaPatternSets').textContent.includes('No pattern sets'));assert(el('rpaTree').textContent.includes('No nodes'));assert(el('rpaAddRoot').disabled);
// A pattern-set file is one set, a single pattern a one-pattern set named after it.
post({type:'reactionPatternAnalysis/patternSets',sets:[
  {id:'nonring',name:'nonring',source:'/p/nonring.reactionset.pft',patterns:[{name:'a',reactant_smarts:'C',pattern_id:3},{name:'b',reactant_smarts:'O'}]},
  {id:'mclafferty',name:'mclafferty',source:'/p/m.reaction.pft',patterns:[{name:'mclafferty',reactant_smarts:'C=O'}]},
  {id:'flavonoids',name:'flavonoids',patterns:[{name:'flavone',reactant_smarts:'c1ccccc1'}]}]});
assert.equal(el('rpaPatternSets').querySelectorAll('tbody tr').length,3);assert.equal(el('rpaAddRoot').disabled,false);
// Adding the same set again does not duplicate it.
post({type:'reactionPatternAnalysis/patternSets',sets:[{id:'mclafferty',name:'mclafferty',patterns:[{name:'mclafferty',reactant_smarts:'C=O'}]}]});
assert.equal(el('rpaPatternSets').querySelectorAll('tbody tr').length,3);
// Build:  nonring ├─ mclafferty ─ flavonoids   └─ flavonoids   plus root mclafferty.
el('rpaRootSet').value='nonring';el('rpaAddRoot').click();
const nodeIds=()=>[...el('rpaTree').querySelectorAll('[data-rpa-node]')].map(item=>item.dataset.rpaNode);
const addChild=(parent,set)=>{const select=[...el('rpaTree').querySelectorAll('[data-rpa-child-set]')].find(item=>item.dataset.rpaChildSet===parent);select.value=set;[...el('rpaTree').querySelectorAll('[data-rpa-add-child]')].find(item=>item.dataset.rpaAddChild===parent).click();};
addChild('n1','mclafferty');addChild('n2','flavonoids');addChild('n1','flavonoids');
el('rpaRootSet').value='mclafferty';el('rpaAddRoot').click();
assert.deepEqual(nodeIds(),['n1','n2','n3','n4','n5']);
assert(el('rpaTree').textContent.includes('nonring + mclafferty → +flavonoids'));
assert.equal(el('rpaTreeStatus').textContent,'2 root(s) · 3 edge(s)');
// One set at several nodes; a used set cannot be deleted.
assert.equal(el('rpaPatternSets').querySelector('[data-rpa-set-delete="flavonoids"]').disabled,true);
assert.equal(el('rpaPatternSets').querySelectorAll('tbody tr')[2].children[2].textContent,'2');
// Edit PatternSet: view, remove and add patterns, rename.
[...el('rpaTree').querySelectorAll('[data-rpa-edit-set]')].find(item=>item.dataset.rpaEditSet==='nonring').click();
assert.equal(el('rpaPatternSetEditor').hidden,false);assert.equal(el('rpaPatternSetEditor').querySelectorAll('tbody tr').length,2);
el('rpaPatternSetEditor').querySelector('[data-rpa-pattern-remove="1"]').click();assert.equal(el('rpaPatternSetEditor').querySelectorAll('tbody tr').length,1);
messages.length=0;el('rpaSetBrowse').click();assert.deepEqual(messages.at(-1),{type:'reactionPatternAnalysis/pickPatternFiles',target:'nonring'});
post({type:'reactionPatternAnalysis/patternSets',target:'nonring',sets:[{id:'x',name:'x',patterns:[{name:'b',reactant_smarts:'O'},{name:'c',reactant_smarts:'N'}]}]});
assert.equal(el('rpaPatternSetEditor').querySelectorAll('tbody tr').length,3);
el('rpaSetName').value='nonring single bonds';el('rpaSetName').dispatchEvent(new window.Event('change',{bubbles:true}));
assert(el('rpaTree').textContent.includes('nonring single bonds'));
// Delete Node removes the subtree.
[...el('rpaTree').querySelectorAll('[data-rpa-delete-node]')].find(item=>item.dataset.rpaDeleteNode==='n5').click();assert.deepEqual(nodeIds(),['n1','n2','n3','n4']);
window.dispatchEvent(new window.MessageEvent('message',{data:{type:'reactionPatternAnalysis/sourceList',sources:[{path:'a.msds',kind:'dataset',label:'a.msds',records:3,columns:['SMILES','AdductType','CollisionEnergy','PrecursorMZ']}]}}));
assert.equal(el('rpaSourceRows').querySelectorAll('tbody tr').length,1);
el('rpaParams').value='model.json';el('rpaOutput').value='out';el('rpaWorkers').value='3';messages.length=0;el('rpaCopyCommand').click();
const sent=messages.at(-1).config;
assert.deepEqual(sent.tree.nodes,[{id:'n1',patternSet:'nonring',parent:null},{id:'n2',patternSet:'mclafferty',parent:'n1'},{id:'n3',patternSet:'flavonoids',parent:'n2'},{id:'n4',patternSet:'flavonoids',parent:'n1'}]);
assert.deepEqual(sent.tree.patternSets.find(item=>item.id==='nonring').patterns.map(item=>item.name),['a','b','c']);
assert(!('pattern_id' in sent.tree.patternSets[0].patterns[0]));assert.equal(sent.tree.patternSets[0].name,'nonring single bonds');
assert.equal(sent.sources[0].columns.smiles,'SMILES');assert.equal(sent.numWorkers,3);assert.equal(sent.extend,false);
el('rpaSaveConfig').click();assert.equal(messages.at(-1).type,'reactionPatternAnalysis/saveConfig');
const drop=new window.Event('drop',{bubbles:true,cancelable:true});Object.defineProperty(drop,'dataTransfer',{value:{types:['text/uri-list'],files:[],getData:type=>type==='text/uri-list'?'file:///tmp/analysis-config.json':''}});el('rpaLoadConfig').dispatchEvent(drop);await new Promise(resolve=>setTimeout(resolve,0));assert.equal(messages.at(-1).type,'reactionPatternAnalysis/loadConfigFile');assert.equal(messages.at(-1).path,'/tmp/analysis-config.json');
messages.length=0;el('rpaImportFragmenterPatterns').click();assert.deepEqual(messages.at(-1),{type:'reactionPatternAnalysis/fragmenterPatternSet',path:'model.json'});
window.dispatchEvent(new window.MessageEvent('message',{data:{type:'reactionPatternAnalysis/progress',message:'Loading saved result files',current:25,total:100,percent:22.5}}));
assert.equal(el('rpaCalculationProgress').hidden,false);assert.equal(el('rpaProgress').value,22.5);assert.equal(el('rpaProgressCount').textContent,'25 / 100');

// Opening a result restores Input / Run (tree included, nodes marked as calculated) and Results from A only.
const resultRoot=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-hdf5-result-')),resultFile=path.join(resultRoot,'result.reactionpatternanalysis.pft');
fs.writeFileSync(path.join(resultRoot,'reaction-pattern-analysis-chemistry-config.json'),'{}');
const tree={patternSets:[{id:'A',name:'A',patterns:[{name:'a',reactant_smarts:'C'}]},{id:'B',name:'B',patterns:[{name:'b',reactant_smarts:'C=O'}]}],nodes:[{id:'r',patternSet:'A',parent:null},{id:'e',patternSet:'B',parent:'r'}]};
const nodeSummary=[{node_row:0,node_id:'r',parent_node_row:-1,parent_node_id:'',pattern_set_id:'A',pattern_set_name:'A',depth:0,is_root:true,condition_id:0,parent_condition_id:-1,path_label:'A',condition_label:'A',parent_condition_label:'',added_patterns:['a'],applicable_compound_count:2,calculated_compound_count:2,overall_peak_coverage:0.25,overall_intensity_coverage:0.5,weighted_peak_coverage:0.25,mean_peak_coverage:0.2},
 {node_row:1,node_id:'e',parent_node_row:0,parent_node_id:'r',pattern_set_id:'B',pattern_set_name:'B',depth:1,is_root:false,condition_id:1,parent_condition_id:0,path_label:'A → B',condition_label:'A + B',parent_condition_label:'A',added_patterns:['b'],applicable_compound_count:1,calculated_compound_count:1,improved_compound_count:1,improved_sample_count:3,weighted_peak_coverage:0.45,parent_weighted_peak_coverage:0.25,weighted_delta_peak_coverage:0.2,weighted_delta_intensity_coverage:0.1,mean_delta_peak_coverage:0.2,mean_delta_intensity_coverage:0.05,newly_assigned_peak_count:7,overall_peak_coverage:0.3}];
const info={format:'hdf5',metadata:{analysis_status:'completed',created_at:'now'},counts:{compounds:2,sample_summary:5},
  configuration:{run:{sources:[{path:'/data/a.msds',kind:'dataset',columns:{smiles:'SMILES',adduct:'AdductType',collision_energy:'CE',precursor_mz:'PrecursorMZ'}}],params:'/moved/chemistry.json',outputDir:'/elsewhere',tree,maxReactionCount:3,numWorkers:4,chunkSize:2,massTolerance:'10 ppm'},tree},
  datasets:[{dataset_id:1,label:'a.msds',source_path:'/data/a.msds',valid_record_count:5}],
  tree:{patternSets:[{pattern_set_id:'A',name:'A',patterns:[{id:0,name:'a',reactant_smarts:'C'}]},{pattern_set_id:'B',name:'B',patterns:[{id:1,name:'b',reactant_smarts:'C=O'}]}]},
  overall:{datasets:1,compounds:2,samples:5,peaks:40,totalIntensity:12,patternSets:2,roots:1,edges:1,conditions:2,compoundsWithImprovement:1,samplesWithImprovement:3,newlyAssignedPeaks:7,newlyAssignedIntensity:1.5,calculatedConditionTrees:3},nodeSummary};
const restoredResult=editor.resultConfiguration(resultFile,info);
assert.deepEqual(restoredResult.config.tree,tree);assert.deepEqual(restoredResult.computedNodes,['r','e']);assert.equal(restoredResult.config.extend,true);
assert.equal(restoredResult.config.outputDir,resultRoot);
assert.equal(restoredResult.config.params,path.join(resultRoot,'reaction-pattern-analysis-chemistry-config.json'));assert.equal(restoredResult.sources[0].records,5);
messages.length=0;
el('rpaResultPath').value='';
post({type:'reactionPatternAnalysis/configLoaded',path:resultFile,...restoredResult});
post({type:'reactionPatternAnalysis/resultOpened',path:resultFile,info});
assert.equal(el('rpaWorkers').value,'4');assert.equal(el('rpaMaxReactions').value,'3');assert.equal(el('rpaExtend').checked,true);
assert.deepEqual(nodeIds(),['r','e']);assert.equal(el('rpaTree').querySelectorAll('.rpa-badge').length,2);
// Adding a node to a loaded result: the new node is reported as new and the run extends the result.
addChild('e','A');assert(el('rpaTreeStatus').textContent.includes('1 new node(s)'));
messages.length=0;el('rpaCopyCommand').click();assert.equal(messages.at(-1).config.extend,true);assert.equal(messages.at(-1).config.tree.nodes.length,3);
[...el('rpaTree').querySelectorAll('[data-rpa-edit-set]')].find(item=>item.dataset.rpaEditSet==='B').click();assert(el('rpaPatternSetEditor').textContent.includes('used by calculated nodes'));
messages.length=0;
assert(document.querySelector('[data-rpa-tab="results"]').classList.contains('active'));
assert.equal(el('rpaResults').hidden,false);assert.equal(el('rpaResultPath').value,resultFile);
assert.equal(el('rpaCompoundFilters').open,false);assert.equal(el('rpaPeakFilters').open,false);
assert(markup.indexOf('id="rpaCompoundFilters"')<markup.indexOf('id="rpaFrequencyMetric"')&&markup.indexOf('id="rpaFrequencyMetric"')<markup.indexOf('id="rpaCompoundTable"'));
assert(el('rpaOverall').textContent.includes('Tree edges'));assert.equal(el('rpaEdgeTable').querySelectorAll('tbody tr').length,2);
assert(el('rpaEdgeTable').textContent.includes('A → +B'));assert(el('rpaEdgeTable').textContent.includes('A (root)'));
assert(el('rpaEdgeTable').querySelector('[data-rpa-full-label="Delta peak coverage = child − parent (weighted)"]'));
assert.equal(el('rpaEdgeTable').querySelectorAll('tbody tr')[0].children[2].textContent,'—');  // a root has no delta
assert.equal(el('rpaEdgeTable').querySelectorAll('tbody tr')[1].children[2].textContent,'20.00%');
assert.equal(el('rpaResultTree').querySelectorAll('[data-rpa-tree-node]').length,2);
const queried=()=>messages.filter(value=>value.type==='reactionPatternAnalysis/query').map(value=>value.request);
el('rpaEdgeFilter').value='zzz';el('rpaEdgeFilter').dispatchEvent(new window.Event('input'));assert.equal(el('rpaEdgeTable').querySelectorAll('tbody tr').length,0);el('rpaEdgeFilter').value='';el('rpaEdgeFilter').dispatchEvent(new window.Event('input'));
messages.length=0;el('rpaResultTree').querySelector('[data-rpa-tree-node="1"]').click();
assert.deepEqual(queried().map(value=>value.action),['compoundList','scoreFrequency','nodeDistribution']);assert.equal(queried()[0].filters.nodeRow,'1');assert.equal(queried()[2].nodeRow,1);
const result=(action,value)=>window.dispatchEvent(new window.MessageEvent('message',{data:{type:'reactionPatternAnalysis/result',action,result:value}}));
const stats=metric=>[0,1].map(flag=>({node_row:1,metric,without_precursor:flag,n:4,mean:0.15,median:0.1,std:0.05,q25:0.05,q75:0.2,min:0,max:0.3,histogram_json:'{"edges":[0,0.5,1],"counts":[3,1]}'}));
result('nodeDistribution',{nodeRow:1,rows:[...stats('delta_peak_coverage'),...stats('peak_coverage')]});
const detail=el('rpaEdgeDetail').textContent;
for(const text of ['Parent condition','Child condition','Added pattern set','Δ peak / intensity coverage','A + B'])assert(detail.includes(text),text);
assert(el('rpaEdgeDetail').querySelector('svg rect.rpa-bar'));
result('scoreFrequency',{metric:'intensity_coverage',withoutPrecursor:false,bins:20,filteredSamples:4,filters:{nodeRow:'1'},edges:[0,0.5,1],series:[{key:'node:0',label:'A',nodeRow:0,n:4,counts:[3,1]},{key:'node:1',label:'A + B',nodeRow:1,n:4,counts:[2,2]}]});
assert(el('rpaFrequencyChart').querySelector('svg'));assert.equal(el('rpaFrequencySeries').querySelectorAll('input').length,2);
const completedOnly=el('rpaCompoundFilters').querySelector('[data-rpa-cfilter="completedOnly"]');completedOnly.checked=true;messages.length=0;el('rpaCompoundApply').click();assert.deepEqual(queried().map(value=>value.action),['compoundList','scoreFrequency','nodeDistribution']);assert.equal(queried()[1].filters.completedOnly,true);
el('rpaFrequencyBins').value='12';messages.length=0;el('rpaFrequencyBins').dispatchEvent(new window.Event('change'));assert.equal(queried()[0].bins,12);
messages.length=0;el('rpaFrequencyCopy').click();assert.equal(messages.at(-1).type,'reactionPatternAnalysis/copySvg');assert(messages.at(-1).svg.includes('<metadata>'));
result('compoundList',{total:1,offset:0,rows:[{compound_id:'cabc',canonical_smiles:'CCO',sample_count:4,peak_count:20,evaluated_node_count:2,improving_edge_count:1,new_peak_union_count:3,status:'completed',node:{status:'completed',mean_delta_peak_coverage:0.2,newly_assigned_peak_count_total:3}}]});
assert.equal(el('rpaCompoundTable').querySelectorAll('tbody tr').length,1);assert(el('rpaCompoundTable').textContent.includes('20.00%'));
assert(el('rpaCompoundTable').querySelector('.rpa-smiles-value'));assert(el('rpaCompoundTable').querySelector('.rpa-copy svg'));messages.length=0;el('rpaCompoundTable').querySelector('[data-rpa-copy-smiles]').click();assert.deepEqual(messages.at(-1),{type:'copyText',text:'CCO'});
messages.length=0;el('rpaCompoundTable').querySelector('[data-rpa-compound="cabc"]').click();
assert.deepEqual(queried().map(value=>[value.action,value.compoundId]),[['compoundSummary','cabc'],['compoundPeaks','cabc']]);
assert.equal(queried()[1].limit,20);assert.equal(queried()[1].offset,0);
result('compoundSummary',{compound:{compound_id:'cabc',canonical_smiles:'CCO',sample_count:4,peak_count:20,status:'completed'},nodes:[{node_row:0,is_root:true,status:'completed',weighted_peak_coverage:0.25,condition_label:'A'},{node_row:1,is_root:false,status:'completed',weighted_peak_coverage:0.45,parent_weighted_peak_coverage:0.25,weighted_delta_peak_coverage:0.2,improved_sample_count:2,newly_assigned_peak_count_total:3,mean_delta_peak_coverage:0.15,condition_label:'A + B'}],conditions:[]});
assert.equal(el('rpaCompoundNodes').querySelectorAll('tbody tr').length,2);assert.equal(el('rpaCombinationNode').options.length,2);
const peakRows=[{sample_id:7,peak_index:0,dataset_id:1,record_index:3,sample_index:0,mz:31,intensity:1,relative_intensity:1,'node::0':'missing','node::1':'new'},{sample_id:7,peak_index:1,dataset_id:1,record_index:3,sample_index:0,mz:47,intensity:1,relative_intensity:1,'node::0':'assigned','node::1':'assigned'},{sample_id:8,peak_index:0,dataset_id:1,record_index:4,sample_index:1,mz:47,intensity:1,relative_intensity:1,'node::0':'assigned','node::1':null}];
const peakNodes=[{node_row:0,label:'A',column:'node::0',pathLabel:'A',isRoot:true},{node_row:1,label:'B',column:'node::1',pathLabel:'A → B',isRoot:false}];
result('compoundPeaks',{total:3,offset:0,rows:peakRows,nodes:peakNodes});
const cells=row=>[...el('rpaPeakTable').querySelectorAll('tbody tr')[row].querySelectorAll('td')].slice(-2).map(cell=>cell.textContent);
assert.deepEqual(cells(0),['✗','NEW']);assert.deepEqual(cells(1),['✓','✓']);assert.deepEqual(cells(2),['✓','—']);
assert([...el('rpaPeakNodeState').options].some(option=>option.value==='1:new'));
messages.length=0;el('rpaPeakTable').querySelector('[data-rpa-peak-sort="mz"]').click();assert.equal(queried()[0].sort,'mz');assert.equal(queried()[0].descending,false);result('compoundPeaks',{total:3,offset:0,rows:peakRows,nodes:peakNodes});
messages.length=0;el('rpaPeakTable').querySelector('[data-rpa-peak-sort="mz"]').click();assert.equal(queried()[0].descending,true);result('compoundPeaks',{total:3,offset:0,rows:[...peakRows].reverse(),nodes:peakNodes});assert(el('rpaPeakTable').querySelector('tbody tr').textContent.includes('47.000000'));
messages.length=0;el('rpaPeakTable').querySelector('[data-rpa-peak-sort="mz"]').click();assert.equal(queried()[0].sort,null);result('compoundPeaks',{total:3,offset:0,rows:peakRows,nodes:peakNodes});assert(el('rpaPeakTable').querySelector('tbody tr').textContent.includes('31.000000'));
const peakAdduct=el('rpaPeakFilters').querySelector('[data-rpa-pfilter="adduct"]');peakAdduct.value='Na';messages.length=0;el('rpaPeakApply').click();assert.equal(peakAdduct.value,'Na');assert.equal(queried()[0].filters.adduct,'Na');el('rpaPeakReset').click();assert.equal(peakAdduct.value,'');
messages.length=0;el('rpaPeakTable').querySelector('[data-rpa-detail="1"]').click();
assert.deepEqual(queried().map(value=>[value.action,value.nodeRow,value.sampleId,value.peakIndex]),[['peakDetail',1,7,0]]);
messages.length=0;el('rpaCompoundNodes').querySelector('[data-rpa-cn="1"]').click();assert.equal(queried()[0].action,'compoundNodeDistribution');assert.equal(queried()[0].nodeRow,1);
const compoundStats=metric=>[0,1].map(flag=>({metric,without_precursor:flag,n:4,mean:0.15,median:0.1,std:0.05,q25:0.05,q75:0.2,min:0,max:0.3,histogram:{edges:[0,0.5,1],counts:[3,1]}}));
result('compoundNodeDistribution',{compoundId:'cabc',nodeRow:1,default:'mean',distributions:[...compoundStats('delta_peak_coverage'),...compoundStats('delta_intensity_coverage')],samples:[{sample_index:0,delta_peak_coverage:0.1,newly_assigned_peak_count:1}]});
assert.equal(el('rpaDistributionDetail').hidden,false);assert.equal(el('rpaDistributionDetailMetric').value,'delta_peak_coverage');
assert.equal(el('rpaDistributionDetail').querySelector('tbody b').textContent,'15.00%');  // mean shown first, emphasized
for(const label of ['Mean','Median','Std','Q25','Q75','Min','Max','Sample-level values'])assert(el('rpaDistributionDetail').textContent.includes(label),label);
assert(el('rpaDistributionDetail').querySelector('svg rect.rpa-bar'));
messages.length=0;el('rpaLoadCompoundCombinations').click();assert.deepEqual([queried()[0].action,queried()[0].nodeRow],['compoundReactionCombinations',0]);
// Drag & drop of a result on the configuration button or on the result section opens it.
const dropOn=async(target,transfer)=>{const before=messages.length;const event=new window.Event('drop',{bubbles:true,cancelable:true});Object.defineProperty(event,'dataTransfer',{value:{types:[],files:[],getData:()=>'',...transfer}});target.dispatchEvent(event);await new Promise(resolve=>setTimeout(resolve,0));return messages.length>before?messages.at(-1):null;};
const explorerResult={types:['text/uri-list'],getData:type=>type==='text/uri-list'?'file://'+resultFile.split(path.sep).map(encodeURIComponent).join('/').replace(/^\/?/,'/'):''};
assert.deepEqual(await dropOn(el('rpaLoadConfig'),explorerResult),{type:'reactionPatternAnalysis/openResult',path:resultFile});
assert.deepEqual(await dropOn(el('rpaResultDrop'),explorerResult),{type:'reactionPatternAnalysis/openResult',path:resultFile});
// A result dropped from the OS has no path: it must not be uploaded without its compounds/ and details/ siblings.
window.uploadDataset=async()=>{throw new Error('result must not be uploaded');};
assert.equal(await dropOn(el('rpaResultDrop'),{files:[{name:'result.reactionpatternanalysis.pft',size:10}]}),null);
assert.match(el('rpaResultInfo').textContent,/must be dropped from VS Code Explorer.*compounds\/ and details\//);
assert.equal(await dropOn(el('rpaLoadConfig'),{files:[{name:'result.reactionpatternanalysis.pft',size:10,text:async()=>'{}'}]}),null);
assert.match(el('rpaRunStatus').textContent,/must be dropped from VS Code Explorer/);
window.close();
fs.rmSync(resultRoot,{recursive:true,force:true});
// Native libraries (e.g. UCX via torch) can print warnings to stdout; the result
// reader must skip those lines instead of discarding the backend.
const noisyExtension=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-noisy-backend-'));
fs.mkdirSync(path.join(noisyExtension,'src/features/reaction-pattern-analysis'),{recursive:true});
fs.writeFileSync(path.join(noisyExtension,'src/features/reaction-pattern-analysis/backend.py'),['import json, sys','print("[0.0] UCX  ERROR inotify_add_watch(/tmp) failed: No space left on device", flush=True)','request = json.loads(sys.stdin.readline())','print(json.dumps({"requestId": request["requestId"], "ok": True, "result": {"isResult": True}}), flush=True)'].join('\n'));
// A non-standard name forces isResultFile() to ask the backend (action: detect).
const pftHeader=path.join(noisyExtension,'renamed-result.h5');fs.writeFileSync(pftHeader,Buffer.from([0x89,0x48,0x44,0x46,0x0d,0x0a,0x1a,0x0a]));
assert.equal(await editor.isResultFile({extensionPath:noisyExtension},noisyExtension,pftHeader),true);
fs.rmSync(noisyExtension,{recursive:true,force:true});
console.log('reaction pattern analysis UI checks passed');
})().catch(error=>{console.error(error);process.exit(1);});
