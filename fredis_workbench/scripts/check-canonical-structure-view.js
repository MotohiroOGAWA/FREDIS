const assert = require('assert/strict');
const {JSDOM} = require('jsdom');
const view = require('../src/features/fragment-tree-result/view');
const dom = new JSDOM('<button id="resultTheme"></button><button id="resultDocs"></button><button id="resultGithub"></button><button id="resultHelp"></button><div id="resultHelpText" hidden></div><div id="detail"></div>', {runScripts:'outside-only'});
const w = dom.window;
w.HTMLElement.prototype.scrollIntoView = function() {};
Object.defineProperty(w.SVGElement.prototype, 'viewBox', {get(){return {baseVal:{width:620,height:360}};}});
w.eval("window.vscode={getState:()=>({}),setState(){},postMessage(){}};window.detail=document.getElementById('detail');window.esc=value=>String(value??'').replace(/&/g,'&amp;').replace(/</g,'&lt;');");
w.eval(view.script());
const nodes = [
  {id:5,smiles:'CCO',precursor:true,reactionSets:[[],[0]],positiveBranchReactions:[0]},
  {id:8,smiles:'C',precursor:false,reactionSets:[[1]],positiveBranchReactions:[]}
];
const edges = [
  {id:0,source:5,target:8,reactionIndex:0,label:1,labelName:'positive',transitions:[]},
  {id:1,source:5,target:8,reactionIndex:1,label:0,labelName:'weak_negative',transitions:[]},
  {id:2,source:5,target:5,reactionIndex:2,label:-1,labelName:'ignored',transitions:[]}
];
const sample = (id,group,ce,intensity)=>({id,branchGroup:group,collisionEnergy:ce,adduct:group?'[M+Na]+':'[M+H]+',mainAdduct:group?'[M+Na]+':'[M+H]+',nodes:nodes.map(n=>({...n,observed:n.id===8,observedIntensity:n.id===8?intensity:0})),edges,nodeCount:2,edgeCount:3,edgeTransitionCount:0,nodeIdSource:'CompoundBatch row ID',peaks:[{index:0,mz:15,intensity,precursor:false,matches:[]}]});
w.render({file:'canonical.preft.pt',summary:{samples:3,nodes:2,edges:3,edgeTransitions:0,primitiveReactions:3,labelledEdges:2,positiveBranches:1,branchGroups:2,maxHistoryReactionCount:1},sourceSmiles:'CCO',sourceSvg:'<svg viewBox="0 0 620 360"></svg>',reactions:[],branchGroups:[{id:0,mainAdduct:'[M+H]+'},{id:1,mainAdduct:'[M+Na]+'}],samples:[sample(0,0,10,.2),sample(1,0,40,.9),sample(2,1,20,.5)]});
const el=id=>w.document.getElementById(id);
assert.equal(el('resultSample').value,'0');
assert.deepEqual([...el('resultSample').options].filter(o=>!o.hidden).map(o=>o.value),['0','1']);
assert.equal(el('sampleTree').querySelectorAll('path.tree-edge').length,3);
assert.equal(el('sampleTree').querySelectorAll('.branch-label-1').length,1);
const curves=[...el('sampleTree').querySelectorAll('path.tree-edge')].map(n=>n.getAttribute('d'));
assert.notEqual(curves[0],curves[1],'parallel reactions must be independently selectable');
assert(curves[2].includes(' C '),'identity self loops must remain visible');
el('resultSample').value='1';el('resultSample').onchange();
assert(el('resultNodeRows').textContent.includes('0.9000'));
el('resultBranchGroup').value='1';el('resultBranchGroup').onchange();
assert.equal(el('resultSample').value,'2');
assert.deepEqual([...el('resultSample').options].filter(o=>!o.hidden).map(o=>o.value),['2']);
assert(el('resultNodeRows').textContent.includes('0.5000'));
el('sampleTree').querySelector('[data-edge="1"] .edge-hit').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
assert(el('edgeDetail').textContent.includes('weak_negative'));
setTimeout(()=>w.close(),0);
console.log('Canonical structure view: branch selection, CE intensity, labels, parallel edges and self loops passed.');
