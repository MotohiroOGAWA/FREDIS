const assert=require('assert/strict'),fs=require('fs'),path=require('path'),os=require('os'),Module=require('module');
const dropOn=async(w,target,transfer)=>{w.validateDroppedPaths=paths=>paths;const event=new w.Event('drop',{bubbles:true,cancelable:true});Object.defineProperty(event,'dataTransfer',{value:{types:[],files:[],getData:()=>'',...transfer}});target.dispatchEvent(event);await new Promise(resolve=>setTimeout(resolve,0));};
const {JSDOM,VirtualConsole}=require('jsdom');
const project=path.resolve(__dirname,'../..');let selected;
const mock={workspace:{workspaceFolders:[{uri:{fsPath:project}}],getConfiguration:()=>({get:(key,fallback)=>fallback})},window:{showOpenDialog:async()=>selected}};
const load=Module._load;Module._load=function(name,parent,main){return name==='vscode'?mock:load.call(this,name,parent,main);};
const periodicRows=require('../src/features/reaction-pattern-set/periodic-table').rows;assert(periodicRows.every(row=>row.length===18));assert.equal(periodicRows[1][12],'B');assert.equal(periodicRows[1][17],'Ne');assert.equal(periodicRows[2][12],'Al');assert.equal(periodicRows[2][17],'Ar');
const extension=require('../src/extension'),preset=require('./fixtures').model;
const messages=[],errors=[],console=new VirtualConsole();console.on('jsdomError',error=>errors.push(error.detail?.stack||error.message));
const dom=new JSDOM(extension.workbenchHtml(require('./fixtures').dataConfig,{modelConfig:preset}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:console,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){w.lastScrollTarget=this;};w.ResizeObserver=class{observe(){}disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:s=>s};}});
(async()=>{
 const w=dom.window,d=w.document,post=data=>w.dispatchEvent(new w.MessageEvent('message',{data}));
 try{
  assert.deepEqual(errors,[]);
  const nav=d.querySelector('#navigationRail [data-page=reactionPatterns]');assert(nav);
  assert.equal(nav.closest('details').querySelector('summary span').textContent,'Data');
  nav.click();assert(!d.getElementById('reactionPatternApp').hidden);assert(d.getElementById('form').hidden);
  assert(!d.getElementById('reactionPatternEmpty').hidden);
  d.querySelector('#navigationRail [data-page=data]').click();const reactionPatternDetails=[...d.querySelectorAll('#dataParameterEditor [data-model-block=fragmenter_params]>details')].find(item=>item.querySelector('summary')?.textContent.startsWith('Reaction Pattern Set ·'));assert(reactionPatternDetails);reactionPatternDetails.open=true;reactionPatternDetails.dispatchEvent(new w.Event('toggle'));
  assert(!d.getElementById('reactionPatternApp').hidden);assert(!d.getElementById('form').hidden);assert(d.querySelector('#navigationRail [data-page=data]').classList.contains('active'));
  const original=preset.fragmenter_params.fragment_ion_tree_builder.reaction_pattern_set.patterns.length;
  assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,original);
  d.getElementById('addReactionPattern').click();const index=original;assert.equal(d.activeElement.dataset.pattern,String(index));assert.equal(w.lastScrollTarget,d.activeElement.closest('.pattern-card'));
  function input(selector,value){const element=d.querySelector(selector);assert(element,selector);element.value=value;element.dispatchEvent(new w.Event('input',{bubbles:true}));}
  input('[data-pattern="'+index+'"][data-key=name]','added_carbon_oxygen');
  input('[data-pattern="'+index+'"][data-key=reactant_smarts]','[#6:1]-[#8:2]');
  d.querySelector('[data-add-product="'+index+'"]').click();assert.equal(d.activeElement.dataset.product,'0');assert.equal(w.lastScrollTarget,d.activeElement.closest('.product-row'));
  input('[data-pattern="'+index+'"][data-product="0"][data-key=name]','carbon');
  input('[data-pattern="'+index+'"][data-product="0"][data-key=smarts]','[#6:1]');
  d.getElementById('saveReactionPatterns').click();assert.equal(messages.at(-1).type,'saveReactionPatternSet');assert.equal(messages.at(-1).value.reaction_pattern_set.name,d.getElementById('reactionPatternSetName').value);assert.equal(w.getComputedStyle(d.querySelector('.reaction-pattern-set-actions')).flexWrap,'wrap');assert(['0','0px'].includes(w.getComputedStyle(d.getElementById('loadReactionPatterns')).minWidth));
  assert.equal(messages.at(-1).value.reaction_pattern_set.patterns[index].name,'added_carbon_oxygen');
  const config=w.eval('getConfig()'),added=config.fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set.patterns[index];
  assert.equal(config.fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set.patterns.length,original+1);
  assert.deepEqual(JSON.parse(JSON.stringify(added)),{name:'added_carbon_oxygen',reactant_smarts:'[#6:1]-[#8:2]',products:[{name:'carbon',smarts:'[#6:1]'}]});
  const args=extension.buildArgs(config);assert.deepEqual(JSON.parse(args[args.indexOf('--params-json')+1]).fragment_ion_tree_builder.reaction_pattern_set.patterns[index],JSON.parse(JSON.stringify(added)));
  assert.equal(preset.fragmenter_params.fragment_ion_tree_builder.reaction_pattern_set.patterns.length,original);
  d.getElementById('addVisualPattern').click();assert(!d.getElementById('visualBuilder').hidden);assert.equal(d.activeElement.id,'builderSmiles');assert.equal(d.getElementById('builderSourceType').value,'smiles');
  assert(d.getElementById('applyParameterPatterns').hidden);
  const graph={atoms:[{index:0,x:0,y:0,symbol:'C'},{index:1,x:1,y:1,symbol:'O'}],bonds:[{index:0,begin:0,end:1,order:1}]};
  const drawing=d.getElementById('drawMolecule').onclick(),request=messages.at(-1);
  post({type:'chemistryResult',requestId:request.requestId,result:graph});await drawing;
  w.eval('productGraph=JSON.parse(JSON.stringify(molGraph));renderProduct()');
  for(const id of ['moleculeCanvas','productCanvas']){
   const canvas=d.getElementById(id),svg=canvas.querySelector('svg'),lasso=svg.querySelector('.selection-lasso');
   svg.createSVGPoint=()=>({x:0,y:0,matrixTransform(){return {x:this.x,y:this.y};}});
   svg.getScreenCTM=()=>({inverse:()=>({})});svg.setPointerCapture=()=>{};
   assert(lasso.hasAttribute('hidden'));assert.equal(w.getComputedStyle(lasso).display,'none');
   canvas.onpointerdown({button:0,clientX:10,clientY:20,pointerId:1,target:svg,preventDefault(){}});
   canvas.onpointermove({clientX:30,clientY:40});
   assert(!lasso.hasAttribute('hidden'));assert.equal(w.getComputedStyle(lasso).display,'block');
   assert.equal(lasso.getAttribute('points'),'10,20 30,40');
   canvas.onpointermove({clientX:50,clientY:60});assert.equal(lasso.getAttribute('points'),'10,20 30,40 50,60');
   canvas.onpointercancel();assert(canvas.querySelector('.selection-lasso').hasAttribute('hidden'));
  }
  // Both canvases share toggle/deselect, shape selection, pan and zoom handling.
  function pointer(canvas,kind,extra={}){
   const svg=canvas.querySelector('svg');
   svg.createSVGPoint=()=>({x:0,y:0,matrixTransform(){return {x:this.x,y:this.y};}});
   svg.getScreenCTM=()=>({inverse:()=>({})});svg.setPointerCapture=()=>{};
   canvas['onpointer'+kind]({button:0,clientX:10,clientY:20,pointerId:1,target:svg,preventDefault(){},...extra});
  }
  for(const id of ['moleculeCanvas','productCanvas']){
   const canvas=d.getElementById(id),step=canvas.closest('.ce-step');
   function clickAtom(shift=false){pointer(canvas,'down',{target:canvas.querySelector('[data-mol-atom="0"]'),shiftKey:shift});pointer(canvas,'up');}
   clickAtom();assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   clickAtom();assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   clickAtom();clickAtom(true);assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   const box=step.querySelector('[data-shape="box"]');box.click();assert.equal(box.getAttribute('aria-pressed'),'true');
   pointer(canvas,'down');pointer(canvas,'move',{clientX:100,clientY:100});assert.equal(canvas.querySelector('.selection-lasso').getAttribute('points'),'10,20 100,20 100,100 10,100 10,20');canvas.onpointercancel();
   pointer(canvas,'down',{clientX:0,clientY:0});pointer(canvas,'move',{clientX:700,clientY:440});pointer(canvas,'up');assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   const bondSelector=id==='productCanvas'?'[data-product-bond="base:0"]':'[data-mol-bond="0"]';assert(canvas.querySelector(bondSelector).classList.contains('selected'));
   pointer(canvas,'down',{clientX:300,clientY:170,shiftKey:true});pointer(canvas,'move',{clientX:335,clientY:235});pointer(canvas,'up');assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));assert(canvas.querySelector('[data-mol-atom="1"]').classList.contains('selected'));assert(!canvas.querySelector(bondSelector).classList.contains('selected'));
   pointer(canvas,'down',{clientX:0,clientY:0});pointer(canvas,'move',{clientX:700,clientY:440});pointer(canvas,'up');
   step.querySelector('[data-remove-atom="0"]').click();assert(!canvas.querySelector(bondSelector).classList.contains('selected'));assert(canvas.querySelector('[data-mol-atom="1"]').classList.contains('selected'));
   pointer(canvas,'down',{clientX:0,clientY:0});pointer(canvas,'move',{clientX:700,clientY:440});pointer(canvas,'up');
   pointer(canvas,'down',{clientX:0,clientY:0,shiftKey:true});pointer(canvas,'move',{clientX:700,clientY:440});pointer(canvas,'up');assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   const before=canvas.querySelector('svg').getAttribute('viewBox');
   pointer(canvas,'down',{button:2,target:canvas.querySelector('[data-mol-atom="0"]')});pointer(canvas,'move',{clientX:50,clientY:60});pointer(canvas,'up');
   assert.notEqual(canvas.querySelector('svg').getAttribute('viewBox'),before);assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   step.querySelector('[data-zoom="fit"]').click();step.querySelector('[data-shape="lasso"]').click();
   clickAtom();step.querySelector('[data-mode="edit"]').click();
   pointer(canvas,'down',{target:canvas.querySelector('[data-mol-atom="0"]')});pointer(canvas,'up');
   assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   pointer(canvas,'down',{target:canvas.querySelector('[data-mol-atom="1"]')});pointer(canvas,'up');
   assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));assert(!canvas.querySelector('[data-mol-atom="1"]').classList.contains('selected'));
   assert.equal(canvas.nextElementSibling.textContent,'1 atoms · 0 bonds'+(id==='productCanvas'?' · 1 molecules':''));
   step.querySelector('[data-mode="selection"]').click();clickAtom();assert(step.querySelector('.ce-help').textContent.includes('Shift'));

  }
  const sourceCanvas=d.getElementById('moleculeCanvas');pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-atom="0"]')});pointer(sourceCanvas,'up');
  const transforms=[...sourceCanvas.querySelectorAll('.mol-atom')].map(a=>a.getAttribute('transform').match(/[-\d.]+/g).map(Number));assert.equal(transforms[1][0]-transforms[0][0],65);assert.equal(transforms[1][1]-transforms[0][1],65);
  const periodic=d.querySelector('[data-periodic]');assert(periodic);const dialog=d.querySelector('.ce-dialog');dialog.showModal=()=>dialog.setAttribute('open','');dialog.close=()=>dialog.removeAttribute('open');periodic.click();assert(dialog.hasAttribute('open'));
  const element=symbol=>[...dialog.querySelectorAll('.ce-periodic button')].find(b=>b.textContent===symbol);
  assert.equal(element('C').style.gridColumn,'14');assert.equal(element('C').style.gridRow,'2');assert.equal(element('Ne').style.gridColumn,'18');assert.equal(element('Na').style.gridRow,'3');assert.equal(element('Na').style.gridColumn,'1');assert.equal(element('C').getAttribute('aria-pressed'),'true');assert.equal(w.getComputedStyle(element('C')).backgroundColor,'rgb(8, 125, 221)');
  element('N').click();assert.equal(element('N').getAttribute('aria-pressed'),'true');assert.equal(w.getComputedStyle(element('N')).backgroundColor,'rgb(8, 125, 221)');
  dialog.querySelector('.actions button').click();d.querySelector('[data-periodic]').click();assert.equal(element('N').getAttribute('aria-pressed'),'true');element('N').click();assert.equal(element('N').getAttribute('aria-pressed'),'false');dialog.close();
  element('N').click();element('O').click();const exclude=d.querySelector('#atomConstraints [data-exclude-elements]');assert(exclude);exclude.click();assert.equal(w.eval('atomConstraintState[0].excludeElements'),true);assert(d.querySelector('#atomConstraints').textContent.includes('Excluded elements'));assert(element('C').classList.contains('active'));assert(element('N').classList.contains('active'));assert(element('O').classList.contains('active'));dialog.close();d.querySelector('[data-periodic]').click();assert(element('C').classList.contains('active'));assert(element('N').classList.contains('active'));assert(element('O').classList.contains('active'));dialog.close();d.querySelector('#atomConstraints [data-exclude-elements]').click();assert.equal(w.eval('atomConstraintState[0].excludeElements'),false);
  const any=d.querySelector('#atomConstraints [data-am="any"]');any.checked=true;any.dispatchEvent(new w.Event('change',{bubbles:true}));assert(d.querySelector('#atomConstraints [data-non-h]'));assert(d.querySelector('#atomConstraints [data-non-h]').checked);
  const nonH=d.querySelector('#atomConstraints [data-non-h]');nonH.checked=false;nonH.dispatchEvent(new w.Event('change',{bubbles:true}));assert.equal(w.eval('atomConstraintState[0].nonHydrogen'),false);assert.equal(w.eval('atomConstraintState[0].mode'),'any');
  assert(d.querySelector('#atomConstraints [data-h-count="any"]').checked);assert(!d.querySelector('#atomConstraints [data-h-count="0"]').checked);
  for(const count of [1,2]){const checkbox=d.querySelector('#atomConstraints [data-h-count="'+count+'"]');checkbox.checked=true;checkbox.dispatchEvent(new w.Event('change',{bubbles:true}));}
  assert.equal(w.eval('JSON.stringify(atomConstraintState[0].hydrogenCounts)'),JSON.stringify([1,2]));assert(!d.querySelector('#atomConstraints [data-h-count="any"]').checked);assert(d.querySelector('#atomConstraints [data-h-count="1"]').checked);assert(d.querySelector('#atomConstraints [data-h-count="2"]').checked);
  for(const count of [1,2]){const checkbox=d.querySelector('#atomConstraints [data-h-count="'+count+'"]');checkbox.checked=false;checkbox.dispatchEvent(new w.Event('change',{bubbles:true}));}
  assert.equal(w.eval('JSON.stringify(atomConstraintState[0].hydrogenCounts)'),JSON.stringify([]));assert(d.querySelector('#atomConstraints [data-h-count="any"]').checked);

  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="0"]')});pointer(sourceCanvas,'up');
  d.querySelector('#atomConstraints').closest('.ce-step').querySelector('[data-mode="edit"]').click();
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="0"]')});pointer(sourceCanvas,'up');
  d.querySelector('#bondConstraints [data-rs="inRing"]').click();
  assert.equal(w.eval('bondConstraintState[0].ringStatus'),'inRing');assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);
  w.eval("chemistry=async(command,payload)=>command==='reactant'?{smarts:'[#6:1]-[#8:2]',atomMapBySource:{0:1,1:2}}:command==='molecule'?"+JSON.stringify(graph)+":command==='productFromSelection'?{smarts:'[#6:1].[#8:2]'}:command==='productState'?{keptAtoms:[0,1],keptBonds:[],addedBonds:[],atomOverrides:{},bondQueries:{},bondOverrides:{}}:{valid:true}");
  const patternCount=d.querySelectorAll('#reactionPatterns .pattern-card').length;await d.getElementById('generateReactant').onclick();assert(!d.querySelector('.ce-step .ce-body').hidden);assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount);assert.equal(d.querySelector('[aria-label="Reactant SMARTS"]').value,'[#6:1]-[#8:2]');
  await d.getElementById('applyReactant').onclick();
  assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount);
  const steps=d.querySelectorAll('.ce-step');assert(steps[0].querySelector('.ce-body').hidden);assert(!steps[1].querySelector('.ce-body').hidden);assert.equal(steps[0].querySelector('.ce-status').textContent,'Completed');
  pointer(sourceCanvas,'down',{button:2});pointer(sourceCanvas,'move',{clientX:50,clientY:60});pointer(sourceCanvas,'up');assert.equal(steps[0].querySelector('.ce-status').textContent,'Completed');
  steps[0].querySelector('.ce-header button').click();assert(!steps[0].querySelector('.ce-body').hidden);assert(steps[1].querySelector('.ce-body').hidden);
  steps[1].querySelector('.ce-header button').click();assert(steps[0].querySelector('.ce-body').hidden);assert(!steps[1].querySelector('.ce-body').hidden);
  steps[0].querySelector('.ce-header [aria-label="Copy Reactant SMARTS"]').click();assert.equal(messages.at(-1).type,'copyText');assert.equal(messages.at(-1).text,'[#6:1]-[#8:2]');
  await d.getElementById('newVisualProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,1);assert(d.querySelector('[data-visual-product="0"]').textContent.endsWith(' *'));
  const pc=d.getElementById('productCanvas');assert(pc.querySelector('[data-mol-atom="0"]').classList.contains('selected'));assert(pc.querySelector('[data-product-bond="base:0"]').classList.contains('selected'));
  assert(!steps[1].querySelector('#builderProductSource'));assert(!steps[1].querySelector('#builderProductSourceType'));assert(!steps[1].querySelector('#drawProduct'));assert(!steps[1].querySelector('[data-tool="deleteAtom"]'));assert(!steps[1].querySelector('[data-tool="cut"]'));
  pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');assert(!pc.querySelector('[data-product-bond="base:0"]').classList.contains('selected'));
  assert.equal(pc.nextElementSibling.textContent,'2 atoms · 0 bonds · 2 molecules');
  await d.getElementById('generateProduct').onclick();assert(d.querySelector('[aria-label="Product SMARTS"]').value.includes('.'));assert(!steps[1].querySelector('.ce-body').hidden);
  assert(!steps[1].querySelector('[data-undo-product]'));pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');assert(pc.querySelector('[data-product-bond="base:0"]').classList.contains('selected'));
  steps[1].querySelector('[data-mode="edit"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');assert(steps[1].querySelector('[data-product-order="preserve"]').checked);assert.equal(w.eval('Object.keys(productBondOverrides).length'),0);
  const double=steps[1].querySelector('[data-product-order="2"]');assert(double);double.click();assert(pc.querySelector('[data-product-bond="base:0"]').classList.contains('bond-double'));assert.equal(w.eval('productBondOverrides[0]'),'2');
  steps[1].querySelector('[data-product-order="preserve"]').click();assert(pc.querySelector('[data-product-bond="base:0"]').classList.contains('bond-single'));assert.equal(w.eval('Object.keys(productBondOverrides).length'),0);
  assert(!steps[1].querySelector('.ce-reference'));assert(!steps[1].querySelector('[data-tool="change"]'));assert(!steps[1].querySelector('[aria-label="New bond type"]'));
  assert.equal(d.getElementById('applyProduct').textContent,'Save');assert(!steps[1].querySelector('.ce-tabs'));assert(![...steps[1].querySelectorAll('button')].some(b=>b.textContent==='Reaction SMARTS'));
  const productField=steps[1].querySelector('[aria-label="Product SMARTS"]'),reactionField=steps[1].querySelector('[aria-label="Reaction SMIRKS"]');assert.equal(productField.closest('.ce-output').nextElementSibling,reactionField.closest('.ce-output'));
  steps[1].querySelector('[aria-label="Copy Product SMARTS"]').click();assert(steps[1].querySelector('[aria-label="Copy Product SMARTS"]').classList.contains('ce-clicked'));assert.equal(messages.at(-1).text,productField.value);steps[1].querySelector('[aria-label="Copy Reaction SMIRKS"]').click();assert.equal(messages.at(-1).text,reactionField.value);
  await d.getElementById('applyProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,1);assert(!d.querySelector('[aria-label="Product 1 SMARTS"]'));assert(!d.querySelector('[data-visual-product="0"]').textContent.endsWith(' *'));await d.getElementById('applyProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,1);assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount);
  steps[1].querySelector('[data-mode="selection"]').click();pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="0"]')});pointer(pc,'up');assert(d.querySelector('[data-visual-product="0"]').textContent.endsWith(' *'));d.getElementById('restoreSavedProduct').click();assert(!d.querySelector('[data-visual-product="0"]').textContent.endsWith(' *'));
  await d.getElementById('newVisualProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,2);assert(d.querySelector('[data-visual-product="1"]').textContent.endsWith(' *'));await d.getElementById('applyProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,2);assert(d.querySelector('[data-visual-product="1"]').classList.contains('active'));
  const nameInput=d.getElementById('builderProductName');nameInput.value='renamed draft';nameInput.dispatchEvent(new w.Event('input',{bubbles:true}));assert(d.querySelector('[data-visual-product="1"]').textContent.endsWith(' *'));await d.getElementById('registerReactionPattern').onclick();assert(steps[2].querySelector('.ce-error').textContent.includes('Save modified products'));assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount);
  await d.querySelector('[data-visual-product="0"]').onclick();await d.querySelector('[data-visual-product="1"]').onclick();assert.equal(nameInput.value,'renamed draft');assert(d.querySelector('[data-visual-product="1"]').textContent.endsWith(' *'));await d.getElementById('applyProduct').onclick();assert.equal(d.querySelector('[data-visual-product="1"]').textContent,'renamed draft');assert.equal(d.querySelectorAll('[data-visual-product]').length,2);
  // Restore covers name, inclusion, bond edits and the saved output, without saving other Products.
  const savedSecond={name:nameInput.value,atoms:w.eval('JSON.stringify([...productSelectedAtoms])'),overrides:w.eval('JSON.stringify(productBondOverrides)'),smarts:productField.value};
  nameInput.value='discard me';nameInput.dispatchEvent(new w.Event('input',{bubbles:true}));steps[1].querySelector('[data-mode="edit"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');steps[1].querySelector('[data-product-order="2"]').click();assert.equal(w.eval('productBondOverrides[0]'),'2');steps[1].querySelector('[data-mode="selection"]').click();pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="0"]')});pointer(pc,'up');assert(!pc.querySelector('[data-product-bond="base:0"]').classList.contains('selected'));assert(!d.getElementById('restoreSavedProduct').disabled);d.getElementById('restoreSavedProduct').click();assert.equal(nameInput.value,savedSecond.name);assert.equal(w.eval('JSON.stringify([...productSelectedAtoms])'),savedSecond.atoms);assert.equal(w.eval('JSON.stringify(productBondOverrides)'),savedSecond.overrides);assert.equal(productField.value,savedSecond.smarts);assert(!d.querySelector('[data-visual-product="1"]').textContent.endsWith(' *'));assert(d.getElementById('restoreSavedProduct').disabled);
  // Keep the old Product active until async chemistry completes, and ignore stale loads.
  w.eval('window.fastProductChemistry=chemistry;window.pendingProductLoads=[];chemistry=(command,payload)=>command==="molecule"?new Promise(resolve=>window.pendingProductLoads.push(()=>resolve(window.fastProductChemistry(command,payload)))):window.fastProductChemistry(command,payload)');
  const loadingButton=d.querySelector('[data-visual-product="0"]');const oldLoad=loadingButton.onclick();assert.equal(loadingButton.getAttribute('aria-busy'),'true');assert.equal(w.eval('visualProductIndex'),1);assert.equal(nameInput.value,savedSecond.name);
  const newestLoad=d.querySelector('[data-visual-product="1"]').onclick();assert.equal(w.pendingProductLoads.length,2);w.pendingProductLoads[1]();await newestLoad;w.pendingProductLoads[0]();await oldLoad;assert(!loadingButton.hasAttribute('aria-busy'));assert.equal(w.eval('visualProductIndex'),1);assert.equal(nameInput.value,savedSecond.name);w.eval('chemistry=window.fastProductChemistry');
  await d.querySelector('[data-visual-product="0"]').onclick();assert.equal(nameInput.value,'product_1');assert.equal(d.querySelector('[data-visual-product="0"]').textContent,'product_1');await d.querySelector('[data-visual-product="1"]').onclick();assert.equal(nameInput.value,savedSecond.name);assert.equal(w.eval('JSON.stringify([...productSelectedAtoms])'),savedSecond.atoms);
  await d.querySelector('[data-visual-product="0"]').onclick();assert(d.querySelector('[data-visual-product="0"]').classList.contains('active'));assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount);
  await d.querySelector('[data-remove-product-draft="1"]').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,1);assert(d.querySelector('[data-visual-product="0"]').classList.contains('active'));
  assert.equal(steps.length,3);assert.equal(d.getElementById('generateProduct').parentElement,d.getElementById('applyProduct').parentElement);assert(d.getElementById('applyProduct').classList.contains('ce-apply'));assert.equal(d.getElementById('registerReactionPattern').closest('.ce-step'),steps[2]);d.getElementById('reviewReactionPattern').click();assert(!steps[2].querySelector('.ce-body').hidden);assert(steps[1].querySelector('.ce-body').hidden);
  await d.getElementById('registerReactionPattern').onclick();assert(d.getElementById('visualBuilder').hidden);assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount+1);assert.equal(w.eval('reactionPatternModel.reaction_pattern_set.patterns[visualPatternIndex].products.length'),1);
  await d.querySelector('[data-remove-product-draft="0"]').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,0);assert.equal(w.eval('reactionPatternModel.reaction_pattern_set.patterns[visualPatternIndex].products.length'),1);
  await d.getElementById('registerReactionPattern').onclick();assert(steps[2].querySelector('.ce-error').textContent.includes('Add at least one product'));assert.equal(w.eval('reactionPatternModel.reaction_pattern_set.patterns[visualPatternIndex].products.length'),1);
  await d.getElementById('newVisualProduct').onclick();await d.getElementById('applyProduct').onclick();await d.getElementById('registerReactionPattern').onclick();assert(d.getElementById('visualBuilder').hidden);assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount+1);
  await w.eval('editVisualPattern(visualPatternIndex)');await d.querySelector('[data-remove-product-draft="0"]').onclick();assert.equal(w.eval('reactionPatternModel.reaction_pattern_set.patterns[visualPatternIndex].products.length'),1);await d.getElementById('newVisualProduct').onclick();await d.getElementById('applyProduct').onclick();
  const registeredSnapshot=w.eval('JSON.stringify(reactionPatternModel.reaction_pattern_set.patterns)');
  w.eval('window.validChemistry=chemistry;chemistry=async(command,payload)=>{if(command==="validate")throw new Error("Rejected draft");return window.validChemistry(command,payload)}');await d.getElementById('registerReactionPattern').onclick();assert.equal(w.eval('JSON.stringify(reactionPatternModel.reaction_pattern_set.patterns)'),registeredSnapshot);assert.equal(steps[2].querySelector('.ce-error').textContent,'Rejected draft');assert(!d.getElementById('visualBuilder').hidden);w.eval('chemistry=window.validChemistry');
  // Added bonds have independent inclusion and can be edited/removed without atom deletion.
  w.eval('productGraph={atoms:[{index:0,x:0,y:0,symbol:"C"},{index:1,x:1,y:0,symbol:"C"},{index:2,x:2,y:1,symbol:"O"}],bonds:[{index:0,begin:0,end:1,order:1},{index:1,begin:1,end:2,order:1}]};productAtomMaps={0:1,1:2,2:3};productAtomOverrides={};productBondOverrides={};productAddedBonds=[];productSelectedAtoms=new Set([0,1,2]);nextProductBondId=1;renderProduct()');
  steps[1].querySelector('[data-mode="selection"]').click();
  for(const bond of pc.querySelectorAll('[data-product-bond].selected')){pointer(pc,'down',{target:bond,shiftKey:true});pointer(pc,'up');}
  pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="1"]'),shiftKey:true});pointer(pc,'up');const productOnly=steps[1].querySelector('[data-product-view="product"]'),reactantContext=steps[1].querySelector('[data-product-view="context"]');await productOnly.parentElement.onclick({target:productOnly});assert.equal(productOnly.getAttribute('aria-pressed'),'true');assert(!pc.querySelector('[data-mol-atom="1"]'));assert(pc.querySelector('[data-mol-atom="0"]'));assert(pc.querySelector('[data-mol-atom="2"]'));assert.equal(pc.querySelectorAll('[data-product-bond]').length,0);
  await reactantContext.parentElement.onclick({target:reactantContext});assert.equal(reactantContext.getAttribute('aria-pressed'),'true');assert(pc.querySelector('[data-mol-atom="1"]'));assert(!pc.querySelector('[data-mol-atom="1"]').classList.contains('selected'));assert.equal(pc.querySelectorAll('[data-product-bond]').length,2);pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="1"]')});pointer(pc,'up');
  steps[1].querySelector('[data-tool="add"]').click();
  pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="0"]')});pointer(pc,'up');assert(pc.querySelector('[data-mol-atom="0"]').classList.contains('bond-start'));
  pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="2"]')});pointer(pc,'up');
  assert(pc.querySelector('[data-product-bond="added:1"]').classList.contains('selected'));assert.equal(pc.nextElementSibling.textContent,'3 atoms · 1 bonds · 2 molecules');assert(!steps[1].querySelector('[data-product-order="preserve"]'));assert(steps[1].querySelector('[data-product-order="1"]').checked);steps[1].querySelector('[data-product-order="2"]').click();assert(pc.querySelector('[data-product-bond="added:1"]').classList.contains('bond-double'));assert.equal(w.eval('productAddedBonds[0].order'),'2');await productOnly.parentElement.onclick({target:productOnly});assert.equal(pc.querySelectorAll('[data-product-bond]').length,1);assert(pc.querySelector('[data-product-bond="added:1"]'));assert(!pc.querySelector('[data-product-bond="base:0"]'));steps[1].querySelector('[data-product-order="3"]').click();assert(pc.querySelector('[data-product-bond="added:1"]').classList.contains('bond-triple'));assert.equal(w.eval('productAddedBonds[0].order'),'3');await reactantContext.parentElement.onclick({target:reactantContext});
  steps[1].querySelector('[data-mode="selection"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="added:1"]')});pointer(pc,'up');assert(!pc.querySelector('[data-product-bond="added:1"]').classList.contains('selected'));assert.equal(w.eval('productAddedBonds.length'),1);assert.equal(pc.nextElementSibling.textContent,'3 atoms · 0 bonds · 3 molecules');
  pointer(pc,'down',{target:pc.querySelector('[data-product-bond="added:1"]')});pointer(pc,'up');steps[1].querySelector('[data-mode="edit"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="added:1"]')});pointer(pc,'up');
  steps[1].querySelector('[data-mode="selection"]').click();pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="0"]'),shiftKey:true});pointer(pc,'up');assert(!pc.querySelector('[data-product-bond="added:1"]').classList.contains('selected'));assert(pc.querySelector('[data-mol-atom="2"]').classList.contains('selected'));assert.equal(w.eval('productAddedBonds.length'),1);pointer(pc,'down',{target:pc.querySelector('[data-product-bond="added:1"]')});pointer(pc,'up');steps[1].querySelector('[data-mode="edit"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="added:1"]')});pointer(pc,'up');
  assert(!steps[1].querySelector('[data-product-order="preserve"]'));assert(steps[1].querySelector('[data-product-order="3"]').checked);
  steps[1].querySelector('[data-remove-added-bond]').click();assert(!pc.querySelector('[data-product-bond="added:1"]'));assert.equal(w.eval('productAddedBonds.length'),0);
  assert(!steps[1].querySelector('[data-undo-product]'));
  // Regression with the user's exact SMILES and the real RDKit messaging backend.
  const {execFileSync}=require('child_process');
  w.testChemistry=(command,payload)=>JSON.parse(execFileSync('python',[path.join(project,'fredis_workbench/src/features/reaction-pattern-set/backend.py')],{cwd:project,input:JSON.stringify({command,payload}),encoding:'utf8'})).result;
  w.eval('chemistry=async(command,payload)=>window.testChemistry(command,payload)');
  d.getElementById('addVisualPattern').click();d.getElementById('builderSourceType').value='smarts';d.getElementById('builderSmiles').value='[#6:1]-[#8:2]';d.getElementById('builderPatternName').value='SMARTS all selected';await d.getElementById('drawMolecule').onclick();
  const smartsGraph=w.eval('molGraph');assert.equal(w.eval('selectedAtoms.size'),smartsGraph.atoms.length);assert.equal(w.eval('selectedBonds.size'),smartsGraph.bonds.length);assert.equal(sourceCanvas.querySelectorAll('.mol-atom.selected').length,smartsGraph.atoms.length);assert.equal(sourceCanvas.querySelectorAll('.bond-group.selected').length,smartsGraph.bonds.length);
  await d.getElementById('applyReactant').onclick();assert.equal(steps[0].querySelector('.ce-error').textContent,'');assert.equal(d.querySelector('[aria-label="Reactant SMARTS"]').value,'[#6:1]-[#8:2]');
  d.getElementById('addVisualPattern').click();d.getElementById('builderSmiles').value='C=CC=CC=CC=C';await d.getElementById('drawMolecule').onclick();w.eval('selectedAtoms=new Set(molGraph.atoms.map(a=>a.index));selectedBonds=new Set(molGraph.bonds.map(b=>b.index));renderMolecule()');await d.getElementById('generateReactant').onclick();
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-atom="0"]')});pointer(sourceCanvas,'up');await d.getElementById('generateReactant').onclick();const renumbered=d.querySelector('[aria-label="Reactant SMARTS"]').value,renumberedMaps=[...renumbered.matchAll(/:(\d+)\]/g)].map(match=>Number(match[1]));assert.deepEqual(renumberedMaps,[1,2,3,4,5,6,7]);assert.deepEqual(Object.values(w.eval('atomMapBySource')).sort((a,b)=>a-b),[1,2,3,4,5,6,7]);assert([...sourceCanvas.querySelectorAll('.mol-atom.selected .atom-index')].some(label=>label.textContent==='1'));
  await d.getElementById('applyReactant').onclick();assert(!steps[1].querySelector('.ce-body').hidden);assert.deepEqual(Object.values(w.eval('productAtomMaps')).sort((a,b)=>a-b),[1,2,3,4,5,6,7]);
  d.getElementById('addVisualPattern').click();assert.equal(d.getElementById('builderSourceType').value,'smiles');
  d.getElementById('builderSmiles').value='CC=CC1CCCCC1';await d.getElementById('drawMolecule').onclick();assert.equal(w.eval('selectedAtoms.size'),0);assert.equal(w.eval('selectedBonds.size'),0);
  const actual=w.eval('molGraph'),ringBond=actual.bonds.find(b=>b.inRing),chainBond=actual.bonds.find(b=>!b.inRing);
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+chainBond.index+'"]')});pointer(sourceCanvas,'up');assert(d.querySelector('#bondConstraints [data-rs="notInRing"]').checked);
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+chainBond.index+'"]'),shiftKey:true});pointer(sourceCanvas,'up');
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+ringBond.index+'"]')});pointer(sourceCanvas,'up');assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);
  for(const id of [ringBond.begin,ringBond.end]){pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-atom="'+id+'"]')});pointer(sourceCanvas,'up');}
  steps[0].querySelector('[data-mode="edit"]').click();pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+ringBond.index+'"]')});pointer(sourceCanvas,'up');
  d.querySelector('#bondConstraints [data-rs="notInRing"]').click();assert(d.querySelector('#bondConstraints [data-rs="notInRing"]').checked);
  d.querySelector('#bondConstraints [data-rs="inRing"]').click();assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);
  await d.getElementById('generateReactant').onclick();assert(!steps[0].querySelector('.ce-body').hidden);assert(steps[1].querySelector('.ce-body').hidden);assert(d.querySelector('[aria-label="Reactant SMARTS"]').value.includes('@'));assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount+1);
  await d.getElementById('applyReactant').onclick();assert(!steps[1].querySelector('.ce-body').hidden);assert.equal(w.eval('Object.keys(productAtomOverrides).length'),0);assert.equal(w.eval('Object.keys(productBondOverrides).length'),0);
  await d.getElementById('newVisualProduct').onclick();await d.getElementById('generateProduct').onclick();const inherited=d.querySelector('[aria-label="Product SMARTS"]').value;assert(inherited.includes('@'));assert(!inherited.includes('.'));
  steps[1].querySelector('[data-mode="selection"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');assert(!pc.querySelector('[data-product-bond="base:0"]').classList.contains('selected'));await d.getElementById('generateProduct').onclick();assert.equal(steps[1].querySelector('.ce-error').textContent,'');assert(d.querySelector('[aria-label="Product SMARTS"]').value.includes('.'),d.querySelector('[aria-label="Product SMARTS"]').value);
  assert.equal(pc.nextElementSibling.textContent,'2 atoms · 0 bonds · 2 molecules');await d.getElementById('applyProduct').onclick();assert(!d.querySelector('[data-visual-product="0"]').closest('.ce-product-item').querySelector('.ce-counts'));assert.equal(pc.nextElementSibling.textContent,'2 atoms · 0 bonds · 2 molecules');assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount+1);await d.getElementById('registerReactionPattern').onclick();assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,patternCount+2);
  const exclusionPattern={name:'Excluded C N O',reactant_smarts:'[!#6;!#7;!#8;!#1:5]',products:[]};post({type:'reactionPatternLoaded',pattern:exclusionPattern,index:-1});await w.eval('editVisualPattern(reactionPatternModel.reaction_pattern_set.patterns.length-1)');assert(d.querySelector('#atomConstraints [data-exclude-elements]').checked);assert.equal(w.eval('JSON.stringify(atomConstraintState[0].elements)'),JSON.stringify(['C','N','O']));assert(d.querySelector('#atomConstraints [data-non-h]').checked);assert(d.querySelector('#atomConstraints').textContent.includes('Excluded elements'));await d.getElementById('generateReactant').onclick();assert(d.querySelector('[aria-label="Reactant SMARTS"]').value.includes('!#6'));w.eval('reactionPatternModel.reaction_pattern_set.patterns.pop();renderReactionPatterns()');
  const hydrogenPattern={name:'McLafferty H',reactant_smarts:'[#6;H1,H2,H3,H4:5]',products:[]};post({type:'reactionPatternLoaded',pattern:hydrogenPattern,index:-1});await w.eval('editVisualPattern(reactionPatternModel.reaction_pattern_set.patterns.length-1)');for(const count of [1,2,3,4])assert(d.querySelector('#atomConstraints [data-h-count="'+count+'"]').checked);assert(!d.querySelector('#atomConstraints [data-h-count="any"]').checked);await d.getElementById('generateReactant').onclick();assert.equal(d.querySelector('[aria-label="Reactant SMARTS"]').value,'[#6;H1,H2,H3,H4:1]');w.eval('reactionPatternModel.reaction_pattern_set.patterns.pop();renderReactionPatterns()');
  const restoredPattern={name:'Restored constraints',reactant_smarts:'[!#1:3]-[#6&!#1&!#1&!#1;H1,H2,H3,H4:4]',products:[]};post({type:'reactionPatternLoaded',pattern:restoredPattern,index:-1});await w.eval('editVisualPattern(reactionPatternModel.reaction_pattern_set.patterns.length-1)');assert(d.querySelector('#bondConstraints [data-rs="any"]').checked);assert(!d.querySelector('#bondConstraints [data-rs="notInRing"]').checked);assert.equal(w.eval('atomConstraintState[1].nonHydrogen'),true);assert.equal(w.eval('atomConstraintState[1].mode'),'elements');await d.getElementById('generateReactant').onclick();const restored=d.querySelector('[aria-label="Reactant SMARTS"]').value;assert(!restored.includes('@'));assert.equal((restored.match(/!#1/g)||[]).length,2,restored);w.eval('reactionPatternModel.reaction_pattern_set.patterns.pop();renderReactionPatterns()');
  const anyBondPattern={name:'Any bond',reactant_smarts:'[#6:1]~[#8:2]',products:[]};post({type:'reactionPatternLoaded',pattern:anyBondPattern,index:-1});await w.eval('editVisualPattern(reactionPatternModel.reaction_pattern_set.patterns.length-1)');assert(d.querySelector('#bondConstraints [data-bm="any"]').checked);await d.getElementById('generateReactant').onclick();assert(d.querySelector('[aria-label="Reactant SMARTS"]').value.includes('~'));w.eval('reactionPatternModel.reaction_pattern_set.patterns.pop();renderReactionPatterns()');
  const ringTopologyPattern={name:'Ring topology without query',reactant_smarts:'[#6:1]1-[#6:2]-[#6:3]-1',products:[]};post({type:'reactionPatternLoaded',pattern:ringTopologyPattern,index:-1});await w.eval('editVisualPattern(reactionPatternModel.reaction_pattern_set.patterns.length-1)');assert(d.querySelector('#bondConstraints [data-rs="any"]').checked);await d.getElementById('generateReactant').onclick();assert(!d.querySelector('[aria-label="Reactant SMARTS"]').value.includes('@'));w.eval('reactionPatternModel.reaction_pattern_set.patterns.pop();renderReactionPatterns()');
  const ringOrPattern={name:'Ring bond alternatives',reactant_smarts:'[#6:1]1-,=;@[#6:2]-[#6:3]-1',products:[]};post({type:'reactionPatternLoaded',pattern:ringOrPattern,index:-1});await w.eval('editVisualPattern(reactionPatternModel.reaction_pattern_set.patterns.length-1)');assert(d.querySelector('#bondConstraints [data-bt="single"]').checked);assert(d.querySelector('#bondConstraints [data-bt="double"]').checked);assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);await d.getElementById('generateReactant').onclick();const ringOr=d.querySelector('[aria-label="Reactant SMARTS"]').value;assert(ringOr.includes('-,='),ringOr);assert(ringOr.includes('@'),ringOr);w.eval('reactionPatternModel.reaction_pattern_set.patterns.pop();renderReactionPatterns()');
  await dropOn(w,d.getElementById('loadReactionPatterns'),{getData:type=>type==='text/uri-list'?'file:///tmp/imported%20set.reactionset.pft':''});
  assert.equal(messages.at(-1).type,'loadReactionPatternSet');assert.equal(messages.at(-1).path,'/tmp/imported set.reactionset.pft');
  const pattern={name:'imported',reactant_smarts:'[#6:1]-[#8:2]',products:[{name:'carbon',smarts:'[#6:1]'}]};
  await dropOn(w,d.getElementById('loadSinglePattern'),{files:[{name:'pattern.json',text:async()=>JSON.stringify(pattern)}]});
  assert.equal(messages.at(-1).type,'loadReactionPattern');assert.deepEqual(JSON.parse(messages.at(-1).json),pattern);
  post({type:'reactionPatternLoaded',pattern,index:-1});assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,original+4);
  post({type:'reactionPatternError',message:'Invalid JSON'});assert.equal(d.getElementById('reactionPatternErrorMessage').textContent,'Invalid JSON');
  post({type:'reactionPatternLoaded',pattern:JSON.parse(JSON.stringify(pattern)),index:-1});assert.equal(d.getElementById('reactionPatternErrorMessage').textContent,'');assert.deepEqual(errors,[]);
  // The editor stays attached to the selected pattern across rerenders and renumbering.
  const last=w.eval('reactionPatternModel.reaction_pattern_set.patterns.length-1');
  await w.eval('editVisualPattern('+last+')');
  const builder=d.getElementById('visualBuilder');
  assert.equal(builder.parentElement,d.querySelectorAll('#reactionPatterns > .pattern-card')[last]);
  assert.equal(builder.querySelector('h2').textContent,'Reaction Pattern Editor (Pattern '+(last+1)+')');
  assert(builder.contains(w.lastScrollTarget));
  d.getElementById('builderPatternName').value='Unregistered draft';
  d.querySelector('[data-remove-pattern="0"]').click();
  assert.equal(d.getElementById('visualBuilder'),builder);
  assert.equal(builder.parentElement,d.querySelectorAll('#reactionPatterns > .pattern-card')[last-1]);
  assert.equal(builder.querySelector('h2').textContent,'Reaction Pattern Editor (Pattern '+last+')');
  assert.equal(d.getElementById('builderPatternName').value,'Unregistered draft');
  d.getElementById('closeBuilder').click();assert(builder.hidden);assert.equal(d.activeElement.dataset.pattern,String(last-1));
  await w.eval('editVisualPattern(0)');assert.equal(builder.parentElement,d.querySelector('#reactionPatterns > .pattern-card'));
  d.querySelector('[data-remove-pattern="0"]').click();assert(builder.hidden);assert(builder.isConnected);
  d.getElementById('addVisualPattern').click();assert.equal(builder.parentElement.id,'reactionPatternApp');assert.equal(builder.querySelector('h2').textContent,'Reaction Pattern Editor');
  assert.equal(w.lastScrollTarget,builder);assert.equal(d.activeElement.id,'builderSmiles');assert.deepEqual(errors,[]);
  const setName=d.getElementById('reactionPatternSetName').value;d.getElementById('builderPatternName').value='Discard this draft';
  const clearPatterns=d.getElementById('clearReactionPatterns');assert(!clearPatterns.disabled);clearPatterns.click();
  assert.equal(d.querySelectorAll('#reactionPatterns .pattern-card').length,0);assert(!d.getElementById('reactionPatternEmpty').hidden);assert(clearPatterns.disabled);assert(builder.hidden);assert.equal(d.getElementById('builderPatternName').value,'');assert.equal(d.getElementById('reactionPatternSetName').value,setName);
  assert.equal(w.eval('getConfig().fragmenterParams.fragment_ion_tree_builder.reaction_pattern_set.patterns.length'),0);assert.deepEqual(errors,[]);

 }finally{dom.window.close();}
 const filename=require.resolve('../src/extension'),compiled=new Module(filename,module);compiled.filename=filename;compiled.paths=module.paths;
 compiled._compile(fs.readFileSync(filename,'utf8')+'\nmodule.exports.readReactionPatternImport=readReactionPatternImport;module.exports.reactionPatternSetSavePath=reactionPatternSetSavePath;',filename);
 const savePath=compiled.exports.reactionPatternSetSavePath;
 assert.equal(savePath('my_set'),'my_set.reactionset.pft');
 assert.equal(savePath('新しいセット','/tmp/old.reactionset.pft'),'/tmp/新しいセット.reactionset.pft');
 assert.equal(extension.reactionPatternSetSavePath('renamed_set','/tmp/old_name.reactionset.pft'),'/tmp/renamed_set.reactionset.pft');
 assert.equal(savePath('a/b:c'),'a_b_c.reactionset.pft');
 assert.equal(savePath('   '),'patterns.reactionset.pft');
 // A name that already carries the current suffix round-trips as-is.
 assert.equal(savePath('named.reactionset.pft'),'named.reactionset.pft');
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-pattern-import-'));
 try{
  const file=path.join(root,'set.reactionset.pft'),value={reaction_pattern_set:{name:'imported_set',patterns:[]}};
  fs.writeFileSync(file,JSON.stringify(value));const context={extensionPath:path.dirname(filename)};
  assert.deepEqual(await compiled.exports.readReactionPatternImport(context,{path:file}),{value,path:file});
  assert.deepEqual(await compiled.exports.readReactionPatternImport(context,{json:JSON.stringify(value)}),{value,path:''});
  selected=[{fsPath:file}];assert.deepEqual(await compiled.exports.readReactionPatternImport(context,{}),{value,path:file});
  await assert.rejects(()=>compiled.exports.readReactionPatternImport(context,{json:'invalid JSON'}),SyntaxError);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
 process.stdout.write('Workbench reaction-pattern navigation, addition, preparation apply, visible source/product drag trails, shared selection, Shift box removal, pan isolation, accordion transitions, Copy, periodic table, product operations and import checks passed.\n');
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
