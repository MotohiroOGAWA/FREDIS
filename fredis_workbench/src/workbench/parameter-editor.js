// A DOM-only configuration editor shared by data preparation and training.
const molEncoderFields=require('./mol-encoder-fields');
const molDescriptors=require('./mol-descriptors');
function createParameterEditor(container, initial, kind = "training", editReactionPatternSet, editAdductRuleSet) {
  const clone=value=>JSON.parse(JSON.stringify(value));
  const dataValue=value=>({fragmenter_params:clone(value.fragmenter_params||value),symbols:clone(value.symbols||['C','N','O','P','S','F','Cl','Br','I']),max_unique_fragment_smiles:value.max_unique_fragment_smiles??-1,max_reaction_combinations:value.max_reaction_combinations??-1});
  const trainingValue=value=>{const source=value?.params||value||{},encoderDefaults={...Object.fromEntries(molEncoderFields.map(field=>[field[5],field[3]])),descriptors:['FormalCharge']},result={mol_encoder_params:{...encoderDefaults,...(source.mol_encoder_params||{})}};for(const key of ['reaction_model_params','spectrum_model_params','generation_params','fine_tuning'])if(source[key]&&typeof source[key]==='object'&&!Array.isArray(source[key]))result[key]=clone(source[key]);result.generation_params??={};return result;};
  // 'fragmenter' edits a standalone fragmenter config file: the same Fragmenter
  // Parameters block as data preparation, without its symbols or search limits.
  const fragmenterValue=value=>({fragmenter_params:clone(value?.fragmenter_params||value||{})});
  const valueFor=value=>kind==='data'?dataValue(value):kind==='fragmenter'?fragmenterValue(value):trainingValue(value);
  const fragmenterKind=kind==='data'||kind==='fragmenter';
  let model=valueFor(initial);
  const descriptions={
    architecture:'Spectrum generator architecture. Source-anchored reaction models operate on primitive reactions.',
    max_reaction_count:'Maximum number of primitive reactions in a normalized fragment reaction state.',
    precursor_candidate_max_reaction_count:'Maximum reaction count used to find candidates for the observed precursor.',
    mass_tolerance:'Peak assignment mass tolerance, e.g. 0.01Da,10ppm.',
    reaction_pattern_set:'Patterns defining eligible primitive reactions from the source molecule.',
    reactant_smarts:'SMARTS query that matches a reaction site in the source molecule.',
    smarts:'Product SMARTS describing retained atoms and bond changes.',
    adduct_type:'Observed precursor adduct associated with this fragment ion rule.',
    hydrogen_shift_min:'Inclusive minimum net hydrogen shift shared by the rule set.',
    hydrogen_shift_max:'Inclusive maximum net hydrogen shift shared by the rule set.',
    fragment_ion_adducts:'Fragment ion adducts allowed for this precursor adduct, e.g. [M+H]+.',
    symbols:'Elements included in prepared atom and formula features. Select elements using the periodic table.',
    max_unique_fragment_smiles:'Skip a source molecule as soon as its generated fragments exceed this many distinct canonical SMILES. The source molecule itself is not counted. -1 means unlimited.',
    max_reaction_combinations:'Skip a source molecule as soon as its reaction search examines more than this many raw reaction combinations (every size from 1 to max_reaction_count), before RDKit runs them. -1 means unlimited.',
    hidden_dim:'Hidden representation dimension for this model component.',
    branch_main_adduct_dim:'Main-adduct feature width used only by the Branch Scorer.',
    ...Object.fromEntries(molEncoderFields.map(field=>[field[5],field[6]])),
    descriptors:'Molecular descriptors calculated from each molecule and concatenated with the Mol Encoder graph embedding.',
    num_layers:'Number of transformer layers.',num_heads:'Number of attention heads. Must divide the corresponding hidden dimension.',
    max_degree:'Maximum degree represented in graph structural embeddings.',max_spatial_dist:'Maximum atom graph distance encoded by the molecular encoder.',max_edge_dist:'Maximum bond distance encoded by the molecular encoder.',
    reaction_neighborhood_mode:'How the Primitive Reaction Encoder folds in atoms outside the reacting site: hop_pooling pools hop-distance neighbors into the reaction; none ignores them entirely.',
    reaction_neighborhood_max_hop:'Number of hop distances pooled by hop_pooling (1..3). Disabled when Neighborhood Mode is none. Changes the number of per-hop projection layers; a resumed run always keeps the checkpoint\'s own value.',
    state_num_layers:'Number of Set Transformer layers encoding the current reaction set.',state_dropout:'Dropout in the reaction state Set Transformer.',
    branch_path_threshold:'Minimum cumulative path probability retained by fragment tree search. Zero disables threshold pruning. Saved with the model; prediction can override it.',
    max_fragment_nodes:'Maximum fragment nodes shared by the complete (compound, main adduct) branch group. Saved with the model; prediction can override it.',
    ion_prediction_threshold:'Minimum ion confidence (sigmoid probability) kept when generating a spectrum. Below this, a candidate adduct/hydrogen-shift is suppressed instead of appearing as a small peak. Does not affect training loss.',
    peak_intensity_threshold:'Minimum max-normalized intensity retained in generated spectra. Does not affect training loss.',
    ion_embedding_dim:'Embedding width for fragment-ion-adduct identity.',hydrogen_shift_embedding_dim:'Embedding width for the net hydrogen shift.',state_hidden_dim:'Hidden width of the ion-state explanation encoder.',main_adduct_dim:'Main-adduct feature width passed separately to the intensity head.',collision_energy_dim:'Collision-energy feature width passed separately to the intensity head.',
    generation_params:'Spectrum generation settings, separate from the model architecture. They are saved with the model and used by validation and prediction; prediction can override them.',
    name:'Human-readable name stored in this configuration.',
    freeze_mol_encoder:'Freeze the molecular encoder during training.',
  };
  const templates={patterns:{name:'new_pattern',reactant_smarts:'[!#1:1]-[!#1:2]',products:[{name:'product',smarts:'[!#1:1]'}]},products:{name:'product',smarts:'[!#1:1]'},adduct_rules:{name:'new_rule',adduct_type:'[M+H]+',fragment_ion_adducts:['[M+H]+']},fragment_ion_adducts:'[M+H]+',symbols:'C'};
  const optional={reaction_model_params:{branch_main_adduct_dim:128,num_heads:4,state_num_layers:2,reaction_neighborhood_mode:'hop_pooling',reaction_neighborhood_max_hop:3},mol_encoder_params:{dropout:0,descriptors:['FormalCharge']},spectrum_model_params:{ion_embedding_dim:32,hydrogen_shift_embedding_dim:16,state_hidden_dim:128,main_adduct_dim:128,collision_energy_dim:16},generation_params:{branch_path_threshold:0,max_fragment_nodes:100,ion_prediction_threshold:0.5,peak_intensity_threshold:0}};
  // Purely cosmetic sub-headings for the sections with the most fields; any
  // field not listed here (e.g. a newer option) still renders, just ungrouped.
  const fieldGroups={
    mol_encoder_params:{'Encoder Parameters':molEncoderFields.map(field=>field[5]),'Molecular Descriptors':['descriptors']},
    reaction_model_params:{
      'Architecture':['hidden_dim','branch_main_adduct_dim','num_heads','reaction_neighborhood_mode','reaction_neighborhood_max_hop'],
      'Branch Scorer · Architecture':['state_num_layers'],
    },
    spectrum_model_params:{
      'Fragment Transformer · Architecture':['hidden_dim','num_layers','num_heads'],
      'Ion-State Encoder · Architecture':['ion_embedding_dim','hydrogen_shift_embedding_dim','state_hidden_dim'],
      'Intensity Head · Architecture':['main_adduct_dim','collision_energy_dim'],
    },
    generation_params:{
      'Fragment Tree Search':['branch_path_threshold','max_fragment_nodes'],
      'Spectrum Output':['ion_prediction_threshold','peak_intensity_threshold'],
    },
  };
  // A resumed/saved model_config always has these (training_model_config writes
  // them from the single shared --dropout), but they must never be editable
  // here independently of it -- only one dropout value can ever be specified.
  const linkedToSharedDropout=new Set(['reaction_model_params.state_dropout','spectrum_model_params.dropout']);
  // max_roles is sized automatically and Mol Encoder symbols are owned by the
  // prepared dataset. A loaded config may carry both, but neither is edited
  // in the training architecture form.
  const derivedFromDataset=new Set(['reaction_model_params.max_roles','mol_encoder_params.symbols']);
  // Closed-choice architecture settings are selects, not free-form text inputs.
  const choices={reaction_neighborhood_mode:['hop_pooling','none']};
  const basic=new Set(['fragmenter_params.fragment_ion_tree_builder.max_reaction_count','fragmenter_params.precursor_candidate_max_reaction_count','fragmenter_params.mass_tolerance']);
  const badge=full=>full.endsWith('.max_reaction_count')?'Dataset inherited':full.startsWith('generation_params.')?'Generation':/(hidden_dim|embedding_dim|main_adduct_dim|collision_energy_dim|branch_main_adduct_dim|num_layers|num_heads|state_num_layers|equivalent_state_aggregation|reaction_neighborhood_mode|reaction_neighborhood_max_hop)$/.test(full)?'Architecture':'';
  const node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const button=(text,fn)=>{const b=node('button',text);b.type='button';b.onclick=fn;return b;};
  const title=key=>String(key).replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase()).replace('Smarts','SMARTS');
  function help(key,path){return descriptions[key]||'Configure '+path.replaceAll('_',' ').replaceAll('.',' / ')+'. This value is passed to the FREDIS Python workflow.';}
  function renderValue(parent,key,host,path,excludeBasic=false){
    const full=path?path+'.'+key:String(key),value=parent[key];if(excludeBasic&&basic.has(full))return;if(linkedToSharedDropout.has(full)||derivedFromDataset.has(full))return;
    if(key==='reaction_pattern_set'&&editReactionPatternSet){const details=node('details'),summary=node('summary','Reaction Pattern Set · '+(value.name||'(unnamed)')),mount=node('div');mount.className='embedded-set-editor';details.append(summary,mount);details.ontoggle=()=>{if(details.open)editReactionPatternSet(clone(parent[key]),updated=>{parent[key]=clone(updated);summary.textContent='Reaction Pattern Set · '+(updated.name||'(unnamed)');},mount);};host.append(details);return;}
    if(key==='fragment_ion_adduct_rule_set'&&editAdductRuleSet){const mainAdducts=set=>[...new Set((set.adduct_rules||[]).map(rule=>rule.adduct_type).filter(Boolean))].join(', ')||'no main adducts',details=node('details'),summary=node('summary','Fragment Ion Adduct Rule Set · '+(value.name||'(unnamed)')+' · '+mainAdducts(value)),mount=node('div');mount.className='embedded-set-editor';details.append(summary,mount);details.ontoggle=()=>{if(details.open)editAdductRuleSet(clone(parent[key]),updated=>{parent[key]=clone(updated);summary.textContent='Fragment Ion Adduct Rule Set · '+(updated.name||'(unnamed)')+' · '+mainAdducts(updated);},mount);};host.append(details);return;}
    if(full==='mol_encoder_params.descriptors'){
      const group=node('fieldset'),legend=node('legend','Descriptors');legend.dataset.help=help(key,full);group.append(legend);
      for(const name of molDescriptors){const wrapper=node('label'),input=node('input');wrapper.className='check';input.type='checkbox';input.dataset.parameterPath=full+'.'+name;input.checked=value.includes(name);
        input.onchange=()=>{parent[key]=molDescriptors.filter(item=>item===name?input.checked:parent[key].includes(item));};wrapper.append(input,node('span',name));group.append(wrapper);}
      host.append(group);return;
    }
    if(Array.isArray(value)||value&&typeof value==='object'){
      const group=node('fieldset'),legend=node('legend',title(key));legend.dataset.help=help(key,full);group.append(legend);
      if(Array.isArray(value)){
        value.forEach((item,index)=>{const row=node('div');row.className='parameter-array-item';renderValue(value,index,row,full);row.append(button('Remove',()=>{value.splice(index,1);render();}));group.append(row);});
        group.append(button('Add '+title(key),()=>{value.push(clone(templates[key]??value[0]??''));render();}));
      }else{
        // Every optional field is filled in with its default up front instead
        // of behind a one-by-one "Override" button; "Use Default" (below) still
        // resets a single field.
        for(const [field,defaultValue]of Object.entries(optional[key]||{}))if(!(field in value))value[field]=defaultValue;
        const groups=fieldGroups[key];
        if(groups){
          const grouped=new Set();
          for(const [label,fields]of Object.entries(groups)){
            const present=fields.filter(field=>field in value);
            if(!present.length)continue;
            const heading=node('div',label);heading.className='parameter-group-heading';group.append(heading);
            for(const field of present){grouped.add(field);renderValue(value,field,group,full,excludeBasic);}
          }
          for(const field of Object.keys(value))if(!grouped.has(field))renderValue(value,field,group,full,excludeBasic);
        }else{
          Object.keys(value).forEach(child=>renderValue(value,child,group,full,excludeBasic));
        }
      }
      host.append(group);return;
    }
    const fieldKey=typeof key==='number'?path.split('.').at(-1):key;
    const wrapper=node('label'),label=node('span',typeof key==='number'?title(fieldKey)+' '+(key+1):title(key));label.dataset.help=help(fieldKey,full);const mark=badge(full);if(mark){const tag=node('small',mark);tag.className='parameter-badge';label.append(' ',tag);}const options=choices[fieldKey];if(options&&!options.includes(value))parent[key]=options[0];const input=node(options?'select':'input');input.dataset.parameterPath=full;
    // Neighborhood Max Hop is meaningless once Neighborhood Mode is 'none';
    // re-render on mode change so its disabled state stays in sync.
    if(options){for(const choice of options){const option=node('option',title(choice));option.value=choice;input.append(option);}input.value=parent[key];input.onchange=()=>{parent[key]=input.value;if(fieldKey==='reaction_neighborhood_mode')render();};}else input.type=typeof value==='boolean'?'checkbox':typeof value==='number'?'number':'text';
    if(options){wrapper.append(label,input);if(optional[path]?.[key]!==undefined)wrapper.append(button('Use Default',()=>{delete parent[key];render();}));host.append(wrapper);return;}
    if(fieldKey==='reaction_neighborhood_max_hop'&&parent.reaction_neighborhood_mode==='none')input.disabled=true;
    if(input.type==='number')input.step='any';if(input.type==='checkbox'){input.checked=value;wrapper.className='check';}else input.value=value??'';
    input.oninput=()=>{if(input.type==='number'&&(!input.value.trim()||!Number.isFinite(Number(input.value)))){input.setCustomValidity('Enter a finite number.');return;}input.setCustomValidity('');parent[key]=input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value):input.value;};
    wrapper.append(label,input);if(optional[path]?.[key]!==undefined)wrapper.append(button('Use Default',()=>{delete parent[key];render();}));host.append(wrapper);
  }
  function renderSymbols(section){
    const rows=['H . . . . . . . . . . . . . . . . He','Li Be . . . . . . . . . . B C N O F Ne','Na Mg . . . . . . . . . . Al Si P S Cl Ar','K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr','Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe','Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn','Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og','. . Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu . .','. . Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr . .'];
    const list=node('p','Selected: '+model.symbols.join(', '));section.append(list);const grid=node('div');grid.className='periodic-table';grid.setAttribute('aria-label','Select element symbols');
    for(const row of rows)for(const symbol of row.split(' ')){if(symbol==='.'){grid.append(node('span'));continue;}const selected=model.symbols.includes(symbol);const tile=button(symbol,()=>{model.symbols=selected?model.symbols.filter(item=>item!==symbol):[...model.symbols,symbol];render();if(container.dispatchEvent)container.dispatchEvent(new Event('input',{bubbles:true}));});tile.dataset.symbol=symbol;tile.setAttribute('aria-pressed',String(selected));tile.title=symbol+(selected?' — enabled':' — disabled');grid.append(tile);}section.append(grid);
  }
  function render(){
    const expanded=new Set([...container.querySelectorAll('[data-advanced]')].filter(n=>!n.hidden).map(n=>n.dataset.advanced));for(const [app,mount]of [['reactionPatternApp','reactionPatternPageMount'],['adductApp','adductPageMount']]){const element=document.getElementById?.(app),page=document.getElementById?.(mount);if(element&&page&&container.contains?.(element)){element.hidden=true;page.append(element);}}container.replaceChildren();
    for(const [key,value]of Object.entries(model)){
      if(kind==='data'&&key==='symbols'){const section=node('section');section.append(node('h2','Symbols'));renderSymbols(section);container.append(section);continue;}
      if(kind==='training'&&['architecture','fine_tuning','fragmenter_params','adduct_type_strs','symbols','max_unique_fragment_smiles','max_reaction_combinations'].includes(key))continue;
      const section=node('section'),heading=node('h2',key==='fragmenter_params'?'Fragmenter Parameters':key==='mol_encoder_params'?'Molecular Architecture':title(key));section.dataset.modelBlock=key;section.append(heading);
      if(kind==='data'&&['max_unique_fragment_smiles','max_reaction_combinations'].includes(key))continue;
      if(kind==='data'&&key==='fragmenter_params'){renderValue(model,'max_unique_fragment_smiles',section,'');renderValue(model,'max_reaction_combinations',section,'');}
      if(key==='fragmenter_params')for(const path of basic){const parts=path.split('.');let parent=model;for(const part of parts.slice(0,-1))parent=parent?.[part];if(parent&&parts.at(-1) in parent)renderValue(parent,parts.at(-1),section,parts.slice(0,-1).join('.'));}
      const advanced=node('div');advanced.dataset.advanced=key;advanced.hidden=!expanded.has(key);
      if(fragmenterKind&&key==='fragmenter_params'){const builder=value.fragment_ion_tree_builder||{};for(const child of Object.keys(builder))if(child!=='max_reaction_count')renderValue(builder,child,section,'fragmenter_params.fragment_ion_tree_builder',true);for(const child of Object.keys(value))if(!['fragment_ion_tree_builder','precursor_candidate_max_reaction_count','mass_tolerance'].includes(child))renderValue(value,child,section,'fragmenter_params',true);}
      else if(['mol_encoder_params','reaction_model_params','spectrum_model_params'].includes(key)||(kind==='data'&&key!=='architecture')){const label=key==='mol_encoder_params'?'Molecular Architecture':title(key),toggle=button('Advanced '+label,()=>{advanced.hidden=!advanced.hidden;toggle.setAttribute('aria-expanded',String(!advanced.hidden));});toggle.dataset.parameterToggle=key;toggle.setAttribute('aria-expanded',String(!advanced.hidden));section.append(toggle);renderValue(model,key,advanced,'',true);section.append(advanced);}
      else renderValue(model,key,section,'');container.append(section);
    }
  }
  render();return {getValue:()=>clone(model),setValue:value=>{model=valueFor(value);render();}};
}
function client(){
  const el=id=>document.getElementById(id),form=el('form'),training=el('trainingForm');
  const dataInitial={fragmenter_params:initial.fragmenterParams,symbols:initial.symbols,max_unique_fragment_smiles:initial.maxUniqueFragmentSmiles,max_reaction_combinations:initial.maxReactionCombinations};
  function editPatterns(target){return (value,apply,mount)=>{reactionPatternModel={reaction_pattern_set:value};reactionPatternPath='';mount.append(el('reactionPatternApp'));el('reactionPatternApp').hidden=false;el('applyParameterPatterns').hidden=true;el('reactionPatternPath').textContent='Embedded in '+(target==='data'?'Data Preparation':'Training');onReactionPatternChange=()=>{apply(reactionPatternModel.reaction_pattern_set);form.dispatchEvent(new Event('input',{bubbles:true}));};renderReactionPatterns();};}
  function editAdducts(target){return (value,apply,mount)=>{adductModel={fragment_ion_adduct_rule_set:value};adductPath='';mount.append(el('adductApp'));el('adductApp').hidden=false;el('applyParameterAdductRules').hidden=true;el('adductPath').textContent='Embedded in '+(target==='data'?'Data Preparation':'Training');onAdductChange=()=>{apply(adductModel.fragment_ion_adduct_rule_set);form.dispatchEvent(new Event('input',{bubbles:true}));};renderAdductRules();};}
  const editors={data:createParameterEditor(el('dataParameterEditor'),dataInitial,'data',editPatterns('data'),editAdducts('data')),training:createParameterEditor(el('trainingParameterEditor'),initialTraining.modelConfig||{},'training',editPatterns('training'),editAdducts('training'))};
  function snapshot(target,application){const config={application};for(const input of target.elements){if(input.name)config[input.name]=input.type==='number'?Number(input.value):input.type==='checkbox'?input.checked:input.value;}return config;}
  getConfig=()=>{const value=editors.data.getValue();return {...snapshot(form,'fragment-tree-data-preparation'),fragmenterParams:value.fragmenter_params,symbols:value.symbols,maxUniqueFragmentSmiles:value.max_unique_fragment_smiles,maxReactionCombinations:value.max_reaction_combinations};};
  getTrainingConfig=()=>({...snapshot(training,'fragment-tree-training'),modelConfig:editors.training.getValue()});
  for(const target of ['data']){const zone=el(target+'ParameterDrop'),load=el(target+'ParameterLoad'),save=el(target+'ParameterSave');load.onclick=()=>vscode.postMessage({type:'parameters/pick',target});save.onclick=()=>vscode.postMessage({type:'parameters/save',target,fragmenterParams:editors.data.getValue().fragmenter_params});installFileDropZone(zone,{kind:'file',upload:false,onDrop:paths=>vscode.postMessage({type:'parameters/load',target,path:paths[0]}),onFiles:async files=>{const file=files[0];if(file.size>2000000)throw new Error('Fragmenter config is too large.');vscode.postMessage({type:'parameters/import',target,name:file.name,json:await file.text()});},onError:message=>{el(target+'ParameterStatus').textContent=message;}});}
  window.addEventListener('message',event=>{const m=event.data;if(m.type==='training/checkpoint'&&!m.error&&[training.elements.resume.value,training.elements.fineTuneResume.value].includes(m.path)&&['resume','patterns'].includes(training.querySelector('input[name=trainingMode]:checked')?.value)){editors.training.setValue(m.modelConfig);training.elements.lr.value=m.lr;training.elements.weightDecay.value=m.weightDecay;for(const [key,value]of Object.entries(m.trainingSettings||{})){const input=training.elements[key];if(input){if(input.type==='checkbox')input.checked=value;else input.value=value;}}const fine=m.modelConfig.fine_tuning;if(fine){training.elements.fineTuneCheckpoint.value=fine.source_checkpoint||fine.base_checkpoint||fine.checkpoint_path||'';training.elements.headOnly.checked=!!fine.head_only;}}if(m.type==='parameters/loaded'&&editors[m.target]&&!(m.target==='training'&&(training.querySelector('input[name=trainingMode]:checked')?.value==='resume'||training.querySelector('input[name=trainingMode]:checked')?.value==='patterns'&&training.elements.fineTuneResume.value))){editors[m.target].setValue({...editors[m.target].getValue(),fragmenter_params:m.fragmenterParams});const status=el(m.target+'ParameterStatus');if(status)status.textContent='Loaded '+(m.path?.split(/[\\/]/).at(-1)||'fragmenter config')+'.';}if(m.type==='parameters/saved'){const status=el(m.target+'ParameterStatus');if(status)status.textContent='Saved '+(m.path?.split(/[\\/]/).at(-1)||'fragmenter config')+'.';}if(m.type==='parameters/error'){const status=el(m.target+'ParameterStatus');if(status)status.textContent=m.error;}if(m.type==='config'&&(m.config.modelConfig||m.config.fragmenterParams))editors.data.setValue(m.config.modelConfig||{fragmenter_params:m.config.fragmenterParams,symbols:m.config.symbols,max_unique_fragment_smiles:m.config.maxUniqueFragmentSmiles,max_reaction_combinations:m.config.maxReactionCombinations});if(m.type==='trainingConfig'&&m.config.modelConfig)editors.training.setValue(m.config.modelConfig);});
  let timer;const tooltip=el('helpTooltip');document.addEventListener('mouseover',event=>{const label=event.target.closest('[data-help]');if(!label||!label.dataset.help)return;clearTimeout(timer);timer=setTimeout(()=>{const rect=label.getBoundingClientRect();tooltip.textContent=label.dataset.help;tooltip.style.left=Math.max(8,Math.min(rect.left,innerWidth-390))+'px';tooltip.style.top=Math.min(rect.bottom+7,innerHeight-110)+'px';tooltip.classList.add('visible');},500);});document.addEventListener('mouseout',event=>{if(event.target.closest('[data-help]')){clearTimeout(timer);tooltip.classList.remove('visible');}});
}
function importHtml(target){if(target==='training')return '<div id="trainingParameterEditor" class="parameter-editor"></div>';return `<div id="${target}ParameterDrop" class="fragmenter-config-tools" title="Drop a fragmenter config JSON here"><span id="${target}ParameterStatus" role="status">Fragmenter config</span><button type="button" id="${target}ParameterLoad">Load</button><button type="button" id="${target}ParameterSave">Save</button></div><div id="${target}ParameterEditor" class="parameter-editor"></div>`;}
module.exports={createParameterEditor,importHtml,script:()=>`const molEncoderFields=${JSON.stringify(molEncoderFields)};const molDescriptors=${JSON.stringify(molDescriptors)};${createParameterEditor.toString()}(${client.toString()})();`,css:()=>`.parameter-editor fieldset{border:1px solid var(--border);border-radius:5px;margin:12px 0;padding:14px;min-width:0}.parameter-editor legend{font-weight:600}.parameter-editor input[type=checkbox]{width:auto}.parameter-array-item{border-left:2px solid var(--border);margin:10px 0;padding:0 12px}.parameter-editor fieldset>label{display:inline-block;vertical-align:top;width:calc(50% - 16px);margin-right:16px}.parameter-editor .check{display:inline-flex}.parameter-group-heading{flex-basis:100%;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;opacity:.65;margin:14px 0 2px}.parameter-group-heading:first-child{margin-top:0}.parameter-badge{display:inline-block;padding:1px 5px;border:1px solid var(--border);border-radius:10px;font-size:9px;opacity:.75}.fragmenter-config-tools{display:flex;align-items:center;justify-content:flex-end;gap:7px;min-height:30px;margin:-12px 0 10px;padding:4px 0;border:1px dashed transparent}.fragmenter-config-tools span{max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--quiet);font-size:11px}.fragmenter-config-tools button{padding:4px 10px;font-size:11px}.fragmenter-config-tools.path-drop-active{border-color:var(--accent);background:var(--soft)}#helpTooltip{pointer-events:none}@media(max-width:700px){.parameter-editor fieldset>label{width:100%}.fragmenter-config-tools span{max-width:150px}}`};
