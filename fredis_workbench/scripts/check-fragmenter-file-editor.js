const assert=require('assert/strict'),fs=require('fs'),path=require('path'),os=require('os'),Module=require('module');
const {JSDOM,VirtualConsole}=require('jsdom');
const extensionPath=path.resolve(__dirname,'..'),project=path.resolve(extensionPath,'..');
const providers=new Map();let changed,document,currentText,saveDialogResult,openDialogResult;const informationMessages=[];
const vscode={
  window:{
    registerCustomEditorProvider:(name,provider)=>{providers.set(name,provider);return {dispose(){}};},
    showSaveDialog:async()=>saveDialogResult,
    showOpenDialog:async()=>openDialogResult,
    showInformationMessage:text=>{informationMessages.push(text);}
  },
  workspace:{
    getConfiguration:()=>({get:(key,fallback)=>fallback}),
    workspaceFolders:[],
    onDidChangeTextDocument:fn=>{changed=fn;return {dispose(){}};},
    applyEdit:async edit=>{currentText=edit.text;changed({document});return true;}
  },
  env:{clipboard:{writeText:async()=>{}}},
  Range:class{constructor(a,b){this.a=a;this.b=b;}},
  WorkspaceEdit:class{replace(uri,range,text){this.text=text;}},
  Uri:{file:p=>({fsPath:p})}
};
const load=Module._load;Module._load=function(name,...args){return name==='vscode'?vscode:load.call(this,name,...args);};
const feature=require('../src/features/fragmenter/editor');
const {ensureFileSuffix}=require('../src/features/reaction-pattern-set/host');
const context={subscriptions:[],extensionPath};feature.register(context);

// Manifest: the dedicated extension opens in this editor and gets the FREDIS icon.
const manifest=require('../package.json');
const contributed=manifest.contributes.customEditors.find(editor=>editor.viewType===feature.VIEW_TYPE);
assert(contributed,'package.json must contribute the Fragmenter editor');
assert.deepEqual(contributed.selector.map(item=>item.filenamePattern),['*.fragmenter.pft']);
assert(manifest.activationEvents.includes('onCustomEditor:'+feature.VIEW_TYPE));
assert(manifest.contributes.languages[0].extensions.includes(feature.FILE_SUFFIX));
assert.equal(ensureFileSuffix('/tmp/a',feature.FILE_SUFFIX),'/tmp/a.fragmenter.pft');
assert.equal(ensureFileSuffix('/tmp/a.json',feature.FILE_SUFFIX),'/tmp/a.fragmenter.pft');

// Content: the file stores fragmenter_params itself; model configs are rejected.
const preset=JSON.parse(fs.readFileSync(path.join(project,'fredis/presets/spectrum_generator_params/source_anchored_pos_model_config.json'),'utf8')).fragmenter_params;
assert.deepEqual(feature.normalizeDocument({fragmenter_params:preset}),preset);
assert.throws(()=>feature.normalizeDocument({params:{fragmenter_params:preset}}),/not a model config/);

(async()=>{
  const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-fragmenter-')),file=path.join(tmp,'mine'+feature.FILE_SUFFIX);
  // A partial file: missing options are filled from the project preset for editing.
  const partial={fragment_ion_tree_builder:{max_reaction_count:2,reaction_pattern_set:preset.fragment_ion_tree_builder.reaction_pattern_set,fragment_ion_adduct_rule_set:preset.fragment_ion_tree_builder.fragment_ion_adduct_rule_set}};
  currentText=JSON.stringify(partial);
  document={uri:{fsPath:file,toString:()=>file},getText:()=>currentText,positionAt:offset=>offset};
  const posted=[],errors=[],sink=new VirtualConsole();sink.on('jsdomError',error=>errors.push(error.message));
  let receive;const panel={webview:{options:{},html:'',postMessage:message=>{posted.push(message);setTimeout(()=>dom.window.dispatchEvent(new dom.window.MessageEvent('message',{data:message})));},onDidReceiveMessage:fn=>{receive=fn;}},onDidDispose(){}};
  providers.get(feature.VIEW_TYPE).resolveCustomTextEditor(document,panel);
  const toHost=[];let dom=null;
  dom=new JSDOM(panel.webview.html,{runScripts:'dangerously',virtualConsole:sink,beforeParse(window){window.structuredClone=value=>JSON.parse(JSON.stringify(value));window.acquireVsCodeApi=()=>({postMessage:message=>{toHost.push(message);receive(message);},getState:()=>({}),setState(){}});window.HTMLElement.prototype.scrollIntoView=function(){};}});
  const window=dom.window,el=id=>window.document.getElementById(id),wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  try{
    await wait(20);
    assert.equal(toHost[0].type,'ready');
    const block=el('fragmenterParameterEditor').querySelector('[data-model-block="fragmenter_params"]');
    assert(block,'the shared Fragmentation editor must be mounted');
    assert(!el('fragmenterParameterEditor').querySelector('.periodic-table'),'a fragmenter file has no data-preparation symbols');
    assert(!el('fragmenterParameterEditor').querySelector('[data-parameter-path="max_unique_fragment_smiles"]'),'search limits belong to data preparation, not to the fragmenter file');
    assert.equal(el('fragmenterPath').textContent,file);
    // Editing a field writes the whole fragmenter_params back into the document.
    const maxReactions=el('fragmenterParameterEditor').querySelector('[data-parameter-path="fragmenter_params.fragment_ion_tree_builder.max_reaction_count"]');
    assert(maxReactions,'Max Reaction Count must be editable');
    assert.equal(maxReactions.value,'2');
    maxReactions.value='3';maxReactions.dispatchEvent(new window.Event('input',{bubbles:true}));
    await wait(260);
    const written=JSON.parse(currentText);
    assert.equal(written.fragment_ion_tree_builder.max_reaction_count,3);
    assert.equal(written.precursor_candidate_max_reaction_count,preset.precursor_candidate_max_reaction_count,'missing preset options are written back');
    assert(!('fragmenter_params' in written),'the document stores fragmenter_params itself');
    // The embedded Reaction Pattern Set editor edits the same document.
    const reactionPatternDetails=[...el('fragmenterParameterEditor').querySelectorAll('details')].find(item=>item.querySelector('summary').textContent.startsWith('Reaction Pattern Set'));
    reactionPatternDetails.open=true;reactionPatternDetails.dispatchEvent(new window.Event('toggle'));
    assert(reactionPatternDetails.contains(el('reactionPatternApp'))&&!el('reactionPatternApp').hidden,'the Reaction Pattern Set editor must be embedded');
    window.eval("reactionPatternModel.reaction_pattern_set.name='renamed_set';onReactionPatternChange();");
    await wait(260);
    assert.equal(JSON.parse(currentText).fragment_ion_tree_builder.reaction_pattern_set.name,'renamed_set');
    const adductDetails=[...el('fragmenterParameterEditor').querySelectorAll('details')].find(item=>item.querySelector('summary').textContent.startsWith('Fragment Ion Adduct Rule Set'));
    adductDetails.open=true;adductDetails.dispatchEvent(new window.Event('toggle'));
    assert(adductDetails.contains(el('adductApp'))&&!el('adductApp').hidden,'the Adduct Rule Set editor must be embedded');
    // External edits (e.g. the text editor) refresh the form.
    currentText=JSON.stringify({...JSON.parse(currentText),mass_tolerance:'0.02Da'});changed({document});await wait(10);
    assert.equal(el('fragmenterParameterEditor').querySelector('[data-parameter-path="fragmenter_params.mass_tolerance"]').value,'0.02Da');
    // Save As writes a copy with the dedicated suffix.
    saveDialogResult={fsPath:path.join(tmp,'copy.json')};el('fragmenterSaveAs').click();await wait(20);
    const copy=path.join(tmp,'copy'+feature.FILE_SUFFIX);
    assert(fs.existsSync(copy),'Save As must enforce '+feature.FILE_SUFFIX);
    assert.equal(JSON.parse(fs.readFileSync(copy,'utf8')).mass_tolerance,'0.02Da');
    assert(el('fragmenterStatus').textContent.includes('copy'+feature.FILE_SUFFIX));
    // Load imports another config (a model-style wrapper is unpacked) into this file.
    const other=path.join(tmp,'other'+feature.FILE_SUFFIX);fs.writeFileSync(other,JSON.stringify({fragmenter_params:{...preset,mass_tolerance:'5ppm'}}));
    openDialogResult=[{fsPath:other}];el('fragmenterLoad').click();await wait(30);
    assert.equal(JSON.parse(currentText).mass_tolerance,'5ppm');
    assert.equal(el('fragmenterParameterEditor').querySelector('[data-parameter-path="fragmenter_params.mass_tolerance"]').value,'5ppm');
    fs.writeFileSync(other,JSON.stringify({params:{fragmenter_params:preset}}));el('fragmenterLoad').click();await wait(30);
    assert.match(el('fragmenterError').textContent,/not a model config/);
    assert.deepEqual(errors,[]);
    console.log('Fragmenter file editor: dedicated .fragmenter.pft editor, shared Fragmentation UI, embedded pattern/adduct editors, document sync, Save As and Load passed.');
  }finally{window.close();fs.rmSync(tmp,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
