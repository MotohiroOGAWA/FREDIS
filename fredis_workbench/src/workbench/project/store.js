const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const SCHEMA = 'fredis.workbench.project';
const workflows = ['data', 'training', 'prediction', 'reactionPatterns'];
const pathFields = new Set(['input','validationInput','trainInput','params','outputDir','batchInput','batchOutputDir','modelPath','datasetDir','trainDir','valDir','resume','initializeFrom','fineTuneCheckpoint','fineTuneResume','integrationHead','integrationOutput','checkpoint_path','path','source']);
function metadata(root) { return path.join(root, '.fredis'); }
function atomic(file, value) {
  fs.mkdirSync(path.dirname(file), {recursive:true});
  const temp = file + '.' + crypto.randomUUID() + '.tmp';
  try { fs.writeFileSync(temp, JSON.stringify(value, null, 2) + '\n'); fs.renameSync(temp, file); }
  finally { if (fs.existsSync(temp)) fs.unlinkSync(temp); }
}
function read(root) {
  const value = JSON.parse(fs.readFileSync(path.join(metadata(root),'project.json'),'utf8'));
  if (value.schema !== SCHEMA || value.version !== 1 || typeof value.id !== 'string' || typeof value.name !== 'string' || typeof value.notes !== 'string' || !Array.isArray(value.configurations) || !Array.isArray(value.files) || value.configurations.some(item=>!item||typeof item.id!=='string'||!workflows.includes(item.workflow)||typeof item.label!=='string') || value.files.some(item=>!item||typeof item.id!=='string'||typeof item.path!=='string'||!Array.isArray(item.roles)||!Array.isArray(item.workflows))) throw new Error('Unsupported or damaged FREDIS project metadata. The existing file was not changed.');
  return value;
}
function save(root, value) { value.updatedAt=new Date().toISOString(); atomic(path.join(metadata(root),'project.json'),value); return value; }
function open(directory) {
  const root=fs.realpathSync(path.resolve(directory));
  if(!fs.statSync(root).isDirectory())throw new Error('Choose a project directory.');
  const file=path.join(metadata(root),'project.json');
  if(!fs.existsSync(file))save(root,{schema:SCHEMA,version:1,id:crypto.randomUUID(),name:path.basename(root),notes:'',createdAt:new Date().toISOString(),configurations:[],files:[],drafts:{}});
  read(root);return root;
}
function stamp(file) { try { const stat=fs.statSync(file);return {exists:true,directory:stat.isDirectory(),size:stat.size,mtimeMs:stat.mtimeMs}; } catch(error) { return {exists:false,error:error.code}; } }
function references(config,base) {
  const result=[];
  function visit(value,key='') {
    if(pathFields.has(key))for(const entry of Array.isArray(value)?value:[value])if(typeof entry==='string'&&entry.trim())result.push({path:path.resolve(base,entry.trim()),role:key});
    if(Array.isArray(value)&&!pathFields.has(key))for(const item of value)visit(item,key);
    if(value&&typeof value==='object'&&!Array.isArray(value))for(const [child,item]of Object.entries(value))visit(item,child);
  }
  visit(config);return result;
}
function addReferences(value, refs, workflow, configurationId) {
  for(const ref of refs){let file=value.files.find(item=>item.path===ref.path);const now=new Date().toISOString();
    if(!file){file={id:crypto.randomUUID(),path:ref.path,roles:[],workflows:[],firstSeen:now,pinned:false};value.files.push(file);}
    if(!file.roles.includes(ref.role))file.roles.push(ref.role);
    if(workflow&&!file.workflows.includes(workflow))file.workflows.push(workflow);
    file.lastSeen=now;if(configurationId)file.configurationId=configurationId;
    file.lastRecorded=stamp(ref.path);
  }
}
function validateWorkflow(workflow,config) { if(!workflows.includes(workflow)||!config||typeof config!=='object'||Array.isArray(config))throw new Error('Invalid project configuration.'); }
function snapshot(root,workflow,config,{label,reason='saved',base=root,jobId,sourcePath}={}) {
  validateWorkflow(workflow,config);
  const value=read(root),id=crypto.randomUUID(),refs=references(config,base);
  if(sourcePath)refs.push({path:path.resolve(base,sourcePath),role:'configuration'});
  const record={id,workflow,label:String(label||workflow+' settings').slice(0,200),reason,createdAt:new Date().toISOString(),jobId,sourcePath,files:refs.map(ref=>({...ref,recorded:stamp(ref.path)}))};
  atomic(path.join(metadata(root),'configurations',id+'.json'),{...record,config});
  value.configurations.unshift(record);addReferences(value,refs,workflow,id);save(root,value);return id;
}
function configuration(root,id) {
  if(!read(root).configurations.some(item=>item.id===id))throw new Error('Configuration not found in this project.');
  return JSON.parse(fs.readFileSync(path.join(metadata(root),'configurations',id+'.json'),'utf8'));
}
function drafts(root,values,base=root) {
  const value=read(root);
  for(const [workflow,config]of Object.entries(values)){
    if(workflow==='quick')continue;
    validateWorkflow(workflow,config);
    addReferences(value,references(config,base),workflow);
  }
  atomic(path.join(metadata(root),'drafts.json'),values);
  value.drafts={updatedAt:new Date().toISOString(),workflows:Object.keys(values)};save(root,value);
}
function readDrafts(root) { const file=path.join(metadata(root),'drafts.json'),value=fs.existsSync(file)?JSON.parse(fs.readFileSync(file,'utf8')):{};if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Project form settings are damaged.');for(const [workflow,config]of Object.entries(value))if(workflow!=='quick')validateWorkflow(workflow,config);return value; }
function file(root,id) { const value=read(root).files.find(item=>item.id===id);if(!value)throw new Error('File not found in this project.');return value; }
function pin(root,id) { const value=read(root),item=value.files.find(item=>item.id===id);if(!item)throw new Error('File not found.');item.pinned=!item.pinned;save(root,value); }
function track(root,paths,role='file',workflow,configurationId) { const value=read(root);addReferences(value,paths.map(file=>({path:path.resolve(root,file),role})),workflow,configurationId);save(root,value); }
async function discover(root) {
  const found=[],pending=[{directory:root,depth:0}];let directories=0,truncated=false;
  while(pending.length&&directories<250&&found.length<500){
    const {directory,depth}=pending.pop();directories++;
    let entries;try{entries=await fs.promises.readdir(directory,{withFileTypes:true});}catch{continue;}
    for(const entry of entries){
      if(entry.name.startsWith('.')||['node_modules','__pycache__','venv'].includes(entry.name)||entry.isSymbolicLink())continue;
      const file=path.join(directory,entry.name);
      if(entry.isDirectory()){if(depth<5)pending.push({directory:file,depth:depth+1});else truncated=true;}
      else if(entry.isFile()&&/\.(json|msds|msp|mgf|csv|tsv|parquet|pt|smi|smiles|md|log)$/i.test(entry.name))found.push(file);
      if(found.length>=500){truncated=true;break;}
    }
  }
  track(root,found,'discovered');return {count:found.length,truncated:truncated||pending.length>0};
}
function describe(root) { const value=read(root);return {...value,root,files:value.files.map(item=>{const current=stamp(item.path);return {...item,current,changed:current.exists&&item.lastRecorded?.exists&&!current.directory&&(current.size!==item.lastRecorded.size||current.mtimeMs!==item.lastRecorded.mtimeMs)};})}; }
function notes(root,name,notes) { const value=read(root);value.name=String(name||path.basename(root)).slice(0,200);value.notes=String(notes||'').slice(0,20000);save(root,value); }
function output(root,workflow) { return path.join(root,'runs',workflow+'-'+new Date().toISOString().replace(/[:.]/g,'-')+'-'+crypto.randomBytes(2).toString('hex')); }
module.exports={discover,metadata,open,read,save,stamp,references,snapshot,configuration,drafts,readDrafts,file,pin,track,describe,notes,output,workflows};
