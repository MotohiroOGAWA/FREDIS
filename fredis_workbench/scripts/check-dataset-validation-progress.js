const assert=require('assert/strict');
const fs=require('fs');
const path=require('path');
const Module=require('module');
const originalLoad=Module._load;
Module._load=function(name,parent,main){
  if(name==='vscode')return {workspace:{getConfiguration:()=>({get:()=> 'python'})}};
  return originalLoad.call(this,name,parent,main);
};
const services=require('../src/workbench/services');
Module._load=originalLoad;

(async()=>{
  const root=path.resolve(__dirname,'../..');
  const preset=JSON.parse(fs.readFileSync(path.join(root,'fredis/presets/spectrum_generator_params/source_anchored_pos_model_config.json'),'utf8'));
  const file=path.join(root,'data/minidata/MSMS-Pos-NIST23_cf_mini.msds');
  const progress=[];
  const result=await services.backend({extensionPath:path.resolve(__dirname,'..')},root,'preview',{
    path:file,validateValues:true,reportProgress:true,fragmenterParams:preset.fragmenter_params,
  },{timeoutMs:0,onProgress:value=>progress.push(value)});
  assert(result.valuesChecked);
  assert.equal(result.records,4182);
  assert(progress.length>2);
  assert.deepEqual(progress[0],{current:0,total:4182});
  assert.deepEqual(progress.at(-1),{current:4182,total:4182});
  console.log('Full-dataset validation and streamed progress checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
