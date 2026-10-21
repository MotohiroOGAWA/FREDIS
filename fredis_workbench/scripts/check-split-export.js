const assert=require('assert/strict'),fs=require('fs'),os=require('os'),path=require('path'),Module=require('module'),{spawnSync}=require('child_process');
const root=path.resolve(__dirname,'../..'),temporary=fs.mkdtempSync(path.join(os.tmpdir(),'fredis-split-export-check-'));
const source=path.join(temporary,'input.msds'),output=path.join(temporary,'export');
let picks=0,handler,dispose;
const original=Module._load;
Module._load=function(name,...args){if(name==='vscode')return {workspace:{getConfiguration:()=>({get:()=> 'python'})},window:{showOpenDialog:async()=>{picks++;return [{fsPath:output}];}}};return original.call(this,name,...args);};
const services=require('../src/workbench/services');Module._load=original;
(async()=>{
 try{
  const fixture=spawnSync('python',['-c','import sys; from tests.ml.reaction.TestDatasetSplits import dataset; dataset(["CCO","OCC","CCN"]).save(sys.argv[1])',source],{cwd:root,encoding:'utf8'});
  assert.equal(fixture.status,0,fixture.stderr);
  const posted=[];
  services.attach({webview:{postMessage:m=>posted.push(m),onDidReceiveMessage:fn=>handler=fn},onDidDispose:fn=>dispose=fn},{extensionPath:path.resolve(__dirname,'..')},()=>root);
  await handler({type:'data/export-splits',requestId:'invalid',checkedId:'unknown',checkRequestId:'unknown'});
  assert.equal(picks,0);assert.equal(posted.at(-1).type,'data/error');
  const chemistry=require('./fixtures').dataConfig;
  await handler({type:'data/preflight',requestId:'check-1',config:{...chemistry,datasets:[{id:'dataset-1',path:source}],datasetMode:'automatic',validationRatio:0,testRatio:0,validationSeed:7}});
  const checked=posted.find(m=>m.type==='data/preflight');assert(checked,JSON.stringify(posted));
  assert(checked.checkedId);assert.equal(checked.splits.train.records,3);assert.equal(checked.splits.test.records,0);
  const progress=posted.filter(m=>m.type==='data/preflight-progress');assert(progress.length>3);assert.equal(progress.at(-1).percent,100);
  await handler({type:'data/export-splits',requestId:'export-1',checkedId:checked.checkedId,checkRequestId:'check-1'});
  assert.equal(picks,1);
  const exported=posted.find(m=>m.type==='data/splits-exported');assert(exported,JSON.stringify(posted));
  assert.deepEqual(Object.keys(exported.files),['train']);assert(fs.existsSync(path.join(output,'train.msds')));
  assert(!fs.existsSync(path.join(output,'validation.msds')));assert(!fs.existsSync(path.join(output,'test.msds')));
  assert(posted.some(m=>m.type==='data/splits-export-progress'&&m.percent===100));
  await handler({type:'data/export-splits',requestId:'export-2',checkedId:checked.checkedId,checkRequestId:'check-1'});
  assert.equal(posted.at(-1).type,'data/error');assert.match(posted.at(-1).error,/already contains/);
  console.log('Workbench checks and exports through the split-datasets CLI, streams progress, requires a checked plan, and omits empty datasets.');
 }finally{dispose?.();fs.rmSync(temporary,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
