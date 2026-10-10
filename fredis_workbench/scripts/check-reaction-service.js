const assert=require('assert/strict'),path=require('path'),{spawn}=require('child_process');
const {createClient,forOwner}=require('../src/features/reaction-pattern-set/reaction-service');
const root=path.resolve(__dirname,'../..'),script=path.join(root,'fredis_workbench/src/features/reaction-pattern-set/backend.py');
const pattern={name:'C–O',reactant_smarts:'[#6:1]-[#8:2]',products:[{name:'Split',smarts:'[#6:1].[#8:2]'}]};
(async()=>{let starts=0;const processes=[];const options={python:'python',script,root,spawnProcess:(...args)=>{starts++;const child=spawn(...args);processes.push(child);return child;}};
const client=createClient(options);try{
const preview=await client.request('reactionPreview',{smiles:'CCOCCO',patterns:[pattern]});assert.equal(starts,1);assert(preview.patterns[0].matches.every(m=>!m.loaded&&!m.products.length));
const sites=preview.patterns[0].matches;const products=await Promise.all(sites.map(m=>client.request('reactionPreviewProducts',{smiles:'CCOCCO',pattern,atoms:m.atoms,bonds:m.bonds})));assert.equal(starts,1);assert.deepEqual(products.map(p=>p.products[0].smiles),['CC.OCCO','CCO.CCO','CCOCC.O']);
await assert.rejects(()=>client.request('reactionPreview',{smiles:'invalid',patterns:[pattern]}),/SMILES/);await client.request('reactionPreview',{smiles:'CO',patterns:[pattern]});assert.equal(starts,1,'Errors must not restart the worker');
const killed=processes[0];await new Promise(resolve=>{killed.once('close',resolve);killed.kill();});await client.request('reactionPreview',{smiles:'CO',patterns:[pattern]});assert.equal(starts,2,'Unexpected exit must restart on next request');
}finally{client.dispose();}
await assert.rejects(()=>client.request('reactionPreview',{}),/closed/);
let dispose;const owner={onDidDispose:fn=>{dispose=fn;}};const reused=forOwner(owner,options,{subscriptions:[]});assert.equal(forOwner(owner,options,{subscriptions:[]}),reused);await reused.request('reactionPreview',{smiles:'CO',patterns:[pattern]});assert.equal(starts,3);const pendingClose=reused.request('reactionPreview',{smiles:'CO',patterns:[pattern]});dispose();await assert.rejects(()=>pendingClose,/closed/);await assert.rejects(()=>reused.request('reactionPreview',{}),/closed/);
console.log('Persistent RDKit preview worker: one process for matching/products, concurrent request correlation, errors, restart and owner disposal passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
