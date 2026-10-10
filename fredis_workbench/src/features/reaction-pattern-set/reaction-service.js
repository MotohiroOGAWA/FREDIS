/* One lazy Python/RDKit worker per preview owner, reused across requests. */
const {spawn}=require('child_process');
const clients=new WeakMap();
function createClient({python,script,root,spawnProcess=spawn}){
  let child=null,buffer='',stderr='',sequence=0,disposed=false;
  const pending=new Map();
  function rejectAll(error){for(const waiter of pending.values())waiter.reject(error);pending.clear();}
  function start(){if(child)return;const worker=spawnProcess(python,[script,'--server'],{cwd:root,env:{...process.env,PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(require('path').delimiter)}});child=worker;buffer='';stderr='';
    worker.stdout.on('data',chunk=>{if(child!==worker)return;buffer+=chunk.toString();let newline;while((newline=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,newline).trim();buffer=buffer.slice(newline+1);if(!line)continue;try{const reply=JSON.parse(line),waiter=pending.get(reply.requestId);if(!waiter)continue;pending.delete(reply.requestId);reply.ok?waiter.resolve(reply.result):waiter.reject(new Error(reply.error));}catch(error){rejectAll(new Error('The RDKit worker returned an invalid response.'));child=null;worker.kill();return;}}});
    worker.stderr.on('data',chunk=>{stderr=(stderr+chunk.toString()).slice(-8192);});
    worker.stdin.on('error',error=>{if(child!==worker)return;child=null;rejectAll(error);worker.kill();});
    worker.on('error',error=>{if(child!==worker)return;child=null;rejectAll(error);});
    worker.on('close',()=>{if(child!==worker)return;child=null;rejectAll(new Error(stderr||'The RDKit preview worker stopped.'));});
  }
  return {request(command,payload){if(disposed)return Promise.reject(new Error('The reaction preview is closed.'));return new Promise((resolve,reject)=>{try{start();const requestId=++sequence;pending.set(requestId,{resolve,reject});child.stdin.write(JSON.stringify({requestId,command,payload})+'\n',error=>{if(error){pending.delete(requestId);reject(error);}});}catch(error){reject(error);}});},dispose(){disposed=true;rejectAll(new Error('The reaction preview is closed.'));const worker=child;child=null;if(worker)worker.kill();}};
}
function forOwner(owner,options,context){const key=JSON.stringify([options.python,options.script,options.root]);let entry=clients.get(owner);if(!entry||entry.key!==key){if(entry)entry.client.dispose();const client=createClient(options);entry={key,client};clients.set(owner,entry);if(owner.onDidDispose)owner.onDidDispose(()=>client.dispose());else context.subscriptions?.push(client);}return entry.client;}
module.exports={createClient,forOwner};
