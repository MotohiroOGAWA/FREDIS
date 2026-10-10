// Receive external drag-and-drop files when a webview cannot access their paths.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const CHUNK_SIZE=524288;
function attach(panel,context){
 const uploads=new Map();
 async function discard(id){const record=uploads.get(id);if(!record)return;uploads.delete(id);await record.handle.close().catch(()=>{});await fs.promises.unlink(record.path).catch(()=>{});}
 panel.onDidDispose(()=>{for(const id of uploads.keys())void discard(id);});
 panel.webview.onDidReceiveMessage(async message=>{
  if(!['dataset/uploadStart','dataset/uploadChunk','dataset/uploadFinish','dataset/uploadCancel'].includes(message?.type))return;
  const reply=data=>panel.webview.postMessage({type:'dataset/uploadResponse',requestId:message.requestId,...data});
  try{
   if(message.type==='dataset/uploadStart'){
    const name=path.basename(String(message.name||'dataset'));
    if(!name || name==='.' || name==='..')throw new Error('Invalid file name.');
    if(!Number.isSafeInteger(message.size)||message.size<0)throw new Error('Invalid dataset file size.');
    const storage=context.storageUri||context.globalStorageUri;if(!storage)throw new Error('Workbench storage is unavailable. Use Browse.');
    const directory=path.join(storage.fsPath,'datasets');await fs.promises.mkdir(directory,{recursive:true});
    const id=crypto.randomUUID(),file=path.join(directory,id+path.extname(name).toLowerCase());
    uploads.set(id,{path:file,size:message.size,offset:0,handle:await fs.promises.open(file,'wx')});reply({id});
   }else{
    const record=uploads.get(message.id);if(!record)throw new Error('Dataset upload expired. Drop the file again.');
    if(message.type==='dataset/uploadCancel'){await discard(message.id);reply({});return;}
    if(message.type==='dataset/uploadChunk'){
     if(typeof message.data!=='string'||message.data.length>Math.ceil(CHUNK_SIZE/3)*4||!/^[A-Za-z0-9+/]*={0,2}$/.test(message.data))throw new Error('Invalid dataset upload chunk.');
     const buffer=Buffer.from(message.data,'base64');if(record.offset+buffer.length>record.size)throw new Error('Dataset upload exceeds the declared file size.');
     let written=0;while(written<buffer.length){const result=await record.handle.write(buffer,written,buffer.length-written,record.offset+written);if(!result.bytesWritten)throw new Error('Unable to write the dataset file.');written+=result.bytesWritten;}
     record.offset+=written;reply({offset:record.offset});
    }else{
     if(record.offset!==record.size)throw new Error('Dataset upload is incomplete.');
     await record.handle.close();uploads.delete(message.id);reply({path:record.path});
    }
   }
  }catch(error){if(message.id)await discard(message.id);reply({error:error.message});}
 });
}
function client(){
 const pending=new Map();let sequence=0;
 window.addEventListener('message',event=>{const message=event.data;if(message.type!=='dataset/uploadResponse')return;const entry=pending.get(message.requestId);if(!entry)return;pending.delete(message.requestId);clearTimeout(entry.timer);message.error?entry.reject(new Error(message.error)):entry.resolve(message);});
 function request(message){return new Promise((resolve,reject)=>{const requestId='dataset-upload-'+(++sequence);const timer=setTimeout(()=>{pending.delete(requestId);reject(new Error('Dataset upload timed out. Try Browse.'));},30000);pending.set(requestId,{resolve,reject,timer});vscode.postMessage({...message,requestId});});}
 window.uploadDataset=async(file,progress)=>{
  const {id}=await request({type:'dataset/uploadStart',name:file.name,size:file.size});
  try{
   for(let offset=0;offset<file.size;offset+=524288){const bytes=new Uint8Array(await file.slice(offset,offset+524288).arrayBuffer());let binary='';for(const byte of bytes)binary+=String.fromCharCode(byte);const reply=await request({type:'dataset/uploadChunk',id,data:btoa(binary)});progress?.(Math.round(reply.offset/file.size*100));}
   return (await request({type:'dataset/uploadFinish',id})).path;
  }catch(error){vscode.postMessage({type:'dataset/uploadCancel',id});throw error;}
 };
}
module.exports={attach,script:()=>`(${client.toString()})();`};
