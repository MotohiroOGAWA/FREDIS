const vscode=require('vscode');
const path=require('path');
const store=require('./store');
const ui=require('./ui');
function createContext(context){
  const scoped=Object.create(context);scoped.fredisProject={root:null,error:null};
  const previous=context.workspaceState?.get('fredis.project');
  if(previous)try{store.read(previous);store.readDrafts(previous);scoped.fredisProject.root=previous;}catch(error){scoped.fredisProject.error='Could not reopen project '+previous+': '+error.message;}
  return scoped;
}
function capture(context){const scoped=Object.create(context);scoped.fredisProject={...context.fredisProject};return scoped;}
function root(context){return context.fredisProject?.root||null;}
function record(context,workflow,config,options){const directory=root(context);return directory?store.snapshot(directory,workflow,config,options):undefined;}
function attach(panel,context,applicationRoot){
  const post=value=>panel.webview.postMessage(value);
  const recent=()=>context.globalState?.get('fredis.recentProjects',[])||[];
  const state=(restore=false)=>{const directory=root(context);post({type:'project/state',project:directory?store.describe(directory):null,recent:recent(),restore,drafts:restore&&directory?store.readDrafts(directory):undefined,outputs:directory?Object.fromEntries(['data','training','prediction'].map(key=>[key,store.output(directory,key)])):undefined});};
  const update=()=>{try{state();}catch(error){post({type:'project/error',error:error.message});}};
  const timer=setInterval(()=>{if(root(context))update();},15000);
  panel.onDidDispose(()=>clearInterval(timer));
  let queue=Promise.resolve();
  panel.webview.onDidReceiveMessage(message=>{
    if(!message?.type?.startsWith('project/'))return;
    queue=queue.then(async()=>{
      try{
        if(message.type==='project/ready'){state(true);if(context.fredisProject.error)post({type:'project/error',error:context.fredisProject.error});return;}
        if(['project/open','project/close'].includes(message.type)){
          if(root(context)&&message.root===root(context)&&message.drafts)store.drafts(root(context),message.drafts,applicationRoot(context));
          let directory=null;
          if(message.type==='project/open'){
            let selected=message.path;
            if(!selected){const picked=await vscode.window.showOpenDialog({title:'Choose a FREDIS project directory',canSelectFolders:true,canSelectFiles:false,canSelectMany:false});selected=picked?.[0]?.fsPath;}
            if(!selected){post({type:'project/cancelled'});return;}
            directory=store.open(selected);store.readDrafts(directory);
            const list=[{path:directory,name:store.read(directory).name,openedAt:new Date().toISOString()},...recent().filter(item=>item.path!==directory)].slice(0,15);
            await context.globalState?.update('fredis.recentProjects',list);
          }
          context.fredisProject.root=directory;context.fredisProject.error=null;await context.workspaceState?.update('fredis.project',directory);state(true);return;
        }
        if(message.type==='project/refresh'){state();return;}
        if(!root(context)||message.root!==root(context))throw new Error('This project changed. Refresh Home before editing its history.');
        const directory=root(context);
        if(message.type==='project/track'){if(!Array.isArray(message.paths)||message.paths.some(file=>typeof file!=='string'))throw new Error('Invalid file references.');store.track(directory,message.paths.filter(file=>file.trim()).map(file=>path.resolve(applicationRoot(context),file)),'selected',store.workflows.includes(message.workflow)?message.workflow:undefined);return;}
        if(message.type==='project/drafts'){store.drafts(directory,message.drafts,applicationRoot(context));post({type:'project/draftsSaved',root:directory,at:new Date().toISOString()});return;}
        if(message.type==='project/snapshot'){store.snapshot(directory,message.workflow,message.config,{label:message.label,reason:message.reason||'saved',base:applicationRoot(context),sourcePath:message.sourcePath});state();}
        if(message.type==='project/notes'){store.notes(directory,message.name,message.notes);post({type:'project/notesSaved'});state();}
        if(message.type==='project/pin'){store.pin(directory,message.id);state();}
        if(message.type==='project/discover'){const result=await store.discover(directory);if(root(context)===directory){state();post({type:'project/discovered',...result});}}
        if(message.type==='project/addFiles'){
          const selected=await vscode.window.showOpenDialog({title:'Track project files',canSelectFiles:true,canSelectFolders:false,canSelectMany:true,defaultUri:vscode.Uri.file(directory)});
          if(selected?.length)store.track(directory,selected.map(uri=>uri.fsPath));state();
        }
        if(message.type==='project/restore'){
          const saved=store.configuration(directory,message.id),config=JSON.parse(JSON.stringify(saved.config));
          if(saved.workflow!=='reactionPatterns'){
            const outputField=saved.workflow==='prediction'&&'batchOutputDir' in config?'batchOutputDir':'outputDir';
            config[outputField]=store.output(directory,saved.workflow);
            if('overwrite' in config)config.overwrite=false;if('batchOverwrite' in config)config.batchOverwrite=false;
          }
          post({type:'project/restore',workflow:saved.workflow,config,label:saved.label});
        }
        if(message.type==='project/compare'){
          const current=store.configuration(directory,message.id),history=store.read(directory).configurations,index=history.findIndex(item=>item.id===message.id);
          const previous=history.slice(index+1).find(item=>item.workflow===current.workflow);
          if(!previous)throw new Error('There is no earlier configuration for this workflow.');
          const before=store.configuration(directory,previous.id),changes=[];
          function compare(a,b,key=''){
            if(JSON.stringify(a)===JSON.stringify(b))return;
            if(a&&b&&typeof a==='object'&&typeof b==='object'&&!Array.isArray(a)&&!Array.isArray(b))for(const name of new Set([...Object.keys(a),...Object.keys(b)]))compare(a[name],b[name],key?key+'.'+name:name);
            else changes.push({field:key,before:a,after:b});
          }
          compare(before.config,current.config);post({type:'project/comparison',before:before.label,after:current.label,changes});
        }
        if(message.type==='project/viewConfig'){
          store.configuration(directory,message.id);await vscode.window.showTextDocument(vscode.Uri.file(path.join(store.metadata(directory),'configurations',message.id+'.json')));
        }
        if(message.type==='project/file'){
          const item=store.file(directory,message.id);
          if(message.action==='copy')await vscode.env.clipboard.writeText(item.path);
          else if(!store.stamp(item.path).exists)throw new Error('File is missing: '+item.path);
          else if(message.action==='open'&&!store.stamp(item.path).directory)await vscode.commands.executeCommand('vscode.open',vscode.Uri.file(item.path));
          else await vscode.commands.executeCommand('revealInExplorer',vscode.Uri.file(item.path));
        }
        if(message.type==='project/reveal')await vscode.commands.executeCommand('revealInExplorer',vscode.Uri.file(directory));
      }catch(error){post({type:'project/error',error:error.message});}
    });
    return queue;
  });
}
module.exports={capture,createContext,root,record,attach,store,html:ui.html,script:ui.script,css:ui.css};
