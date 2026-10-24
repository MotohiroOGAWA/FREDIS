// Stream legacy JSON once; persist compact previews and standard MSDataset files.
const fs=require('fs');const path=require('path');const {spawn}=require('child_process');
const {readValidation}=require('../src/features/training-metrics/validation-reader');
(async()=>{
  const run=path.resolve(process.argv[2]||'');
  if(!process.argv[2])throw new Error('Usage: node scripts/convert-validation-spectra.js /path/to/run');
  const directory=path.join(run,'spectrum_validation');
  const files=fs.readdirSync(directory).filter(file=>/^epoch_\d+\.json$/.test(file)).sort((a,b)=>Number(a.match(/\d+/)[0])-Number(b.match(/\d+/)[0]));
  for(const file of files){
    const source=path.join(directory,file),output=source.replace(/\.json$/,'.msds');
    if(fs.existsSync(output))continue;
    console.log('Reading '+file+'…');
    const before=fs.statSync(source);const preview=await readValidation(source);const after=fs.statSync(source);
    if(before.size!==after.size||before.mtimeMs!==after.mtimeMs)throw new Error(file+' changed during conversion; retry.');
    const temporary=source+'.export-preview.tmp';fs.writeFileSync(temporary,JSON.stringify(preview));
    try{
      await new Promise((resolve,reject)=>{
        const child=spawn(process.env.PYTHON||'python',['-m','fredis.ml.training.fragment_tree_training.validation_storage',temporary,output,'--run-directory',run],{cwd:path.resolve(__dirname,'../..'),stdio:['ignore','inherit','inherit']});
        child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(new Error('MSDataset export failed: '+code)));
      });
    }finally{fs.rmSync(temporary,{force:true});}
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
