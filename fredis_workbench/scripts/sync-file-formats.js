// evaluation/file-formats.json is canonical; extension files and VS Code contributions are generated.
const fs = require('fs');
const path = require('path');
const {contributions} = require('../src/file-formats');
const target = path.join(__dirname,'..','package.json');
const pkg = JSON.parse(fs.readFileSync(target,'utf8'));
const registryPath = path.join(__dirname,'..','..','evaluation','file-formats.json');
const registry = JSON.parse(fs.readFileSync(registryPath,'utf8'));
const runtimePath = path.join(__dirname,'..','src','file-formats.generated.json');
const generated = contributions(registry);
const before = JSON.stringify(pkg);
pkg.contributes.customEditors = generated.customEditors;
pkg.contributes.languages.find(item=>item.id==='fredis-workbench-file').extensions = generated.extensions;
const events = generated.customEditors.map(item=>'onCustomEditor:'+item.viewType);
pkg.activationEvents = [...pkg.activationEvents.filter(item=>!item.startsWith('onCustomEditor:')),...events];
if (process.argv.includes('--check')) {
  if (JSON.stringify(JSON.parse(fs.readFileSync(runtimePath,'utf8')))!==JSON.stringify(registry)) throw new Error('Runtime file format registry is stale. Run npm run sync:file-formats.');
  if (JSON.stringify(pkg)!==before) throw new Error('File format contributions are stale. Run npm run sync:file-formats.');
} else {
  fs.writeFileSync(runtimePath,JSON.stringify(registry,null,2)+'\n');
  fs.writeFileSync(target,JSON.stringify(pkg,null,2)+'\n');
}
