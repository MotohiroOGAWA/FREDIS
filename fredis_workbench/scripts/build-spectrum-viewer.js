const fs = require('fs');
const path = require('path');

function build(source = path.resolve(__dirname, '../../fredis/libs/msentity/vscode-extension'),
  target = path.resolve(__dirname, '../vendor/msentity-spectrum-viewer')) {
  const host = path.join(source, 'extension.js');
  if (!fs.existsSync(host)) {
    throw new Error('Initialize fredis/libs/msentity before building Workbench.');
  }
  for (const entry of ['media', 'python']) {
    if (!fs.statSync(path.join(source, entry)).isDirectory()) throw new Error('Missing msentity viewer ' + entry);
  }
  fs.mkdirSync(target, { recursive: true });
  fs.copyFileSync(host, path.join(target, 'extension.js'));
  fs.copyFileSync(path.resolve(source, '../LICENSE'), path.join(target, 'LICENSE'));
  for (const entry of ['media', 'python']) {
    fs.rmSync(path.join(target, entry), { recursive: true, force: true });
    fs.cpSync(path.join(source, entry), path.join(target, entry), {
      recursive: true,
      filter: file => !file.includes('__pycache__') && !file.endsWith('.pyc')
    });
  }
  return target;
}

if (require.main === module) console.log('Built msentity viewer from submodule: ' + build());
module.exports = { build };
