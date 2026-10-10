const fs = require('fs');
const path = require('path');

let configuredRoot = '';

function isViewerRoot(candidate) {
  return Boolean(candidate) && [
    'extension.js', 'media/viewer.js', 'media/viewer.css',
    'media/spectrum.js', 'python/backend.py', 'python/similarity_backend.py'
  ].every(file => fs.existsSync(path.join(candidate, file)));
}

function configure(context, vscode) {
  const repository = path.resolve(__dirname, '../../../../fredis/libs/msentity/vscode-extension');
  const besideSourceExtension = context?.extensionPath
    ? path.resolve(context.extensionPath, '../fredis/libs/msentity/vscode-extension') : '';
  const workspaceSources = (vscode?.workspace?.workspaceFolders || []).flatMap(folder => {
    const root = folder.uri.fsPath;
    return [
      path.join(root, 'fredis/libs/msentity/vscode-extension'),
      path.join(root, 'mnt/app/fredis/libs/msentity/vscode-extension')
    ];
  });
  const bundled = context?.extensionPath
    ? path.join(context.extensionPath, 'vendor/msentity-spectrum-viewer') : '';
  // Always use submodule source when it is available. Packaged Workbenches
  // carry a snapshot built from that same source; no installed msentity VSIX.
  configuredRoot = [repository, besideSourceExtension, ...workspaceSources, bundled].find(isViewerRoot) || '';
  if (!configuredRoot) {
    throw new Error('msentity viewer source was not found. Initialize fredis/libs/msentity or rebuild Workbench with npm run build:spectrum-viewer.');
  }
  return configuredRoot;
}

function root(context, vscode) {
  if (configuredRoot && isViewerRoot(configuredRoot)) return configuredRoot;
  return configure(context, vscode);
}

function file(...parts) {
  return path.join(root(), ...parts);
}

module.exports = { configure, root, file };
