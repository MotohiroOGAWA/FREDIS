const fs = require('fs');
const path = require('path');
const os = require('os');
const Module = require('module');
const childProcess = require('child_process');

// Load upstream unchanged, intercepting only its VS Code registration and host
// environment. Its provider, backend protocol and chemistry stay upstream-owned.
function createHost(root, vscode, options = {}) {
  let provider;
  let backend;
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'fredis-viewer-'));
  const bootstrap = path.join(temporary, 'bootstrap.msp');
  fs.writeFileSync(bootstrap, 'Name: Workbench bootstrap\nNum Peaks: 1\n1 1\n\n');
  const disposables = [];
  const disposable = () => ({ dispose() {} });
  const configuration = vscode.workspace.getConfiguration('msentitySpectrumViewer');
  const overrides = {
    pythonPath: options.pythonPath,
    pageSize: options.pageSize,
    spectrumFloatingWindow: false
  };
  // VS Code exposes gated proposed APIs as enumerable getters. Copying an
  // API namespace evaluates them, even when upstream never uses those APIs.
  // Forward only the properties actually requested by the viewer.
  const overlay = (api, replacements) => new Proxy({}, {
    get: (_target, key) => Object.prototype.hasOwnProperty.call(replacements, key)
      ? replacements[key] : Reflect.get(api, key, api)
  });
  const hostVscode = overlay(vscode, {
    workspace: overlay(vscode.workspace, {
      getConfiguration: section => section === 'msentitySpectrumViewer' ? {
        get: (key, fallback) => overrides[key] ?? configuration.get(key, fallback)
      } : vscode.workspace.getConfiguration(section)
    }),
    window: overlay(vscode.window, {
      createOutputChannel: () => options.outputChannel,
      registerCustomEditorProvider: (type, instance) => {
        if (type === 'msentity.spectrumViewer') provider = instance;
        return disposable();
      }
    }),
    commands: overlay(vscode.commands, { registerCommand: disposable })
  });
  const filename = path.join(root, 'extension.js');
  const upstreamModule = new Module(filename, module);
  upstreamModule.filename = filename;
  upstreamModule.paths = Module._nodeModulePaths(root);
  const originalRequire = upstreamModule.require.bind(upstreamModule);
  upstreamModule.require = name => {
    if (name === 'vscode') return hostVscode;
    if (name === 'child_process') return {
      ...childProcess,
      spawn: (python, args, spawnOptions) => {
        const parameters = [...args];
        if (parameters[1] === path.join(root, 'python/backend.py')) parameters[2] = bootstrap;
        backend = childProcess.spawn(python, parameters, spawnOptions);
        return backend;
      }
    };
    return originalRequire(name);
  };
  try {
    upstreamModule._compile(fs.readFileSync(filename, 'utf8'), filename);
    upstreamModule.exports.activate({
      extensionPath: root, extensionUri: vscode.Uri.file(root), subscriptions: disposables,
      asAbsolutePath: relative => path.join(root, relative)
    });
    if (!provider) throw new Error('msentity did not register its spectrum viewer provider.');
  } catch (error) {
    fs.rmSync(temporary, { recursive: true, force: true });
    throw error;
  }
  let disposed = false;
  return {
    resolveEmbeddedEditor(documentUri, panel) {
      let ready = false;
      let bootstrapping = true;
      const pending = [];
      const send = message => {
        if (disposed) return;
        if (!ready) pending.push(message);
        else panel.webview.postMessage(message);
      };
      const webview = {
        // The Workbench owns its document; upstream's standalone HTML/options
        // setters are intentionally absorbed by this transport facade.
        set options(_value) {},
        set html(_value) {},
        cspSource: panel.webview.cspSource,
        asWebviewUri: uri => panel.webview.asWebviewUri(uri),
        postMessage: message => {
          if (message.type === 'load-progress' && bootstrapping) return;
          if (message.type === 'backend-ready') {
            bootstrapping = false;
            backend.stdin.write(JSON.stringify({ type: 'remove-dataset', dataset_id: message.dataset.id }) + '\n');
            send({ ...message, dataset: null, total_rows: 0, file: '' });
          } else send(message);
        },
        onDidReceiveMessage: listener => panel.webview.onDidReceiveMessage(message => {
          if (message?.type === 'client-ready') {
            ready = true;
            for (const item of pending.splice(0)) send(item);
          } else if (message?.type !== 'ready' || message.datasetId) return listener(message);
        })
      };
      return provider.resolveCustomEditor({ uri: documentUri, fileType: 'msp' }, {
        webview, onDidDispose: listener => panel.onDidDispose(listener)
      });
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      if (backend?.exitCode === null) backend.kill();
      for (const item of disposables) if (item !== options.outputChannel) item.dispose();
      fs.rmSync(temporary, { recursive: true, force: true });
    }
  };
}

module.exports = { createHost };
