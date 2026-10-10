const path = require('path');
const vscode = require('vscode');
const upstream = require('./upstream');

// All viewer behavior belongs to msentity. This adapter only supplies the
// Workbench Python environment and isolates its message channel.
function createEmbeddedProvider(context, outputChannel) {
  const root = upstream.configure(context, vscode);
  const settings = vscode.workspace.getConfiguration('fredis');
  const provider = require('./host').createHost(root, vscode, {
    outputChannel,
    pythonPath: settings.get('pythonPath', 'python'),
    pageSize: settings.get('workbench.spectrumViewerPageSize', 20),
    spectrumFloatingWindow: false
  });
  return {
    resolveEmbeddedEditor(panel) {
      const documentUri = vscode.Uri.file(path.join(context.extensionPath, '.fredis-spectrum-viewer.msds'));
      return provider.resolveEmbeddedEditor(documentUri, {
        webview: {
          cspSource: panel.webview.cspSource,
          asWebviewUri: uri => panel.webview.asWebviewUri(uri),
          postMessage: message => panel.webview.postMessage({ type: 'spectrumViewer', message }),
          onDidReceiveMessage: listener => panel.webview.onDidReceiveMessage(envelope => {
            if (envelope?.type === 'spectrumViewer' && envelope.message && typeof envelope.message === 'object') {
              return listener(envelope.message);
            }
          })
        },
        onDidDispose: listener => panel.onDidDispose(listener)
      });
    },
    dispose: () => provider.dispose()
  };
}

module.exports = { createEmbeddedProvider };
