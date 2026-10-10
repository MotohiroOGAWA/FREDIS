const fs = require('fs');
const upstream = require('./upstream');

function html() {
  return '<div id="spectrumViewerApp" data-app-panel="spectrumViewer" class="spectrum-viewer-page" hidden><iframe id="spectrumViewerFrame" title="msentity Spectrum Viewer"></iframe></div>';
}

function css() {
  return 'body:has(#spectrumViewerApp:not([hidden]))>main{padding:60px 0 0}body:has(#spectrumViewerApp:not([hidden])) #workbenchError:empty{display:none}.spectrum-viewer-page{height:calc(100vh - 60px);min-height:480px}.spectrum-viewer-page iframe{display:block;width:100%;height:100%;border:0;background:var(--vscode-editor-background)}';
}

function script() {
  // Keep upstream CSS, DOM and JavaScript together in their own document.
  // Only the VS Code API transport is supplied by the host.
  const assets = JSON.stringify({
    css: fs.readFileSync(upstream.file('media', 'viewer.css'), 'utf8'),
    js: fs.readFileSync(upstream.file('media', 'viewer.js'), 'utf8')
  }).replace(/</g, '\\u003c');
  return `(() => {
    const frame = document.getElementById('spectrumViewerFrame');
    const assets = ${assets};
    const nonce = document.currentScript.nonce;
    window.addEventListener('message', event => {
      if (event.source === frame.contentWindow) return;
      if (event.data?.type === 'spectrumViewer') {
        frame.contentWindow?.postMessage(event.data.message, '*');
      }
    });
    const theme = getComputedStyle(document.documentElement);
    const variables = Array.from(theme).filter(name => name.startsWith('--vscode-'))
      .map(name => name + ':' + theme.getPropertyValue(name) + ';').join('');
    const syncTheme = () => {
      const root = frame.contentDocument?.documentElement;
      if (!root) return;
      const current = getComputedStyle(document.documentElement);
      for (const name of Array.from(current)) {
        if (name.startsWith('--vscode-')) root.style.setProperty(name, current.getPropertyValue(name));
      }
      frame.contentDocument.body.className = document.body.className;
    };
    frame.addEventListener('load', syncTheme);
    new MutationObserver(syncTheme).observe(document.documentElement, { attributes: true });
    new MutationObserver(syncTheme).observe(document.body, { attributes: true });
    // Explorer transfers in remote windows may contain only ResourceURLs or
    // CodeEditors. Use Workbench's shared decoder before the upstream
    // document handler, without modifying the submodule viewer.
    window.fredisInstallSpectrumDrop = doc => {
      if (!window.installFileDropZone) return;
      window.installFileDropGuard(doc);
      window.installFileDropZone(doc.body, {
        kind: 'file', multiple: true,
        onDrop: paths => {
          const supported = paths.filter(name => /\\.(msds|msp|mgf|tsv|csv)$/i.test(name));
          if (!supported.length) throw new Error('Drop an MSDS, MSP, MGF, TSV, or CSV file.');
          window.fredisMsentityBridge.postMessage({ type: 'drop-datasets', paths: supported });
        },
        onError: message => window.fredisMsentityBridge.postMessage({ type: 'edit-error-notification', message })
      });
    };
    const bootstrap = 'window.acquireVsCodeApi=()=>parent.fredisMsentityBridge;parent.fredisInstallSpectrumDrop(document);';
    frame.srcdoc = '<!doctype html><html><head><meta charset="UTF-8"><style>:root{' + variables + '}' + assets.css + '</style></head><body><div id="app" data-filename="FREDIS Spectrum Viewer"></div><script nonce="' + nonce + '">' + bootstrap + assets.js + ';parent.fredisMsentityBridge.postMessage({type:"client-ready"});<\\/script></body></html>';
  })();`;
}

module.exports = { html, css, script, configure: upstream.configure };
