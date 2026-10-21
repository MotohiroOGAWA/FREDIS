const vscode = require('vscode');
const formats = require('../file-formats');
const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function html(file) {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>body{font-family:var(--vscode-font-family);color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);padding:24px}main{max-width:1000px;margin:auto}p{line-height:1.6}code{overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid var(--vscode-panel-border,#888)}button,input{padding:8px;color:inherit;background:var(--vscode-input-background);border:1px solid var(--vscode-panel-border,#888)}button{cursor:pointer}input{width:100%;box-sizing:border-box}small{display:block;opacity:.75;margin-top:5px}[hidden]{display:none!important}</style></head><body><main><h1>FREDIS File Formats</h1><p>This filename has no supported format association.<br><code>${esc(file)}</code></p><p>Choose the format that matches this file's contents. Opening a format does not rename or convert the file.</p><button id="text">Open as Text</button><p><input id="search" type="search" placeholder="Search formats or extensions" aria-label="Search file formats"></p><table><thead><tr><th>Format</th><th>Extension</th><th></th></tr></thead><tbody>${formats.formats.map(item=>`<tr><td>${esc(item.label)}<small>${esc(item.description)}</small></td><td><code>${esc(item.suffix)}</code></td><td><button data-format="${esc(item.id)}">Open with this format</button></td></tr>`).join('')}</tbody></table><p id="status" role="status"></p></main><script>const vscode=acquireVsCodeApi();document.getElementById('search').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q));};document.getElementById('text').onclick=()=>vscode.postMessage({type:'format/text'});document.querySelectorAll('[data-format]').forEach(button=>button.onclick=()=>vscode.postMessage({type:'format/open',id:button.dataset.format}));window.addEventListener('message',e=>{if(e.data.type==='format/error')document.getElementById('status').textContent=e.data.message;});</script></body></html>`;
}
function attach(panel, uri, openResult) {
  panel.title = 'FREDIS File Formats';
  panel.webview.options = {enableScripts:true};
  panel.webview.html = html(uri.fsPath);
  const listener = panel.webview.onDidReceiveMessage(async message=>{
    try {
      if (message.type==='format/text') await vscode.commands.executeCommand('vscode.openWith',uri,'default');
      if (message.type==='format/open') {
        const entry = formats.format(message.id);
        if (entry.viewType===formats.fallback.viewType) {
          listener.dispose();
          panel.title=entry.label;
          await openResult();
        } else await vscode.commands.executeCommand('vscode.openWith',uri,entry.viewType);
      }
    } catch(error) { await panel.webview.postMessage({type:'format/error',message:error.message}); }
  });
  return listener;
}
module.exports = { html, attach };
