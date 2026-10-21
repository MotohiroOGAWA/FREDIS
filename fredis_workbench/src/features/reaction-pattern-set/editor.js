const fileFormats = require('../../file-formats');
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const reactionPatternUi = require('./ui');
const { commonCss, formCss } = require('./styles');
const reactionPatternVisualEditor = require('./visual-editor');
const reactionPatternPreview = require('./reaction-preview');
const pathDrop = require('../../webview-path-drop');
const host = require('./host');

const VIEW_TYPE = fileFormats.format('reaction-set').viewType;
const PATTERN_VIEW_TYPE = fileFormats.format('reaction').viewType;
// .pft (not .json) so VS Code's contributed language icon actually shows:
// every icon theme already claims plain .json, which always wins over a
// contributed language icon, but no theme claims .pft.
const FILE_SUFFIX = fileFormats.format('reaction-set').suffix;

function emptyDocument() {
  return {
    reaction_pattern_set: {
      name: 'new_reaction_pattern_set',
      patterns: []
    }
  };
}

function normalizeDocument(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('The JSON root must be an object.');
  const set = value.reaction_pattern_set;
  if (!set || typeof set !== 'object' || Array.isArray(set)) throw new Error('The JSON must contain a reaction_pattern_set object.');
  return {
    reaction_pattern_set: {
      name: String(set.name || ''),
      patterns: Array.isArray(set.patterns) ? set.patterns.map(pattern => ({
        name: String(pattern && pattern.name || ''),
        reactant_smarts: String(pattern && pattern.reactant_smarts || ''),
        products: Array.isArray(pattern && pattern.products) ? pattern.products.map(product => ({
          name: String(product && product.name || ''),
          smarts: String(product && product.smarts || '')
        })) : []
      })) : []
    }
  };
}

function normalizePattern(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || 'reaction_pattern_set' in value) throw new Error('The JSON must contain a single reaction pattern object.');
  return normalizeDocument({reaction_pattern_set:{name:'',patterns:[value]}}).reaction_pattern_set.patterns[0];
}

class ReactionPatternSetEditorProvider {
  constructor(context, singlePattern = false) { this.context = context; this.singlePattern = singlePattern; }
  resolveCustomTextEditor(document, panel) {
    const context = this.context, singlePattern = this.singlePattern;
    panel.webview.options = { enableScripts: true };
    panel.webview.html = editorHtml(singlePattern);
    pathDrop.attach(panel);
    let applyingEdit = false;

    const currentValue = () => {
      const source = JSON.parse(document.getText());
      return singlePattern ? {reaction_pattern_set:{name:'',patterns:[normalizePattern(source)]}} : normalizeDocument(source);
    };
    const sendDocument = () => {
      try {
        panel.webview.postMessage({ type: 'reactionPatternSet', value: currentValue(), path: document.uri.fsPath });
      } catch (error) {
        panel.webview.postMessage({ type: 'reactionPatternError', message: error.message });
      }
    };

    const changeSubscription = vscode.workspace.onDidChangeTextDocument(event => {
      if (event.document.uri.toString() === document.uri.toString() && !applyingEdit) sendDocument();
    });
    panel.onDidDispose(() => changeSubscription.dispose());

    async function writeDocument(rawValue) {
      const value = normalizeDocument(rawValue);
      if (singlePattern && value.reaction_pattern_set.patterns.length !== 1) throw new Error('A reaction pattern file must contain exactly one pattern.');
      const savedValue = singlePattern ? value.reaction_pattern_set.patterns[0] : value;
      const text = `${JSON.stringify(savedValue, null, 2)}\n`;
      if (text === document.getText()) return;
      const fullRange = new vscode.Range(document.positionAt(0), document.positionAt(document.getText().length));
      const edit = new vscode.WorkspaceEdit();
      edit.replace(document.uri, fullRange, text);
      applyingEdit = true;
      const applied = await vscode.workspace.applyEdit(edit);
      applyingEdit = false;
      if (!applied) throw new Error('The document could not be updated.');
    }

    panel.webview.onDidReceiveMessage(async message => {
      try {
        if (message.type === 'ready') { sendDocument(); return; }
        if (message.type === 'update') { await writeDocument(message.value); return; }
        if (message.type === 'copyText') { await vscode.env.clipboard.writeText(String(message.text || '')); return; }
        if (message.type === 'chemistry') {
          const result = await host.runChemistryBackend(context, message.command, message.payload, panel);
          panel.webview.postMessage({ type: 'chemistryResult', requestId: message.requestId, result });
          return;
        }
        if (message.type === 'selectElements') {
          const elements = await host.showElementPicker(message.elements);
          if (elements) panel.webview.postMessage({ type: 'elementSelectionResult', requestId: message.requestId, elements });
          return;
        }
        if (message.type === 'loadReactionPatternSet' || message.type === 'loadReactionPattern') {
          const source = await host.readReactionPatternImport(context, message);
          if (!source) return;
          if (message.type === 'loadReactionPatternSet') {
            panel.webview.postMessage({ type: 'reactionPatternSet', value: normalizeDocument(source.value), path: source.path });
          } else {
            panel.webview.postMessage({ type: 'reactionPatternLoaded', path: source.path, pattern: normalizePattern(source.value), index: message.index });
          }
          return;
        }
        if (message.type === 'saveReactionPatternSet') {
          const value = normalizeDocument(message.value);
          const defaultUri = vscode.Uri.file(host.reactionPatternSetSavePath(value.reaction_pattern_set.name, document.uri.fsPath, FILE_SUFFIX));
          const selected = await vscode.window.showSaveDialog({ filters: { 'FREDIS Reaction Pattern Set': ['pft'] }, defaultUri });
          if (selected) {
            const target = vscode.Uri.file(host.ensureFileSuffix(selected.fsPath, FILE_SUFFIX));
            await fs.promises.writeFile(target.fsPath, `${JSON.stringify(value, null, 2)}\n`);
            panel.webview.postMessage({ type: 'reactionPatternSetSaved', path: target.fsPath });
            vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
          }
          return;
        }
        if (message.type === 'saveReactionPattern') {
          const pattern = normalizePattern(message.pattern);
          const selected = await vscode.window.showSaveDialog({ filters: { 'FREDIS Reaction Pattern': ['pft'] }, defaultUri: vscode.Uri.file(document.uri.fsPath) });
          if (selected) {
            const target = vscode.Uri.file(host.ensureFileSuffix(selected.fsPath, fileFormats.format('reaction').suffix));
            await fs.promises.writeFile(target.fsPath, `${JSON.stringify(pattern, null, 2)}\n`);
            vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
          }
          return;
        }
      } catch (error) {
        panel.webview.postMessage({ type: 'reactionPatternError', message: String(error.message || error) });
      }
    });
  }
}

function register(context) {
  for (const [viewType, singlePattern] of [[VIEW_TYPE, false], [PATTERN_VIEW_TYPE, true]]) {
    context.subscriptions.push(
      vscode.window.registerCustomEditorProvider(viewType, new ReactionPatternSetEditorProvider(context, singlePattern), {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false
      })
    );
  }
}

// Reuses the exact same markup, styling and client script as the Workbench's
// Reaction Pattern Set tab (ui.js/visual-editor.js/reaction-preview.js), so this
// standalone file editor can never fall out of sync with the Workbench's editor.
function editorHtml(singlePattern = false) {
  return `<!doctype html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${commonCss()}${formCss()}${pathDrop.css()}</style></head><body><main>
  ${reactionPatternUi.html(singlePattern)}
  <script>
  const vscode=acquireVsCodeApi();
  ${reactionPatternUi.state()}
  const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  ${pathDrop.script({host:true})}
  ${reactionPatternUi.script(singlePattern)}
  ${reactionPatternVisualEditor.script()}
  ${reactionPatternPreview.script()}
  document.getElementById('reactionPatternApp').hidden=false;
  ${singlePattern
    ? "let reactionPatternSyncTimer;onReactionPatternChange=()=>{clearTimeout(reactionPatternSyncTimer);reactionPatternSyncTimer=setTimeout(()=>vscode.postMessage({type:'update',value:reactionPatternModel}),200);};"
    : "onReactionPatternChange=()=>{};"}
  vscode.postMessage({type:'ready'});
  </script>
  </main></body></html>`;
}

module.exports = { register, normalizeDocument, normalizePattern, emptyDocument, FILE_SUFFIX, PATTERN_VIEW_TYPE, VIEW_TYPE };
