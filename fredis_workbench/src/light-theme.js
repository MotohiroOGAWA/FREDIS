// White surfaces for native VS Code light themes, respecting Workbench overrides.
const css=`html:not([data-workbench-theme=dark])>body.vscode-light:not([data-result-theme=dark]),html:not([data-workbench-theme=dark])>body.vscode-high-contrast-light:not([data-result-theme=dark]){--vscode-editor-background:#fff;--vscode-input-background:#fff;--vscode-sideBar-background:#fff;--vscode-textBlockQuote-background:#f7f7f7;--surface:#fff;--panel:#fff;background:#fff}`;
module.exports={css};
