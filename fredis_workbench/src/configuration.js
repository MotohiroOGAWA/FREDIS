const vscode = require('vscode');

const SCHEMA = 'fredis.workbench.configuration';
const SCHEMA_VERSION = 1;

function configurationDocument(kind, config) {
  if (!kind || !config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('Invalid Workbench configuration.');
  }
  return { schema: SCHEMA, schemaVersion: SCHEMA_VERSION, kind, config };
}

function parseConfiguration(value, expectedKind) {
  if (!value || value.schema !== SCHEMA || value.schemaVersion !== SCHEMA_VERSION) {
    throw new Error('This is not a supported FREDIS Workbench configuration.');
  }
  if (value.kind !== expectedKind) {
    throw new Error(`Expected ${expectedKind} settings, but found ${value.kind || 'unknown'}.`);
  }
  if (!value.config || typeof value.config !== 'object' || Array.isArray(value.config)) {
    throw new Error('Workbench configuration payload is missing.');
  }
  return value.config;
}

function configurationSuffix(defaultName) {
  const lowerDefaultName = defaultName.toLowerCase();
  return lowerDefaultName.endsWith('.reaction-pattern-analysis.pft')
    ? '.reaction-pattern-analysis.pft' : lowerDefaultName.endsWith('.pft') ? '.pft' : '.json';
}

async function saveConfiguration(kind, config, defaultName) {
  const suffix = configurationSuffix(defaultName);
  const selected = await vscode.window.showSaveDialog({
    filters: { 'FREDIS Workbench configuration': suffix.endsWith('.pft') ? ['pft'] : ['json'] },
    defaultUri: vscode.Uri.file(defaultName)
  });
  if (!selected) return undefined;
  const target = selected.fsPath.toLowerCase().endsWith(suffix)
    ? selected : vscode.Uri.file(selected.fsPath + suffix);
  const content = JSON.stringify(configurationDocument(kind, config), null, 2) + '\n';
  await vscode.workspace.fs.writeFile(target, Buffer.from(content, 'utf8'));
  return target.fsPath;
}

async function loadConfiguration(kind, parser = parseConfiguration) {
  const selected = await vscode.window.showOpenDialog({
    canSelectMany: false, filters: { 'FREDIS Workbench configuration': ['pft', 'json'] }
  });
  if (!selected?.[0]) return undefined;
  const bytes = await vscode.workspace.fs.readFile(selected[0]);
  return {
    path: selected[0].fsPath,
    config: parser(JSON.parse(Buffer.from(bytes).toString('utf8')), kind)
  };
}

module.exports = {
  SCHEMA, SCHEMA_VERSION, configurationDocument, parseConfiguration,
  configurationSuffix, saveConfiguration, loadConfiguration
};
