const registry = require('./file-formats.generated.json');
const path = require('path');
function format(id) {
  const entry = registry.formats.find(item => item.id === id);
  if (!entry) throw new Error(`Unknown FREDIS format: ${id}`);
  return entry;
}
function match(file) {
  const name = path.basename(file).toLowerCase();
  return [...registry.formats].sort((a,b)=>b.suffix.length-a.suffix.length)
    .find(entry => name.endsWith(entry.suffix) || name === entry.suffix.slice(1));
}
function contributions(source = registry) {
  if (source.schemaVersion !== 1 || !Array.isArray(source.formats)) throw new Error('Unsupported file format registry.');
  const ids = new Set(), suffixes = new Set();
  for (const item of source.formats) {
    if (!item.id || ids.has(item.id) || !/^\.[a-z0-9.-]+$/.test(item.suffix) || suffixes.has(item.suffix) || !item.viewType || !item.editorName || !item.label || !item.description) throw new Error('Invalid or duplicate file format: '+item.id);
    ids.add(item.id); suffixes.add(item.suffix);
  }
  const editors = new Map();
  for (const item of source.formats) {
    if (!editors.has(item.viewType)) editors.set(item.viewType, {viewType:item.viewType,displayName:item.editorName,selector:[],priority:'default'});
    editors.get(item.viewType).selector.push({filenamePattern:'*'+item.suffix});
  }
  editors.get(source.fallback.viewType).selector.unshift({filenamePattern:'*'+source.fallback.suffix});
  return {customEditors:[...editors.values()],extensions:[source.fallback.suffix,...new Set(source.formats.map(item=>item.suffix))]};
}
module.exports = { ...registry, format, match, contributions };
