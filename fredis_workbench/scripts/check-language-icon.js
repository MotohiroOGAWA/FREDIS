const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const [language] = pkg.contributes.languages;
assert.ok(language, 'package.json must contribute a language for the FREDIS file icon');
assert.equal(language.id, 'fredis-workbench-file');

// VS Code only shows a contributed language icon when the active icon theme
// has no opinion for that file at all; virtually every theme already claims
// bare .json, so it always wins over our icon there. Every extension we
// register must therefore end in something no theme recognizes (.pft /
// .fredis-result), never .json -- this guards against silently regressing
// back to a compound .json suffix that would make the icon stop appearing.
for (const extension of language.extensions) {
  assert.ok(!extension.toLowerCase().endsWith('.json'), `${extension} ends in .json and will not reliably get the icon`);
}

for (const key of ['light', 'dark']) {
  const resolved = path.join(root, language.icon[key]);
  assert.ok(fs.existsSync(resolved), `language icon.${key} points to a missing file: ${language.icon[key]}`);
}

// Every customEditors selector pattern should be covered, so a
// newly added viewer extension is never left without the icon.
const missing = [];
for (const editor of pkg.contributes.customEditors) {
  for (const { filenamePattern } of editor.selector) {
    const extension = filenamePattern.replace(/^\*/, '');
    if (!language.extensions.includes(extension)) missing.push(`${editor.viewType}: ${filenamePattern}`);
  }
}
assert.deepEqual(missing, [], `customEditors patterns without a matching language-icon extension: ${missing.join(', ')}`);

console.log('Language icon: package.json contribution, no .json extensions, icon files present and custom-editor coverage checks passed.');
