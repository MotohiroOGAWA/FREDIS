// Exercise list separation, pagination and filtering without a VS Code host.
const assert = require('assert');
const vm = require('vm');
const Module = require('module');
const load = Module._load;
Module._load = function(r, p, i) { return r === 'vscode' ? {} : load.call(this, r, p, i); };
const { script } = require('../src/features/smarts-search/editor');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.value = ''; }
  append(...children) { for (const child of children) { if (typeof child === 'object') child.parent = this; } this.children.push(...children); }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  replaceChildren(...children) { this.children = children; }
  setAttribute() {}
  add(tag) { const e = new Element(tag); this.append(e); return e; }
  insertRow() { return this.add('tr'); }
  insertCell() { return this.add('td'); }
  createTHead() { return this.add('thead'); }
  createTBody() { return this.add('tbody'); }
  querySelectorAll() { return []; }
}
const elements = new Map();
let receive;
vm.runInNewContext(script(), {
  document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element('div')); return elements.get(id); },
    createElement: tag => new Element(tag)
  },
  window: { addEventListener(type, callback) { receive = callback; } },
  vscode: { postMessage() {} }
});
const compounds = Array.from({ length: 101 }, (_, i) => ({ smiles: 'SMILES-' + i, records: 1, matched: true, matchedPatterns: i < 2 ? [0] : [1] }));
compounds.push({ smiles: 'unmatched', records: 2, matched: false });
receive({ data: { type: 'smartsSearchResult', ok: true, result: {
  patterns: [{smarts: 'O', matched: 2, valid: 103, percent: 200/103, uniqueMatched: 2, unique: 102, uniquePercent: 200/102}, {smarts: 'N', matched: 99, valid: 103, percent: 9900/103, uniqueMatched: 99, unique: 102, uniquePercent: 9900/102}],
  compounds, total: 103, valid: 103, matched: 101, unique: 102, uniqueMatched: 101,
  percent: 10100 / 103, uniquePercent: 10100 / 102, missing: 0, invalid: 0
} } });
const [, , scopeLabel, lists] = elements.get('smartsResult').children;
const [detected, undetected] = lists.children;
assert.strictEqual(detected.children[0].textContent, 'Detected compounds (101)');
assert.strictEqual(undetected.children[0].textContent, 'Not detected compounds (1)');
const [title, filter, table, previous, page, next] = detected.children;
const body = table.children[1];
assert.strictEqual(body.children.length, 50);
assert(previous.disabled);
next.onclick(); next.onclick();
assert.strictEqual(body.children.length, 1);
assert(next.disabled);
filter.value = 'SMILES-100'; filter.oninput();
assert.strictEqual(body.children.length, 1);
assert.strictEqual(body.children[0].children[0].textContent, 'SMILES-100');
filter.value = 'absent'; filter.oninput();
assert.strictEqual(body.children.length, 0);
assert(previous.disabled && next.disabled);
console.log('SMARTS compound list checks passed.');

const scope = scopeLabel.children[0];
scope.value = '0'; scope.onchange();
assert.strictEqual(lists.children[0].children[0].textContent, 'Detected compounds (2)');
assert.strictEqual(lists.children[1].children[0].textContent, 'Not detected compounds (100)');
scope.value = '1'; scope.onchange();
assert.strictEqual(lists.children[0].children[0].textContent, 'Detected compounds (99)');
assert.strictEqual(elements.get('smartsQueries').children.length, 1);
elements.get('smartsAddQuery').onclick();
assert.strictEqual(elements.get('smartsQueries').children.length, 2);
console.log('Per-pattern compound lists and query addition checks passed.');

const queries = elements.get('smartsQueries');
queries.children[1].children[1].onclick();
assert.strictEqual(queries.children.length, 1);
assert.strictEqual(queries.children[0].children[1].disabled, true);
console.log('Query removal and minimum query checks passed.');
