const assert = require('assert');
const path = require('path');
const { spawnSync } = require('child_process');
const Module = require('module');
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
  return request === 'vscode' ? {} : originalLoad.call(this, request, parent, isMain);
};
const { buildArgs, shellDisplay, script } = require('../src/features/smarts-search/editor');
const cwd = path.resolve(__dirname, '../..');
const payload = { file: 'fredis/libs/msentity/tests/test_data/sample.msds', smarts: 'C', column: 'SMILES' };
const args = buildArgs(payload);
assert.strictEqual(shellDisplay('python', args),
  'python -m fredis.cli molecule smarts-search --input fredis/libs/msentity/tests/test_data/sample.msds --smarts C --smiles-column SMILES --include-compounds');
assert.strictEqual(shellDisplay('python', ['--smarts', 'c1ccccc1']), 'python --smarts c1ccccc1');
const direct = spawnSync('python', args, { cwd, encoding: 'utf8' });
assert.strictEqual(direct.status, 0, direct.stderr);
const copied = spawnSync('bash', ['-c', shellDisplay('python', args)], { cwd, encoding: 'utf8' });
assert.strictEqual(copied.status, 0, copied.stderr);
assert.deepStrictEqual(JSON.parse(copied.stdout), JSON.parse(direct.stdout));
assert.strictEqual(JSON.parse(direct.stdout).total, 6);
assert.strictEqual(JSON.parse(direct.stdout).compounds.length, 1);
assert.strictEqual(JSON.parse(direct.stdout).compounds[0].records, 6);
assert.strictEqual(JSON.parse(direct.stdout).compounds[0].matched, true);
const unmatched = spawnSync('python', buildArgs({ ...payload, smarts: '[#118]' }), { cwd, encoding: 'utf8' });
assert.strictEqual(unmatched.status, 0, unmatched.stderr);
assert.strictEqual(JSON.parse(unmatched.stdout).compounds[0].matched, false);
for (const change of [{ smarts: '[' }, { column: 'absent' }, { file: 'missing.msds' }]) {
  const failed = spawnSync('python', buildArgs({ ...payload, ...change }), { cwd, encoding: 'utf8' });
  assert.notStrictEqual(failed.status, 0);
  assert(failed.stderr.trim());
  assert.strictEqual(failed.stdout, '');
}
const special = ["a file's name.msds", '[$([#6]);!$(C=O)]', '`literal`'];
const roundTrip = spawnSync('bash', ['-c', shellDisplay('python', ['-c', 'import sys,json; print(json.dumps(sys.argv[1:]))', ...special])], { encoding: 'utf8' });
assert.strictEqual(roundTrip.status, 0, roundTrip.stderr);
assert.deepStrictEqual(JSON.parse(roundTrip.stdout), special);
new Function(script());
console.log('SMARTS CLI and copied command checks passed.');

const multiArgs = buildArgs({ ...payload, smarts: ['C', '[#118]'] });
assert.strictEqual(multiArgs.filter(arg => arg === '--smarts').length, 2);
const multi = spawnSync('python', multiArgs, { cwd, encoding: 'utf8' });
assert.strictEqual(multi.status, 0, multi.stderr);
const result = JSON.parse(multi.stdout);
assert.strictEqual(result.matched, 6);
assert.deepStrictEqual(result.patterns.map(p => p.matched), [6, 0]);
assert.deepStrictEqual(result.compounds[0].matchedPatterns, [0]);
const multiCopied = spawnSync('bash', ['-c', shellDisplay('python', multiArgs)], { cwd, encoding: 'utf8' });
assert.strictEqual(multiCopied.status, 0, multiCopied.stderr);
assert.deepStrictEqual(JSON.parse(multiCopied.stdout), result);
console.log('Multiple SMARTS CLI and copied command checks passed.');
