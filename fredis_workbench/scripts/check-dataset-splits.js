const assert = require('assert/strict');
const Module = require('module');
const load = Module._load;
Module._load = function(name, parent, main) {
  if (name === 'vscode') return {};
  return load.call(this, name, parent, main);
};
const { normalizeConfig, buildArgs, workbenchHtml } = require('../src/extension');
Module._load = load;
for (const fraction of [0, 0.1, 0.25]) {
  const saved = normalizeConfig(JSON.parse(JSON.stringify({input:'input.msds',outputDir:'out',
    validation_ratio:0.1,test_ratio:fraction,validation_seed:7})));
  const args = buildArgs(saved);
  assert.equal(args[args.indexOf('--test-ratio')+1],String(fraction));
  assert.equal(args[args.indexOf('--validation-seed')+1],'7');
  const html = workbenchHtml(saved,{});
  assert(html.includes('name="testRatio"'));
  for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
  const trainArgs = require('../src/extension').buildTrainingArgs({datasetDir:'out',testRatio:fraction});
  assert(!trainArgs.includes('--test-ratio'));
  assert(!trainArgs.includes('out/test_structures'));
}
console.log('Dataset split settings round trip, CLI arguments, generated UI scripts and training inputs passed.');
