const assert = require('assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { JSDOM } = require('jsdom');
const { html } = require('../src/features/training-report/view');
const { readRun } = require('../src/features/training-metrics/editor');

(async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'fredis-training-report-'));
  try {
    await fs.mkdir(path.join(directory, 'spectrum_validation'));
    await fs.writeFile(path.join(directory, 'metrics.tsv'), 'epoch\tglobal_step\ttrain_loss\tvalidation_loss\tcuda_peak_memory_mb\n1\t10\t3\t2\t512\n');
    await fs.writeFile(path.join(directory, 'metric_distributions.tsv'),
      'global_step\tepoch\tsplit\tmetric\tvalue\n' +
      '10\t1\tvalidation\tcosine_similarity@main_adduct:[M+H]+_q10\t0.2\n' +
      '10\t1\tvalidation\tcosine_similarity@main_adduct:[M+H]+_median\t0.3\n' +
      '10\t1\tvalidation\tcosine_similarity@main_adduct:[M+Na]+_median\t0.5\n' +
      '10\t1\tvalidation\tpeak_recall_top5_mean\t0.6\n');
    const spectrum = { main_adduct:'[M+H]+', collision_energy:20, generated_peaks:[{mz:100,intensity:1,precursor:true}], original_peaks:[{mz:100,intensity:1,precursor:true}] };
    await fs.writeFile(path.join(directory, 'spectrum_validation', 'epoch_1.json'), JSON.stringify({
      summary: { samples: 1 }, spectra: [spectrum], representatives: { cosine_similarity: { q10: { spectrum_index: 0, value: 1, target: 1 } } },
    }));
    const manifest = {
      schema: 'fredis.training-report', schemaVersion: 1, status: 'completed', project: 'p', run: 'r',
      completedEpochs: 1, globalStep: 10, updatedAt: 'now', latestValidation: 'spectrum_validation/epoch_1.json',
      datasets: { train: { samples: 2 }, validation: { samples: 1 } }, artifacts: { validationDirectory: 'spectrum_validation' },
    };
    const reportPath = path.join(directory, 'result.training.pft');
    await fs.writeFile(reportPath, JSON.stringify(manifest));

    // The click-to-open viewer must be a thin wrapper: no bespoke chart/spectra
    // rendering of its own, only the shared training-metrics UI, unhidden and auto-loaded.
    const page = html({ manifest, root: directory, reportPath }, '');
    assert.match(page, /Fragment Tree Training Report/);
    assert.match(page, /Representative validation spectra/);
    assert.match(page, /id="metricsApp">/);
    assert.equal((page.match(/Auto refresh/g) || []).length, 1, 'the report must expose only one auto-refresh control');
    assert.doesNotMatch(page, /id="metricsAuto"/, 'the embedded metrics auto-refresh must be hidden in the full report');
    for (const match of page.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)) new Function(match[1]);

    const posted = [];
    let webviewState = { reportScroll: { x: 11, y: 640 } }, scrollX = 0, scrollY = 0, restoredScroll = null;
    const dom = new JSDOM(page, {
      runScripts: 'dangerously',
      beforeParse(window) {
        Object.defineProperty(window, 'scrollX', { get: () => scrollX });
        Object.defineProperty(window, 'scrollY', { get: () => scrollY });
        window.scrollTo = (x, y) => { restoredScroll = { x, y }; };
        window.requestAnimationFrame = callback => { callback(); return 1; };
        window.acquireVsCodeApi = () => ({
          getState: () => webviewState,
          setState: value => { webviewState = value; },
          postMessage: message => posted.push(message),
        });
      },
    });
    try {
      assert.equal(posted.length, 1);
      assert.equal(posted[0].type, 'metricsLoad');
      assert.equal(posted[0].directory, reportPath);

      // Answer the auto-triggered load the same way training-metrics/editor.js's attach(panel) would.
      const data = await readRun(posted[0].directory);
      dom.window.dispatchEvent(new dom.window.MessageEvent('message', { data: { type: 'metricsData', request: posted[0].request, data } }));
      await new Promise(resolve => setTimeout(resolve, 0));

      assert.equal(dom.window.document.querySelectorAll('#representativeSpectra article').length, 1);
      assert.equal(dom.window.document.querySelector('#representativeSpectra select').value, 'q10');
      assert.match(dom.window.document.getElementById('representativeStatus').textContent, /spectra/);
      assert.deepEqual(restoredScroll, { x: 11, y: 640 }, 'page refresh must restore the previous viewport after charts render');
      assert.equal(webviewState.reportScroll, undefined, 'saved viewport must be consumed only once');

      scrollX = 7; scrollY = 930;
      dom.window.document.getElementById('reportRefresh').click();
      assert.equal(webviewState.reportScroll.x, 7, 'Refresh must save the horizontal viewport');
      assert.equal(webviewState.reportScroll.y, 930, 'Refresh must save the vertical viewport');
      assert.equal(posted.at(-1).type, 'refresh');

      dom.window.document.getElementById('metricsAddAll').click();
      const groupHeaders = [...dom.window.document.querySelectorAll('#metricsCharts > div')].filter(node => node.style.textTransform === 'uppercase');
      assert.ok(groupHeaders.some(node => node.textContent === 'Spectrum similarity'), 'expected a Spectrum similarity group header');
      assert.ok(groupHeaders.some(node => node.textContent === 'Peak assignment (top N)'), 'expected a Peak assignment (top N) group header');
      assert.ok(groupHeaders.some(node => node.textContent === 'Loss'), 'expected a Loss group header');
      assert.ok(groupHeaders.some(node => node.textContent === 'Performance & resources'), 'expected a Performance & resources group header');

      const facetCard = [...dom.window.document.querySelectorAll('#metricsCharts > section')].find(section => section.querySelector('strong')?.textContent === 'cosine_similarity @main_adduct');
      const swatchColors = [...facetCard.querySelectorAll('label line')].map(line => line.getAttribute('stroke'));
      assert.equal(new Set(swatchColors).size, swatchColors.length, 'faceted categories must not share the same line color');

      // Changing one card's own Display/precursor control must only replace that
      // card in place, not tear down and rebuild the whole grid (which used to
      // reset scroll position / jump the page to the top on every such change).
      const sectionByTitle = title => [...dom.window.document.querySelectorAll('#metricsCharts > section')].find(section => section.querySelector('strong')?.textContent === title);
      const lossCardBefore = sectionByTitle('loss'), memoryCardBefore = sectionByTitle('cuda_peak_memory_mb');
      const modeSelect = lossCardBefore.querySelector('select');
      modeSelect.value = 'distribution';
      modeSelect.dispatchEvent(new dom.window.Event('change'));
      const lossCardAfter = sectionByTitle('loss'), memoryCardAfter = sectionByTitle('cuda_peak_memory_mb');
      assert.notEqual(lossCardBefore, lossCardAfter, 'the edited card should be refreshed');
      assert.equal(memoryCardBefore, memoryCardAfter, 'unrelated cards must keep their DOM node identity so the page does not jump');
    } finally { dom.window.close(); }
    console.log('Training report wrapper: shared training-metrics UI, auto-load, grouping and representative spectra checks passed.');
  } finally { await fs.rm(directory, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
