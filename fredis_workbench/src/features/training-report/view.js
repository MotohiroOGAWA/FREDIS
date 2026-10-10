const trainingMetrics = require('../training-metrics/editor');

function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])); }

// The chart grid, per-card controls and representative-spectra section are entirely
// delegated to training-metrics/editor.js so the click-to-open viewer and the
// embedded Workbench "Metrics" page always run the exact same rendering script.
function html(data, commonCss) {
  const { manifest, root, reportPath } = data;
  const training = manifest.datasets?.train?.samples ?? '—', validation = manifest.datasets?.validation?.samples ?? '—';
  const directoryLiteral = JSON.stringify(reportPath).replace(/</g, '\\u003c');
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss}
  body{margin:0;background:var(--vscode-editor-background);color:var(--vscode-foreground)}.report-main{padding:24px 28px;max-width:none;width:100%}.report-toolbar{display:flex;gap:12px;align-items:center;padding:14px 24px;border-bottom:1px solid var(--border);background:var(--vscode-input-background)}.report-toolbar strong{margin-right:auto}.status-${escapeHtml(manifest.status)}{color:var(--accent)}</style></head><body><div class="report-toolbar"><strong>FREDIS Workbench <small>/ Training Result</small></strong><span class="status-${escapeHtml(manifest.status)}">${escapeHtml(manifest.status)}</span><label><input id="reportAuto" type="checkbox"> Auto refresh page (15s)</label><button id="reportRefresh">Refresh</button></div><main class="report-main"><header><div><span class="eyebrow">${escapeHtml(manifest.project)} / ${escapeHtml(manifest.run)}</span><h1>Fragment Tree Training Report</h1><p class="muted">${escapeHtml(root)} · updated ${escapeHtml(manifest.updatedAt)}</p></div></header><section class="cards"><article><b>${manifest.completedEpochs||0}</b><span>Completed epochs</span></article><article><b>${manifest.globalStep||0}</b><span>Global step</span></article><article><b>${training}</b><span>Train samples</span></article><article><b>${validation}</b><span>Validation samples</span></article></section>${manifest.error ? `<section><h2>Training failed</h2><pre>${escapeHtml(manifest.error)}</pre></section>` : ''}${trainingMetrics.html({ hidden: false, autoRefresh: false })}</main><script>const vscode=acquireVsCodeApi();${trainingMetrics.script()}
  const initialReportState=vscode.getState()||{};
  let reportScroll=initialReportState.reportScroll;
  const refreshReport=()=>{
    vscode.setState({...(vscode.getState()||{}),reportAuto:document.getElementById('reportAuto').checked,reportScroll:{x:window.scrollX,y:window.scrollY}});
    vscode.postMessage({type:'refresh'});
  };
  window.addEventListener('trainingMetricsRendered',()=>{
    if(!reportScroll)return;
    const position=reportScroll;reportScroll=null;
    requestAnimationFrame(()=>{
      window.scrollTo(position.x,position.y);
      const {reportScroll:ignored,...nextState}=vscode.getState()||{};
      vscode.setState(nextState);
    });
  });
  document.getElementById('metricsDirectory').value=${directoryLiteral};
  document.getElementById('metricsLoad').click();
  document.getElementById('reportAuto').checked=initialReportState.reportAuto ?? ${manifest.status === 'running'};
  document.getElementById('reportAuto').onchange=()=>vscode.setState({...(vscode.getState()||{}),reportAuto:document.getElementById('reportAuto').checked});
  document.getElementById('reportRefresh').onclick=refreshReport;
  setInterval(()=>{if(document.getElementById('reportAuto').checked)refreshReport();},15000);
  </script></body></html>`;
}

module.exports = { html };
