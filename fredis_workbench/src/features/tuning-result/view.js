function client(report){
 const el=id=>document.getElementById(id),svgNS='http://www.w3.org/2000/svg',vscode=acquireVsCodeApi(),saved=vscode.getState()||{};
 const metrics=[...new Set(report.trials.flatMap(t=>[...Object.keys(t.scoreSummaries||{}),...(t.validationHistory||[]).flatMap(row=>Object.keys(row).filter(key=>key.startsWith('validation')))]))].filter(key=>(key.includes('loss')||key.includes('cosine')&&!key.endsWith('_std'))&&!key.includes('@'));
 const label=key=>key.replace(/^validation_/, 'Validation ').replace(/_q25$/, ' Q1').replace(/_q75$/, ' Q3').replaceAll('_',' ');
 for(const key of metrics){const option=document.createElement('option');option.value=key;option.textContent=label(key);el('metric').append(option);}
 const preferred=['validation_cosine_similarity_mean','validation_spectrum_cosine_similarity','validation_cosine','validation_loss'].find(key=>metrics.includes(key))||metrics[0];
 const initial={metric:preferred,aggregation:'max',order:'execution',...saved};
 for(const key of ['metric','aggregation','order'])el(key).value=initial[key];
 if(!el('aggregation').value)el('aggregation').value='max';
 if(!el('metric').value&&metrics.length)el('metric').value=preferred;
 function summary(trial,metric){
  if(trial.scoreSummaries?.[metric])return trial.scoreSummaries[metric];
  const history=trial.validationHistory||[],values=history.map(row=>row[metric]).filter(Number.isFinite);if(!values.length)return {};
  return {max:Math.max(...values),min:Math.min(...values),last:history.at(-1)?.[metric]};
 }
 function score(trial,metric=el('metric').value,aggregation=el('aggregation').value){const value=summary(trial,metric)[aggregation];return trial.exitCode===0&&Number.isFinite(value)?value:null;}
 const columns=metrics.flatMap(metric=>(metric.includes('loss')?['min','max','last']:['max','last']).map(stat=>({metric,stat})));
 const header=el('trialHeaders');for(const {metric,stat}of columns){const th=document.createElement('th');th.textContent=label(metric)+' · '+stat;header.append(th);}const detailHeader=document.createElement('th');detailHeader.textContent='Details';header.append(detailHeader);
 function draw(){
  el('aggregation').querySelector('[value=min]').hidden=!el('metric').value.includes('loss');if(!el('metric').value.includes('loss')&&el('aggregation').value==='min')el('aggregation').value='max';
  const state=Object.fromEntries(['metric','aggregation','order'].map(key=>[key,el(key).value]));vscode.setState({...state,selected:saved.selected});
  const rows=report.trials.map(trial=>({trial,score:score(trial)})),ranked=rows.filter(r=>r.score!==null).sort((a,b)=>(state.metric.includes('loss')?a.score-b.score:b.score-a.score)||a.trial.trial-b.trial.trial),ranks=new Map(ranked.map((r,i)=>[r.trial.trial,i+1]));
  const shown=state.order==='score'?[...ranked,...rows.filter(r=>r.score===null)]:rows;
  const svg=el('chart');svg.replaceChildren();
  function node(tag,attrs,text){const item=document.createElementNS(svgNS,tag);for(const [key,value]of Object.entries(attrs))item.setAttribute(key,String(value));if(text!==undefined)item.textContent=text;svg.append(item);return item;}
  const width=920,height=420,left=100,right=30,top=24,bottom=65,pw=width-left-right,ph=height-top-bottom;
  node('text',{x:width/2,y:height-8,'text-anchor':'middle'},state.order==='score'?'Score rank (best first)':'Execution order (trial number)');
  node('text',{x:18,y:height/2,transform:`rotate(-90 18 ${height/2})`,'text-anchor':'middle'},state.metric+' · '+state.aggregation);
  if(!ranked.length){node('text',{x:width/2,y:height/2,'text-anchor':'middle'},'No completed trials with the selected validation metric.');}
  else{
   let min=Math.min(...ranked.map(r=>r.score)),max=Math.max(...ranked.map(r=>r.score));const padding=(max-min||Math.abs(max)||1)*0.1;min-=padding;max+=padding;
   const maxX=state.order==='score'?ranked.length:Math.max(report.totalTrials||1,...rows.map(r=>r.trial.trial)),x=v=>left+(maxX===1?pw/2:(v-1)/(maxX-1)*pw),y=v=>top+(max-v)/(max-min)*ph;
   for(let i=0;i<=5;i++){const value=min+(max-min)*i/5;node('line',{x1:left,x2:width-right,y1:y(value),y2:y(value),class:'gridline'});node('text',{x:left-12,y:y(value)+4,'text-anchor':'end'},Number(value.toPrecision(4)));}
   for(const value of [...new Set(Array.from({length:Math.min(6,maxX)},(_,i)=>maxX===1?1:Math.round(1+(maxX-1)*i/(Math.min(6,maxX)-1))))])node('text',{x:x(value),y:height-bottom+24,'text-anchor':'middle'},value);
   for(const row of ranked){const trial=row.trial,circle=node('circle',{cx:x(state.order==='score'?ranks.get(trial.trial):trial.trial),cy:y(row.score),r:7,class:'point',tabindex:0,role:'button','aria-label':`Trial ${trial.trial}: ${row.score}`});const title=document.createElementNS(svgNS,'title');title.textContent=`Trial ${trial.trial} · ${row.score}\n${JSON.stringify(trial.config?.parameters||trial.config)}`;circle.append(title);circle.onclick=()=>select(trial,row.score);circle.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(trial,row.score);}};}
  }
  const body=el('trials');body.replaceChildren();for(const row of shown){const tr=document.createElement('tr');for(const value of [row.trial.trial,row.trial.status,ranks.get(row.trial.trial)||'—',row.score===null?'—':row.score.toPrecision(6),...columns.map(({metric,stat})=>{const value=summary(row.trial,metric)[stat];return Number.isFinite(value)?value.toPrecision(6):'—';})]){const td=document.createElement('td');td.textContent=value;tr.append(td);}const td=document.createElement('td'),button=document.createElement('button');button.textContent='Details';button.onclick=()=>select(row.trial,row.score);td.append(button);tr.append(td);body.append(tr);}
  if(saved.selected){const selected=rows.find(r=>r.trial.trial===saved.selected);if(selected)select(selected.trial,selected.score);}
  el('best').textContent=ranked.length?`Best: trial ${ranked[0].trial.trial} · score ${ranked[0].score.toPrecision(6)} · ${report.status}`:`${report.status} · ${report.trials.length}/${report.totalTrials} trials`;
 }
 function select(trial,score){saved.selected=trial.trial;vscode.setState({...vscode.getState(),selected:trial.trial});el('detail').hidden=false;el('detailTitle').textContent=`Trial ${trial.trial} · ${trial.status} · score ${score??'unavailable'}`;el('config').textContent=JSON.stringify({configuration:trial.config,scoreSummaries:trial.scoreSummaries,validationHistory:trial.validationHistory,error:trial.error},null,2);el('openTrial').onclick=()=>vscode.postMessage({type:'openFile',path:trial.outputDir+'/result.training.pft'});}
 for(const key of ['metric','aggregation','order'])el(key).onchange=draw;
 el('scoresExport').onclick=()=>vscode.postMessage({type:'openFile',path:'tuning-scores.tsv'});
 el('refresh').onclick=()=>vscode.postMessage({type:'refresh'});draw();if(saved.selected){const trial=report.trials.find(t=>t.trial===saved.selected);if(trial)select(trial,score(trial));}
}
function html(report,css=''){
 const json=JSON.stringify(report).replace(/</g,'\\u003c');
 return `<!doctype html><html><head><meta charset="utf-8"><style>${css}body{padding:24px;color:var(--vscode-foreground);background:var(--vscode-editor-background)}.controls{display:flex;gap:18px;flex-wrap:wrap}.controls label{display:grid;gap:6px}select,button{color:inherit;background:var(--vscode-input-background);border:1px solid var(--vscode-panel-border,#888);padding:7px}svg{width:100%;max-height:520px;border:1px solid var(--vscode-panel-border,#888);margin:20px 0}svg text{fill:currentColor;font-size:13px}.gridline{stroke:currentColor;opacity:.15}.point{fill:var(--vscode-charts-blue,#529bff);cursor:pointer}.point:hover,.point:focus{stroke:currentColor;stroke-width:3}.score-table{overflow:auto;max-height:600px}table{width:100%;border-collapse:collapse}th{position:sticky;top:0;background:var(--vscode-editor-background);min-width:125px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--vscode-panel-border,#888)}pre{white-space:pre-wrap;overflow-wrap:anywhere}#detail{margin-top:24px}[hidden]{display:none!important}</style></head><body><h1>Fragment Tree Tuning Results</h1><p id="best"></p><div class="controls"><label>Validation metric<select id="metric"></select></label><label>Evaluation<select id="aggregation"><option value="min">Minimum</option><option value="max">Maximum</option><option value="last">Last epoch</option></select></label><label>Horizontal axis<select id="order"><option value="execution">Execution order</option><option value="score">Score rank</option></select></label><button id="refresh">Refresh results</button><button id="scoresExport">Open scores TSV</button></div><p>Q1, Q3, median and mean describe validation spectra within each epoch. Maximum selects the highest epoch value separately for each statistic; Last uses the final completed epoch. Cosine ranks higher values first; loss ranks lower values first. Every statistic is retained in the table, including failed-trial observations; failed and unfinished trials are excluded from ranking.</p><svg id="chart" viewBox="0 0 920 420" role="img" aria-label="Trial scores scatter plot"></svg><div class="score-table"><table><thead><tr id="trialHeaders"><th>Trial</th><th>Status</th><th>Rank</th><th>Selected score</th></tr></thead><tbody id="trials"></tbody></table></div><section id="detail" hidden><h2 id="detailTitle"></h2><button id="openTrial">Open training report</button><pre id="config"></pre></section><script>(${client.toString()})(${json});</script></body></html>`;
}
module.exports={html};
