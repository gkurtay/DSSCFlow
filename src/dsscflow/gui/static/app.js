const $ = (id) => document.getElementById(id);
let lastToken = null;
let lastResult = null;

function setMessage(text, kind='') {
  const el = $('message'); el.textContent = text; el.className = 'message ' + kind;
}

async function filePayload(input, multiple=false) {
  if (multiple) {
    const out=[];
    for (const file of input.files) out.push({name:file.name, text:await file.text()});
    return out;
  }
  const file=input.files[0];
  return file ? {name:file.name, text:await file.text()} : null;
}

function selectedWidths() {
  return [...document.querySelectorAll('.widthCheck:checked')].map(x => Number(x.value));
}

async function buildPayload() {
  const mode=$('mode').value;
  const payload={
    mode,
    widths:selectedWidths(),
    active_sigma:Number($('activeSigma').value),
    active_source:$('activeSource').value || null,
    concentration_molar:Number($('concentration').value),
    path_length_cm:Number($('pathLength').value),
    selected_ids:$('selectedIds').value.split(',').map(x=>x.trim()).filter(Boolean),
  };
  if (!payload.widths.length) throw new Error('Select at least one broadening width.');
  if (mode==='upload') {
    payload.transitions=await filePayload($('transitionsFile'));
    payload.sources=await filePayload($('sourcesFiles'), true);
    payload.descriptors=await filePayload($('descriptorsFile'));
    payload.ifct=await filePayload($('ifctFile'));
    payload.tdm=await filePayload($('tdmFile'));
    if (!payload.transitions) throw new Error('Select a TD-DFT transition CSV.');
    if (!payload.sources.length) throw new Error('Select at least one illumination-source CSV.');
  }
  return payload;
}

function esc(x) { return String(x ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmt(x,n=4) { const v=Number(x); return Number.isFinite(v) ? v.toFixed(n) : '—'; }

function tableHTML(rows, columns, labels={}) {
  if (!rows || !rows.length) return '<p class="hint">No data available.</p>';
  let h='<table><thead><tr>' + columns.map(c=>`<th>${esc(labels[c]||c)}</th>`).join('') + '</tr></thead><tbody>';
  for (const row of rows) {
    h+='<tr>'+columns.map(c=>`<td>${typeof row[c]==='number' ? esc(fmt(row[c], c==='rank'?0:4)) : esc(row[c])}</td>`).join('')+'</tr>';
  }
  return h+'</tbody></table>';
}

function updateSourceOptions(sources, active) {
  const sel=$('activeSource');
  const old=active || sel.value;
  sel.innerHTML='';
  for (const s of sources) {
    const op=document.createElement('option'); op.value=s; op.textContent=s; if(s===old) op.selected=true; sel.appendChild(op);
  }
}

function renderMetrics(meta) {
  const cards=$('metricGrid').querySelectorAll('.metric strong');
  cards[0].textContent=meta.dyes; cards[1].textContent=meta.transitions; cards[2].textContent=meta.sources.length; cards[3].textContent=Number(meta.active_sigma).toFixed(2)+' eV';
}

function renderChecks(checks) {
  const box=$('checksBox');
  if (!checks || !checks.available) { box.innerHTML='<div class="check-card">Publication-regression cards are shown for the bundled DSSC16 demonstration dataset.</div>'; return; }
  const keys=[
    ['D08_top_all_sources','D08 ranks first under all sources'],
    ['BTD_effect_positive_all_sources','BTD PCF effect is positive under all sources'],
    ['robust_pareto_matches_D08_D14_D16','Robust Pareto set is D08, D14, D16'],
  ];
  box.innerHTML=keys.filter(([k])=>k in checks).map(([k,label])=>`<div class="check-card ${checks[k]?'pass':'fail'}"><strong>${checks[k]?'PASS':'CHECK'}</strong><br>${esc(label)}</div>`).join('');
}

const palette=['#195b9b','#e87722','#2b8c6b','#8c5fb2','#bd3d3a','#5d7895','#9b7d22','#2a9db0'];
function svgLineChart(svg, series) {
  svg.innerHTML=''; const W=1000,H=440,L=70,R=25,T=25,B=55; const x0=350,x1=750,y0=0,y1=1;
  const sx=x=>L+(x-x0)/(x1-x0)*(W-L-R), sy=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  let html=`<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="#fff" stroke="#dfe6ef"/>`;
  for(let x=350;x<=750;x+=50){html+=`<line x1="${sx(x)}" y1="${T}" x2="${sx(x)}" y2="${H-B}" stroke="#edf0f4"/><text x="${sx(x)}" y="${H-B+25}" text-anchor="middle" font-size="13" fill="#5d6675">${x}</text>`;}
  for(let y=0;y<=1.0001;y+=0.2){html+=`<line x1="${L}" y1="${sy(y)}" x2="${W-R}" y2="${sy(y)}" stroke="#edf0f4"/><text x="${L-12}" y="${sy(y)+4}" text-anchor="end" font-size="13" fill="#5d6675">${y.toFixed(1)}</text>`;}
  let li=0;
  for(const [name,pts] of Object.entries(series||{})){
    const c=palette[li++%palette.length]; const valid=pts.filter(p=>p[0]>=x0&&p[0]<=x1);
    if(!valid.length) continue;
    const path=valid.map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+','+sy(p[1]).toFixed(1)).join(' ');
    html+=`<path d="${path}" fill="none" stroke="${c}" stroke-width="2.4"/>`;
    html+=`<rect x="${W-R-160}" y="${T+li*22-17}" width="14" height="3" fill="${c}"/><text x="${W-R-140}" y="${T+li*22-12}" font-size="13" fill="#24364d">${esc(name)}</text>`;
  }
  html+=`<text x="${(L+W-R)/2}" y="${H-10}" text-anchor="middle" font-size="14" fill="#24364d">Wavelength / nm</text>`;
  html+=`<text x="18" y="${(T+H-B)/2}" transform="rotate(-90 18 ${(T+H-B)/2})" text-anchor="middle" font-size="14" fill="#24364d">Normalized absorption</text>`;
  svg.innerHTML=html;
}

function renderRanking(rows) {
  $('rankingPreview').innerHTML=tableHTML(rows.slice(0,6),['rank','ID','PCF','PCC_nm','PCB_nm'],{PCF:'PCF',PCC_nm:'PCC / nm',PCB_nm:'PCB / nm'});
  $('rankingTable').innerHTML=tableHTML(rows,['rank','ID','PCF','PCC_nm','PCB_nm'],{PCF:'PCF',PCC_nm:'PCC / nm',PCB_nm:'PCB / nm'});
  const max=Math.max(...rows.map(r=>Number(r.PCF)||0),1e-12);
  $('rankingBars').innerHTML=rows.map(r=>`<div class="bar-row"><b>${esc(r.ID)}</b><div class="bar-track"><div class="bar-fill" style="width:${100*Number(r.PCF)/max}%"></div></div><span>${fmt(r.PCF,4)}</span></div>`).join('');
}

function renderEffects(rows) {
  $('effectTable').innerHTML=tableHTML(rows,['term','effect','mean_plus','mean_minus'],{term:'Factor',effect:'Effect',mean_plus:'Mean (+)',mean_minus:'Mean (−)'});
  if(!rows.length){$('effectBars').innerHTML='';return;}
  const max=Math.max(...rows.map(r=>Math.abs(Number(r.effect))),1e-12);
  $('effectBars').innerHTML=rows.map(r=>{
    const v=Number(r.effect), width=45*Math.abs(v)/max; const left=v>=0?50:50-width;
    return `<div class="effect-row"><b>${esc(r.term)}</b><div class="effect-track"><div class="effect-fill ${v<0?'neg':''}" style="left:${left}%;width:${width}%"></div></div><span>${v>=0?'+':''}${fmt(v,4)}</span></div>`;
  }).join('');
}

function renderPareto(rows) {
  const svg=$('paretoChart');
  if(!rows){svg.innerHTML='<text x="500" y="245" text-anchor="middle" font-size="20" fill="#667085">Upload descriptors + IFCT + fragment-TDM tables to enable Pareto analysis.</text>'; $('paretoTable').innerHTML=''; return;}
  const W=1000,H=500,L=80,R=30,T=30,B=70;
  const xs=rows.map(r=>Number(r.CT_percent)), ys=rows.map(r=>Number(r.optical_value));
  let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys); const xp=(xmax-xmin)*.08||1,yp=(ymax-ymin)*.1||.01; xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;
  const sx=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R), sy=y=>T+(1-(y-ymin)/(ymax-ymin))*(H-T-B);
  let html=`<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="#fff" stroke="#dfe6ef"/>`;
  for(const r of rows){const x=sx(Number(r.CT_percent)),y=sy(Number(r.optical_value));const front=!!r.pareto_front;const c=front?'#d34a3a':'#3979b7';const rad=front?9:6;html+=`<circle cx="${x}" cy="${y}" r="${rad}" fill="${c}" opacity=".88"/><text x="${x+9}" y="${y-9}" font-size="12" font-weight="600" fill="#23344e">${esc(r.ID)}</text>`;}
  html+=`<text x="${(L+W-R)/2}" y="${H-18}" text-anchor="middle" font-size="14">IFCT CT / %</text><text x="20" y="${(T+H-B)/2}" transform="rotate(-90 20 ${(T+H-B)/2})" text-anchor="middle" font-size="14">PCF</text><circle cx="${W-180}" cy="25" r="7" fill="#d34a3a"/><text x="${W-168}" y="30" font-size="12">exact Pareto</text>`;
  svg.innerHTML=html;
  $('paretoTable').innerHTML=tableHTML(rows,['ID','optical_value','CT_percent','lambda_total_eV','pareto_front','epsilon_pareto_front'],{optical_value:'PCF',CT_percent:'IFCT CT / %',lambda_total_eV:'λtotal / eV',pareto_front:'Pareto',epsilon_pareto_front:'ε-Pareto'});
}

function renderResult(r) {
  lastResult=r; lastToken=r.token; $('exportBtn').disabled=false;
  $('versionBadge').textContent='v'+r.version;
  updateSourceOptions(r.meta.sources,r.meta.active_source);
  renderMetrics(r.meta); renderChecks(r.checks); renderRanking(r.ranking); renderEffects(r.factorial_main_effects);
  $('photonSub').textContent=`${r.meta.active_source}; σ = ${Number(r.meta.active_sigma).toFixed(2)} eV`;
  svgLineChart($('spectraChart'),r.spectra);
  $('lambdaTable').innerHTML=tableHTML(r.lambda_max,['ID','lambda_max_nm','epsilon_max'],{lambda_max_nm:'λmax / nm',epsilon_max:'εmax / M⁻¹ cm⁻¹'});
  renderPareto(r.pareto);
  $('robustnessTable').innerHTML=tableHTML(r.robustness,['ID','mean_PCF','min_PCF','max_PCF','cv_PCF','mean_rank','rank_min','rank_max'],{mean_PCF:'Mean PCF',min_PCF:'Min PCF',max_PCF:'Max PCF',cv_PCF:'CV',mean_rank:'Mean rank',rank_min:'Best rank',rank_max:'Worst rank'});
}

async function analyze() {
  try {
    $('analyzeBtn').disabled=true; setMessage('Running DSSCFlow analysis…');
    const payload=await buildPayload();
    const res=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await res.json();
    if(!res.ok) throw new Error(data.error||'Analysis failed.');
    renderResult(data); setMessage('Analysis completed.','ok');
  } catch(e) { setMessage(e.message||String(e),'error'); }
  finally { $('analyzeBtn').disabled=false; }
}

$('mode').addEventListener('change',()=>{$('uploadBlock').classList.toggle('hidden',$('mode').value!=='upload');});
$('analyzeBtn').addEventListener('click',analyze);
$('exportBtn').addEventListener('click',()=>{if(lastToken) window.location=`/api/export?token=${encodeURIComponent(lastToken)}`;});
$('sourcesFiles').addEventListener('change',()=>{
  const names=[...$('sourcesFiles').files].map(f=>f.name.replace(/_200_800nm_10000pt\.csv$/i,'').replace(/\.csv$/i,''));
  if(names.length) updateSourceOptions(names,names[0]);
});
for(const b of document.querySelectorAll('.tab')) b.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); document.querySelectorAll('.tabpane').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); $(b.dataset.tab).classList.add('active');
});
fetch('/api/health').then(r=>r.json()).then(x=>{$('healthStatus').textContent='local server · v'+x.version;$('versionBadge').textContent='v'+x.version;}).catch(()=>{});
analyze();
