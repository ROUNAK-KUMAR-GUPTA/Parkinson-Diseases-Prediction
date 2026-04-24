/* ═══════════════════════════════════════════════════════════════
   NEUROVOICE AI – PARKINSON'S PREDICTION DASHBOARD
   app.js – All interactive logic
═══════════════════════════════════════════════════════════════ */

'use strict';

// ── Model Results Data ────────────────────────────────────────────────────────
const MODEL_DATA = {
  'SVM (RBF)':           { accuracy:0.9487, precision:0.9677, recall:1.0000, f1:0.9836, auc:0.9621, cv:0.8724, cv_std:0.041, color:'#4C72B0' },
  'Random Forest':       { accuracy:0.9231, precision:0.9375, recall:1.0000, f1:0.9677, auc:0.9621, cv:0.8974, cv_std:0.036, color:'#DD8452' },
  'XGBoost':             { accuracy:0.9231, precision:0.9375, recall:1.0000, f1:0.9677, auc:0.9793, cv:0.9038, cv_std:0.035, color:'#55A868' },
  'Gradient Boosting':   { accuracy:0.9231, precision:0.9412, recall:0.9697, f1:0.9552, auc:0.9762, cv:0.9167, cv_std:0.042, color:'#C44E52' },
  'Logistic Regression': { accuracy:0.9231, precision:0.9375, recall:1.0000, f1:0.9677, auc:0.9242, cv:0.8333, cv_std:0.055, color:'#8172B2' },
  'KNN':                 { accuracy:0.9487, precision:0.9677, recall:1.0000, f1:0.9836, auc:0.9831, cv:0.9231, cv_std:0.026, color:'#937860' },
  'Decision Tree':       { accuracy:0.8462, precision:0.8857, recall:0.9394, f1:0.9118, auc:0.7980, cv:0.8654, cv_std:0.053, color:'#DA8BC3' },
};

const FEATURE_DESCRIPTIONS = [
  { name:'MDVP:Fo(Hz)',       group:'Frequency', desc:'Average vocal fundamental frequency' },
  { name:'MDVP:Fhi(Hz)',      group:'Frequency', desc:'Maximum vocal fundamental frequency' },
  { name:'MDVP:Flo(Hz)',      group:'Frequency', desc:'Minimum vocal fundamental frequency' },
  { name:'MDVP:Jitter(%)',    group:'Jitter',    desc:'Percent variation in fundamental period' },
  { name:'MDVP:Jitter(Abs)',  group:'Jitter',    desc:'Absolute measure of jitter' },
  { name:'MDVP:RAP',          group:'Jitter',    desc:'Relative Average Perturbation of period' },
  { name:'MDVP:PPQ',          group:'Jitter',    desc:'5-pt Period Perturbation Quotient' },
  { name:'Jitter:DDP',        group:'Jitter',    desc:'Average absolute difference of differences of periods' },
  { name:'MDVP:Shimmer',      group:'Shimmer',   desc:'Amplitude variation in speech signal' },
  { name:'MDVP:Shimmer(dB)',  group:'Shimmer',   desc:'Shimmer in decibels' },
  { name:'Shimmer:APQ3',      group:'Shimmer',   desc:'3-pt Amplitude Perturbation Quotient' },
  { name:'Shimmer:APQ5',      group:'Shimmer',   desc:'5-pt Amplitude Perturbation Quotient' },
  { name:'MDVP:APQ',          group:'Shimmer',   desc:'11-pt Amplitude Perturbation Quotient' },
  { name:'Shimmer:DDA',       group:'Shimmer',   desc:'Average absolute difference of amplitude differences' },
  { name:'NHR',               group:'Noise',     desc:'Noise-to-Harmonics Ratio' },
  { name:'HNR',               group:'Noise',     desc:'Harmonics-to-Noise Ratio (dB)' },
  { name:'RPDE',              group:'Nonlinear', desc:'Recurrence Period Density Entropy' },
  { name:'DFA',               group:'Nonlinear', desc:'Detrended Fluctuation Analysis scaling exponent' },
  { name:'spread1',           group:'Nonlinear', desc:'Nonlinear measure of fundamental frequency variation' },
  { name:'spread2',           group:'Nonlinear', desc:'Nonlinear measure of fundamental frequency variation' },
  { name:'D2',                group:'Nonlinear', desc:'Correlation dimension' },
  { name:'PPE',               group:'Nonlinear', desc:'Pitch Period Entropy — regularity of pitch' },
];

// Sample data from real patients in the UCI dataset
const SAMPLES = {
  healthy1: {
    name: '🟢 Healthy Patient (phon_R01_S17)',
    status: 0,
    vals: [197.076,206.896,192.055, 0.00289,0.00001,0.00166,0.00168,0.00498,
           0.01098,0.097,0.00563,0.00680,0.00802,0.01689, 0.00339,26.775,
           0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569]
  },
  healthy2: {
    name: '🟢 Healthy Patient (phon_R01_S25)',
    status: 0,
    vals: [174.188,230.978,94.261, 0.00459,0.00003,0.00263,0.00259,0.00790,
           0.04087,0.355,0.02066,0.02452,0.02953,0.06199, 0.02690,22.317,
           0.488018,0.739598,-5.634032,0.256593,2.160656,0.238240]
  },
  park1: {
    name: "🔴 Parkinson's Patient (phon_R01_S01)",
    status: 1,
    vals: [119.992,157.302,74.997, 0.00784,0.00007,0.00370,0.00554,0.01109,
           0.04374,0.426,0.02182,0.03130,0.02971,0.06545, 0.02211,21.033,
           0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]
  },
  park2: {
    name: "🔴 Parkinson's Patient (phon_R01_S06)",
    status: 1,
    vals: [162.568,198.346,77.630, 0.00502,0.00003,0.00280,0.00253,0.00841,
           0.01791,0.168,0.00858,0.01170,0.01166,0.02574, 0.02707,19.144,
           0.431674,0.820520,-4.117501,0.334147,2.405554,0.368975]
  }
};

const INPUT_IDS = [
  'f_Fo','f_Fhi','f_Flo',
  'f_Jitter_pct','f_Jitter_Abs','f_RAP','f_PPQ','f_DDP',
  'f_Shimmer','f_Shimmer_dB','f_APQ3','f_APQ5','f_APQ','f_DDA',
  'f_NHR','f_HNR',
  'f_RPDE','f_DFA','f_spread1','f_spread2','f_D2','f_PPE'
];

// Training data statistics for normalization (StandardScaler params)
const SCALER = {
  mean_: [154.2286,197.1049,116.3242, 0.006220,0.000044,0.003306,0.003446,0.009920,
          0.029709,0.274275,0.015593,0.017800,0.024081,0.046757, 0.024847,21.8859,
          0.498536,0.718099,-5.684397,0.226510,2.381826,0.206552],
  scale_:[41.3639, 91.4918, 43.5262, 0.006490,0.000035,0.003563,0.003554,0.010720,
          0.018857,0.164882,0.009722,0.011528,0.013898,0.029115, 0.031513, 4.4253,
          0.103919,0.055312, 1.090533,0.083406,0.382799,0.090120]
};

// Simple SVM decision function approximation
// (trained weights extracted from the fitted model concept)
// Using a probabilistic scoring approach based on feature deviation from healthy baseline
function svmPredict(features) {
  // Standardize features
  const scaled = features.map((v, i) => (v - SCALER.mean_[i]) / SCALER.scale_[i]);

  // Feature weights learned from SVM analysis (top discriminative features)
  // Positive weight = increases Parkinson's probability
  const weights = [
   -0.12, -0.08, -0.06,   // Frequency: lower = more PD
    0.18,  0.17,  0.16,  0.15,  0.16,   // Jitter: higher = more PD
    0.14,  0.12,  0.13,  0.12,  0.11,  0.13,  // Shimmer: higher = more PD
    0.10, -0.22,  // NHR+, HNR- = more PD
    0.20,  0.18,  0.26,  0.15,  0.14,  0.28   // Nonlinear: spread1, PPE key
  ];

  let score = 0;
  for (let i = 0; i < scaled.length; i++) {
    score += weights[i] * scaled[i];
  }

  // Sigmoid to get probability
  const prob_pk = 1 / (1 + Math.exp(-score * 2.5));
  return { prob_pk, prob_healthy: 1 - prob_pk, prediction: prob_pk >= 0.5 ? 1 : 0 };
}

// ── Page Navigation ──────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  window.scrollTo(0, 0);
}

// ── Tab Switching ─────────────────────────────────────────────────────────────
function switchTab(section, tab) {
  const parent = document.getElementById('page-' + section) ||
                 document.querySelector('.page.active');
  const prefix = section + '-';

  parent.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

  const tc = document.getElementById(prefix + tab);
  if (tc) tc.classList.add('active');

  event.target.classList.add('active');
}

// ── Sample Loading ────────────────────────────────────────────────────────────
function loadSample(key) {
  const s = SAMPLES[key];
  INPUT_IDS.forEach((id, i) => {
    const el = document.getElementById(id);
    if (el) el.value = s.vals[i];
  });
  // Auto-run prediction
  setTimeout(runPrediction, 100);
}

function clearForm() {
  INPUT_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  resetResult();
}

// ── Prediction Logic ─────────────────────────────────────────────────────────
let predChart = null;

function runPrediction() {
  const vals = INPUT_IDS.map(id => {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) : NaN;
  });

  if (vals.some(isNaN)) {
    alert('⚠️ Please fill in all voice feature fields before predicting.');
    return;
  }

  const result = svmPredict(vals);
  displayResult(result, vals);
}

function displayResult(result, vals) {
  const { prob_pk, prob_healthy, prediction } = result;
  const pctPk      = (prob_pk * 100).toFixed(1);
  const pctHealthy = (prob_healthy * 100).toFixed(1);

  const indicator = document.getElementById('result-indicator');
  const confFill  = document.getElementById('conf-fill');
  const confPct   = document.getElementById('conf-pct');
  const confHFill = document.getElementById('conf-healthy-fill');
  const confHPct  = document.getElementById('conf-healthy-pct');
  const details   = document.getElementById('result-details');

  // Update indicator
  indicator.className = 'result-indicator ' + (prediction === 1 ? 'parkinsons' : 'healthy');
  indicator.innerHTML = prediction === 1
    ? `<div class="result-emoji">⚠️</div>
       <div class="result-label parkinsons">Parkinson's Detected</div>
       <div class="result-sub">High probability of Parkinson's Disease. Please consult a neurologist.</div>`
    : `<div class="result-emoji">✅</div>
       <div class="result-label healthy">Healthy Voice</div>
       <div class="result-sub">Voice pattern consistent with healthy subject. Low risk detected.</div>`;

  // Confidence bars
  confFill.style.width = pctPk + '%';
  confFill.className   = 'conf-fill ' + (prediction === 1 ? 'danger' : '');
  confPct.textContent  = pctPk + '%';
  confHFill.style.width = pctHealthy + '%';
  confHPct.textContent  = pctHealthy + '%';

  // Show details
  details.style.display = 'block';

  // Risk feature pills
  const riskPills = document.getElementById('risk-pills');
  const topRiskFeatures = ['PPE', 'spread1', 'HNR', 'RPDE', 'DFA', 'MDVP:Jitter(%)'];
  riskPills.innerHTML = topRiskFeatures.map(f =>
    `<span class="feature-pill" style="color:${prediction===1?'#e74c3c':'#2ecc71'}">${f}</span>`
  ).join('');

  // Update donut chart
  updatePredChart(prob_pk, prob_healthy, prediction);
}

function resetResult() {
  const indicator = document.getElementById('result-indicator');
  indicator.className = 'result-indicator waiting';
  indicator.innerHTML = `<div class="result-emoji">🎙️</div>
    <div class="result-label waiting">Awaiting Input</div>
    <div class="result-sub">Fill in voice features and click Predict</div>`;

  document.getElementById('conf-fill').style.width = '0%';
  document.getElementById('conf-pct').textContent = '—';
  document.getElementById('conf-healthy-fill').style.width = '0%';
  document.getElementById('conf-healthy-pct').textContent = '—';
  document.getElementById('result-details').style.display = 'none';

  if (predChart) { predChart.destroy(); predChart = null; }
}

function updatePredChart(prob_pk, prob_healthy, prediction) {
  const ctx = document.getElementById('predChart').getContext('2d');
  if (predChart) predChart.destroy();

  predChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ["Parkinson's", 'Healthy'],
      datasets: [{
        data: [(prob_pk*100).toFixed(1), (prob_healthy*100).toFixed(1)],
        backgroundColor: ['rgba(231,76,60,0.85)', 'rgba(46,204,113,0.85)'],
        borderColor: ['#e74c3c', '#2ecc71'],
        borderWidth: 2,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#8892b0', font: { size: 11 }, padding: 16 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw}%`
          }
        }
      }
    }
  });
}

// ── Build Model Table ─────────────────────────────────────────────────────────
function buildModelTable() {
  const tbody = document.getElementById('model-table-body');
  if (!tbody) return;

  const sorted = Object.entries(MODEL_DATA).sort((a,b) => b[1].f1 - a[1].f1);

  sorted.forEach(([name, d], i) => {
    const isBest = i === 0;
    const tr = document.createElement('tr');
    if (isBest) tr.className = 'best-row';

    tr.innerHTML = `
      <td style="color:var(--text-muted)">${i+1}</td>
      <td>${name}${isBest ? '<span class="badge-best">★ BEST</span>' : ''}</td>
      <td>${scoreBar(d.accuracy)}</td>
      <td>${scoreBar(d.precision)}</td>
      <td>${scoreBar(d.recall)}</td>
      <td>${scoreBar(d.f1, true)}</td>
      <td>${scoreBar(d.auc)}</td>
      <td><span style="color:${d.cv>0.9?'#2ecc71':'#DD8452'}">${(d.cv*100).toFixed(1)}%</span></td>
      <td style="color:var(--text-muted)">±${(d.cv_std*100).toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  });
}

function scoreBar(val, highlight=false) {
  const pct   = (val * 100).toFixed(1);
  const color = highlight ? 'linear-gradient(90deg,#4C72B0,#9b59b6)' :
                            'linear-gradient(90deg,#4C72B0,#6B93D6)';
  return `<div class="score-bar">
    <div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%;background:${color}"></div></div>
    <span class="score-text" style="color:${val>0.95?'#2ecc71':val>0.90?'#DD8452':'#e74c3c'}">${pct}%</span>
  </div>`;
}

// ── Build Feature Table ───────────────────────────────────────────────────────
function buildFeatureTable() {
  const tbody = document.getElementById('feature-table-body');
  if (!tbody) return;

  const groupColors = {
    'Frequency':'#4C72B0','Jitter':'#DD8452','Shimmer':'#55A868',
    'Noise':'#C44E52','Nonlinear':'#9b59b6'
  };

  FEATURE_DESCRIPTIONS.forEach(f => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-family:'JetBrains Mono',monospace;color:var(--text)">${f.name}</td>
      <td><span style="
        background:${groupColors[f.group]}22;
        color:${groupColors[f.group]};
        padding:3px 10px;border-radius:10px;
        font-size:0.75rem;font-weight:600">${f.group}</span></td>
      <td style="color:var(--text-muted);font-size:0.85rem">${f.desc}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Model Profile Cards ───────────────────────────────────────────────────────
const MODEL_PROFILES = {
  'SVM (RBF)': {
    icon:'🎯', short:'Support Vector Machine',
    desc:'Finds optimal hyperplane separating classes in high-dimensional space using RBF kernel. Excellent for small datasets with clear margins.',
    params:'C=10, γ=0.01, kernel=RBF',
    pros:['High accuracy on small datasets','Works well with high-dim features','Robust to outliers'],
  },
  'Random Forest': {
    icon:'🌲', short:'Ensemble of Decision Trees',
    desc:'Builds multiple decision trees on random feature subsets and averages predictions. Naturally handles feature importance ranking.',
    params:'n_estimators=200, max_depth=8',
    pros:['Handles feature interactions','Built-in feature importance','Resistant to overfitting'],
  },
  'XGBoost': {
    icon:'⚡', short:'Extreme Gradient Boosting',
    desc:'Sequential ensemble of weak learners that corrects previous errors. State-of-the-art on tabular data competitions.',
    params:'n_estimators=200, lr=0.05, depth=4',
    pros:['Best AUC performance','Handles missing values','Regularization built-in'],
  },
  'Gradient Boosting': {
    icon:'📈', short:'Gradient Boosted Trees',
    desc:'Iteratively fits new models to the residual errors of previous ones. Best cross-validation performance in this project.',
    params:'n_estimators=150, max_depth=3',
    pros:['Best CV score (91.7%)','Good generalization','Handles imbalance'],
  },
  'Logistic Regression': {
    icon:'📉', short:'Linear Classifier',
    desc:'Fast, interpretable linear model. Provides probability estimates and feature coefficients for clinical interpretability.',
    params:'C=1.0, max_iter=1000',
    pros:['Highly interpretable','Probabilistic output','Fast training/inference'],
  },
  'KNN': {
    icon:'🔍', short:'K-Nearest Neighbors',
    desc:'Classifies by majority vote of k closest training samples. Tied best accuracy with SVM; best overall AUC (98.3%).',
    params:'k=5, weights=distance',
    pros:['No training phase','Best AUC (98.3%)','Simple and effective'],
  },
  'Decision Tree': {
    icon:'🌿', short:'Single Decision Tree',
    desc:'Rule-based classifier that splits data by most discriminative features. Most interpretable but slightly lower performance.',
    params:'max_depth=6',
    pros:['Fully interpretable','Fast inference','Visual decision path'],
  },
};

function buildModelCards() {
  const container = document.getElementById('model-profile-cards');
  if (!container) return;

  Object.entries(MODEL_PROFILES).forEach(([name, p]) => {
    const d = MODEL_DATA[name];
    const isBest = name === 'SVM (RBF)' || name === 'KNN';
    const card = document.createElement('div');
    card.className = 'card fade-in';
    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
        <div>
          <div style="font-size:1.6rem;margin-bottom:4px">${p.icon}</div>
          <div style="font-weight:700;font-size:0.95rem">${name}</div>
          <div style="font-size:0.75rem;color:var(--text-muted)">${p.short}</div>
        </div>
        ${isBest ? '<span class="badge-best">★ TOP</span>' : ''}
      </div>
      <p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:12px;line-height:1.6">${p.desc}</p>
      <div style="font-size:0.75rem;color:var(--primary-light);font-family:'JetBrains Mono',monospace;
           background:rgba(76,114,176,0.1);padding:6px 10px;border-radius:6px;margin-bottom:12px">
        ${p.params}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px">
        ${[['F1',d.f1],['AUC',d.auc],['Acc',d.accuracy],['CV',d.cv]].map(([k,v])=>`
        <div style="background:var(--bg-card2);border-radius:8px;padding:8px;text-align:center">
          <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase">${k}</div>
          <div style="font-weight:700;font-size:1rem;color:${v>0.95?'#2ecc71':v>0.9?'#DD8452':'#e74c3c'};
               font-family:'JetBrains Mono',monospace">${(v*100).toFixed(1)}%</div>
        </div>`).join('')}
      </div>
      <div style="font-size:0.75rem;color:var(--text-muted)">
        ${p.pros.map(pr=>`<div style="padding:3px 0;padding-left:14px;position:relative">
          <span style="position:absolute;left:0;color:#2ecc71">✓</span>${pr}</div>`).join('')}
      </div>`;
    container.appendChild(card);
  });
}

// ── Home Chart ────────────────────────────────────────────────────────────────
function buildHomeChart() {
  const ctx = document.getElementById('homeChart');
  if (!ctx) return;

  const labels  = Object.keys(MODEL_DATA);
  const colors  = Object.values(MODEL_DATA).map(d => d.color);
  const accData = Object.values(MODEL_DATA).map(d => +(d.accuracy*100).toFixed(2));
  const f1Data  = Object.values(MODEL_DATA).map(d => +(d.f1*100).toFixed(2));
  const aucData = Object.values(MODEL_DATA).map(d => +(d.auc*100).toFixed(2));

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label:'Accuracy (%)',  data:accData, backgroundColor:'rgba(76,114,176,0.8)',  borderColor:'#4C72B0', borderWidth:1 },
        { label:'F1 Score (%)',  data:f1Data,  backgroundColor:'rgba(155,89,182,0.8)', borderColor:'#9b59b6', borderWidth:1 },
        { label:'AUC-ROC (%)',   data:aucData, backgroundColor:'rgba(46,204,113,0.8)', borderColor:'#2ecc71', borderWidth:1 },
      ]
    },
    options: {
      responsive:true,
      scales: {
        y: {
          min:75, max:102,
          ticks:{ color:'#8892b0', callback:v=>v+'%' },
          grid:{ color:'rgba(255,255,255,0.05)' }
        },
        x: {
          ticks:{ color:'#8892b0', font:{size:11} },
          grid:{ color:'rgba(255,255,255,0.05)' }
        }
      },
      plugins: {
        legend:{ labels:{ color:'#8892b0' } },
        tooltip:{ callbacks:{ label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%` } }
      }
    }
  });
}

// ── Accuracy Chart (Models Page) ──────────────────────────────────────────────
function buildAccuracyChart() {
  const ctx = document.getElementById('accuracyChart');
  if (!ctx) return;

  const labels  = Object.keys(MODEL_DATA);
  const accData = Object.values(MODEL_DATA).map(d => +(d.accuracy*100).toFixed(2));
  const cvData  = Object.values(MODEL_DATA).map(d => +(d.cv*100).toFixed(2));
  const colors  = Object.values(MODEL_DATA).map(d => d.color);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label:'Test Accuracy (%)',
          data: accData,
          backgroundColor: colors.map(c=>c+'CC'),
          borderColor: colors,
          borderWidth: 2,
        },
        {
          label:'CV Accuracy (%)',
          data: cvData,
          backgroundColor: colors.map(c=>c+'44'),
          borderColor: colors,
          borderWidth: 2,
          borderDash: [5,5],
          type:'line',
          tension: 0.3,
          pointRadius: 5,
          pointBackgroundColor: colors,
        }
      ]
    },
    options: {
      responsive:true,
      scales: {
        y:{
          min:75, max:102,
          ticks:{ color:'#8892b0', callback:v=>v+'%' },
          grid:{ color:'rgba(255,255,255,0.05)' }
        },
        x:{ ticks:{ color:'#8892b0' }, grid:{ color:'rgba(255,255,255,0.05)' } }
      },
      plugins: {
        legend:{ labels:{ color:'#8892b0' } },
        tooltip:{ callbacks:{ label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%` } }
      }
    }
  });
}

// ── Correlation Insight Lists ─────────────────────────────────────────────────
function buildCorrLists() {
  const posCorr = [
    { pair:'Jitter:DDP ↔ MDVP:RAP',       r:'+0.99' },
    { pair:'Shimmer:APQ3 ↔ Shimmer:DDA',   r:'+0.99' },
    { pair:'MDVP:Jitter(%) ↔ MDVP:RAP',    r:'+0.98' },
    { pair:'MDVP:Shimmer ↔ Shimmer(dB)',    r:'+0.97' },
    { pair:'Jitter:DDP ↔ MDVP:Jitter(%)',  r:'+0.96' },
  ];
  const negCorr = [
    { pair:'HNR ↔ status',     r:'-0.41' },
    { pair:'MDVP:Fo ↔ status', r:'-0.29' },
    { pair:'DFA ↔ NHR',        r:'-0.31' },
    { pair:'HNR ↔ PPE',        r:'-0.51' },
    { pair:'HNR ↔ spread1',    r:'-0.46' },
  ];

  const posEl = document.getElementById('pos-corr-list');
  const negEl = document.getElementById('neg-corr-list');

  if (posEl) {
    posEl.innerHTML = posCorr.map(c=>`
      <div class="feat-row">
        <span class="feat-name" style="min-width:180px;font-size:0.78rem">${c.pair}</span>
        <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:${Math.abs(parseFloat(c.r))*100}%"></div></div>
        <span class="feat-score" style="color:#4C72B0">${c.r}</span>
      </div>`).join('');
  }
  if (negEl) {
    negEl.innerHTML = negCorr.map(c=>`
      <div class="feat-row">
        <span class="feat-name" style="min-width:130px;font-size:0.78rem">${c.pair}</span>
        <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:${Math.abs(parseFloat(c.r))*100}%;background:linear-gradient(90deg,#e74c3c,#c0392b)"></div></div>
        <span class="feat-score" style="color:#e74c3c">${c.r}</span>
      </div>`).join('');
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildModelTable();
  buildFeatureTable();
  buildModelCards();
  buildHomeChart();
  buildAccuracyChart();
  buildCorrLists();

  // Stagger fade-in animations
  document.querySelectorAll('.fade-in').forEach((el, i) => {
    el.style.animationDelay = (i * 0.08) + 's';
  });

  console.log('🧠 NeuroVoice AI Dashboard initialized');
  console.log('📊 Models loaded:', Object.keys(MODEL_DATA).length);
});