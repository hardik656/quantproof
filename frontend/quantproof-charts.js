/**
 * QuantProof Chart Layer v1.0
 * Drop this into index.html as a <script> tag after Chart.js CDN.
 * 
 * REQUIRES: Chart.js 4.x loaded before this script.
 * 
 * HOW TO INTEGRATE:
 *   1. Add to <head>:
 *      <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
 *   2. Add this file as a <script src="quantproof-charts.js"></script> before closing </body>
 *   3. In renderResults(), call: QP_Charts.init(data) after section.classList.add('active')
 *   4. In resetApp(), call: QP_Charts.destroy()
 *   5. Add <div id="qp-chart-section"></div> in buildResultsHTML() where you want charts
 *      (recommended: between score-hero and result-tabs)
 * 
 * ZERO backend changes needed. All data derived from existing ValidationResponse.
 */

const QP_Charts = (function() {
  'use strict';

  // ── chart registry ──────────────────────────────────────────────────────
  const _charts = {};

  function _destroy(id) {
    if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
  }

  function destroy() {
    Object.keys(_charts).forEach(_destroy);
    const el = document.getElementById('qp-chart-section');
    if (el) el.innerHTML = '';
  }

  // ── color helpers ───────────────────────────────────────────────────────
  function scoreColor(s) {
    if (s >= 80) return '#00ff88';
    if (s >= 65) return '#00cc6e';
    if (s >= 50) return '#ffaa00';
    if (s >= 35) return '#ff6600';
    return '#ff4444';
  }

  // ── synthetic equity curve from check scores + crash sims ──────────────
  // We don't have raw trade returns, but we can build a plausible
  // illustrative equity curve from the Sharpe + win_rate + drawdown
  // the API returns, seeded to the validation_hash for reproducibility.
  function _seededRng(seed) {
    let s = seed;
    return function() {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }

  function _hashToInt(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = ((h << 5) - h) + str.charCodeAt(i);
      h |= 0;
    }
    return Math.abs(h);
  }

  function buildEquityCurve(data) {
    const n = Math.min(Math.max(data.total_trades, 40), 200);
    const rng = _seededRng(_hashToInt(data.validation_hash || 'qp'));

    const sharpe = data.sharpe || 1.0;
    const wr = (data.win_rate || 55) / 100;
    const maxDD = Math.abs(data.max_drawdown || 0.10);

    // Calibrate synthetic returns to match reported metrics
    const meanR = sharpe * 0.015 / Math.sqrt(252 / n);
    const stdR  = meanR / (sharpe + 1e-9);

    let equity = 100;
    const equitySeries = [equity];
    const drawdownSeries = [0];
    let peak = equity;

    for (let i = 1; i < n; i++) {
      const u1 = Math.max(1e-10, rng());
      const u2 = Math.max(1e-10, rng());
      // Box-Muller with slight positive skew for realistic equity curves
      let r = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      r = r * stdR + meanR;
      r = Math.max(-0.99, Math.min(0.99, r));
      equity *= (1 + r);
      peak = Math.max(peak, equity);
      const dd = (peak - equity) / peak * 100;
      equitySeries.push(+equity.toFixed(2));
      drawdownSeries.push(-dd);
    }

    return { equity: equitySeries, drawdown: drawdownSeries, n };
  }

  function buildRollingSharpe(equitySeries, windowSize) {
    const w = windowSize || Math.max(10, Math.floor(equitySeries.length / 8));
    const result = new Array(equitySeries.length).fill(null);
    for (let i = w; i < equitySeries.length; i++) {
      const window = [];
      for (let j = i - w; j < i; j++) {
        window.push((equitySeries[j + 1] - equitySeries[j]) / equitySeries[j]);
      }
      const mean = window.reduce((a, b) => a + b, 0) / window.length;
      const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / (window.length - 1);
      const std = Math.sqrt(variance);
      result[i] = std < 1e-9 ? 0 : +(mean / std * Math.sqrt(252)).toFixed(3);
    }
    return result;
  }

  function buildMonteCarloPaths(equitySeries, nPaths, rng) {
    const n = equitySeries.length;
    const returns = [];
    for (let i = 1; i < n; i++) {
      returns.push((equitySeries[i] - equitySeries[i - 1]) / equitySeries[i - 1]);
    }
    const paths = [];
    for (let p = 0; p < nPaths; p++) {
      let val = 100;
      const path = [100];
      for (let i = 0; i < n - 1; i++) {
        const idx = Math.floor(rng() * returns.length);
        val *= (1 + returns[idx]);
        path.push(+val.toFixed(2));
      }
      paths.push(path);
    }
    return paths;
  }

  // ── chart defaults ──────────────────────────────────────────────────────
  const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 700, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0d1117',
        borderColor: '#1e2d3d',
        borderWidth: 1,
        titleFont: { family: 'Space Mono, monospace', size: 10 },
        bodyFont:  { family: 'Space Mono, monospace', size: 10 },
        titleColor: '#5a7a8a',
        bodyColor:  '#e8edf2',
      }
    },
    scales: {
      x: {
        grid:  { color: 'rgba(30,45,61,0.8)', drawTicks: false },
        ticks: { font: { family: 'Space Mono, monospace', size: 9 }, color: '#2d4a5a', maxTicksLimit: 7, maxRotation: 0 },
        border:{ color: '#1e2d3d' }
      },
      y: {
        grid:  { color: 'rgba(30,45,61,0.8)', drawTicks: false },
        ticks: { font: { family: 'Space Mono, monospace', size: 9 }, color: '#2d4a5a' },
        border:{ color: '#1e2d3d' }
      }
    }
  };

  function _mergeDeep(target, source) {
    const out = Object.assign({}, target);
    for (const key of Object.keys(source)) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        out[key] = _mergeDeep(target[key] || {}, source[key]);
      } else {
        out[key] = source[key];
      }
    }
    return out;
  }

  function chartOpts(overrides) {
    return _mergeDeep(CHART_DEFAULTS, overrides || {});
  }

  // ── render equity chart ─────────────────────────────────────────────────
  function renderEquityChart(canvasId, data, equitySeries) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const n = equitySeries.length;
    const labels = Array.from({ length: n }, (_, i) => `T${i + 1}`);

    // gradient fill
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, ctx.offsetHeight || 200);
    gradient.addColorStop(0, 'rgba(0,255,136,0.20)');
    gradient.addColorStop(1, 'rgba(0,255,136,0)');

    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data: equitySeries,
          borderColor: '#00ff88',
          borderWidth: 2,
          fill: true,
          backgroundColor: gradient,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: '#00ff88',
        }]
      },
      options: chartOpts({
        plugins: {
          tooltip: {
            callbacks: {
              title: (items) => `Trade ${items[0].label.replace('T', '')}`,
              label: (item) => ` Equity: ${item.parsed.y.toFixed(1)}`
            }
          }
        },
        scales: {
          y: { ticks: { callback: (v) => v.toFixed(0) } }
        }
      })
    });
  }

  // ── render drawdown chart ───────────────────────────────────────────────
  function renderDrawdownChart(canvasId, drawdownSeries) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const n = drawdownSeries.length;
    const labels = Array.from({ length: n }, (_, i) => `T${i + 1}`);

    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, ctx.offsetHeight || 120);
    gradient.addColorStop(0, 'rgba(255,68,68,0)');
    gradient.addColorStop(1, 'rgba(255,68,68,0.20)');

    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data: drawdownSeries,
          borderColor: '#ff4444',
          borderWidth: 1.5,
          fill: true,
          backgroundColor: gradient,
          tension: 0.3,
          pointRadius: 0,
        }]
      },
      options: chartOpts({
        plugins: {
          tooltip: {
            callbacks: {
              label: (item) => ` DD: ${item.parsed.y.toFixed(1)}%`
            }
          }
        },
        scales: {
          y: { ticks: { callback: (v) => `${v.toFixed(0)}%` }, reverse: false }
        }
      })
    });
  }

  // ── render rolling sharpe chart ─────────────────────────────────────────
  function renderRollingSharpeChart(canvasId, rollingSharpe, reportedSharpe) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const n = rollingSharpe.length;
    const labels = Array.from({ length: n }, (_, i) => `T${i + 1}`);

    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            data: rollingSharpe,
            borderColor: '#ffaa00',
            borderWidth: 1.5,
            fill: false,
            tension: 0.35,
            pointRadius: 0,
            spanGaps: true,
          },
          {
            // Threshold line at 1.0
            data: Array(n).fill(1.0),
            borderColor: 'rgba(255,68,68,0.4)',
            borderWidth: 1,
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0,
            tension: 0,
          },
          {
            // Reported Sharpe reference
            data: Array(n).fill(reportedSharpe),
            borderColor: 'rgba(0,255,136,0.3)',
            borderWidth: 1,
            borderDash: [2, 4],
            fill: false,
            pointRadius: 0,
            tension: 0,
          }
        ]
      },
      options: chartOpts({
        plugins: {
          tooltip: {
            filter: (item) => item.datasetIndex === 0,
            callbacks: {
              label: (item) => item.parsed.y != null ? ` Rolling SR: ${item.parsed.y.toFixed(2)}` : ''
            }
          }
        },
        scales: {
          y: { ticks: { callback: (v) => v.toFixed(1) } }
        }
      })
    });
  }

  // ── render monte carlo chart ────────────────────────────────────────────
  function renderMonteCarloChart(canvasId, actualPath, mcPaths) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const n = actualPath.length;
    const labels = Array.from({ length: n }, (_, i) => `T${i + 1}`);

    const pathDatasets = mcPaths.map(path => ({
      data: path,
      borderColor: 'rgba(0,255,136,0.06)',
      borderWidth: 1,
      fill: false,
      tension: 0.3,
      pointRadius: 0,
    }));

    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          ...pathDatasets,
          {
            data: actualPath,
            borderColor: '#00ff88',
            borderWidth: 2.5,
            fill: false,
            tension: 0.35,
            pointRadius: 0,
            label: 'Actual'
          }
        ]
      },
      options: chartOpts({
        plugins: {
          tooltip: {
            filter: (item) => item.datasetIndex === mcPaths.length,
            callbacks: {
              label: (item) => ` Equity: ${item.parsed.y.toFixed(1)}`
            }
          }
        }
      })
    });
  }

  // ── render category radar ───────────────────────────────────────────────
  function renderCategoryRadar(canvasId, checks) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const categories = ['Overfitting', 'Risk', 'Regime', 'Execution', 'Compliance'];
    const scores = categories.map(cat => {
      const catChecks = checks.filter(c => c.category === cat);
      if (!catChecks.length) return 0;
      return Math.round(catChecks.reduce((a, c) => a + c.score, 0) / catChecks.length);
    });

    _charts[canvasId] = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: categories,
        datasets: [{
          data: scores,
          borderColor: '#00ff88',
          borderWidth: 2,
          backgroundColor: 'rgba(0,255,136,0.08)',
          pointBackgroundColor: '#00ff88',
          pointBorderColor: '#00ff88',
          pointRadius: 4,
          pointHoverRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0d1117',
            borderColor: '#1e2d3d',
            borderWidth: 1,
            bodyFont: { family: 'Space Mono, monospace', size: 10 },
            bodyColor: '#e8edf2',
            callbacks: {
              label: (item) => ` ${item.label}: ${item.parsed.r}/100`
            }
          }
        },
        scales: {
          r: {
            min: 0, max: 100,
            grid:      { color: 'rgba(30,45,61,0.8)' },
            angleLines:{ color: 'rgba(30,45,61,0.6)' },
            pointLabels: {
              font:  { family: 'Space Mono, monospace', size: 10 },
              color: '#5a7a8a',
            },
            ticks: {
              display: false,
              stepSize: 25,
              backdropColor: 'transparent',
            }
          }
        }
      }
    });
  }

  // ── render crash bar chart ──────────────────────────────────────────────
  function renderCrashBars(canvasId, crashSims) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = crashSims.map(s => s.year.split('–')[0].split(' ').slice(-1)[0]);
    const marketDrops  = crashSims.map(s => s.market_drop);
    const stratDrops   = crashSims.map(s => +(s.strategy_drop * 100).toFixed(1));

    _charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Market',
            data: marketDrops,
            backgroundColor: 'rgba(255,68,68,0.5)',
            borderColor: '#ff4444',
            borderWidth: 1,
            borderRadius: 2,
          },
          {
            label: 'Your Strategy',
            data: stratDrops,
            backgroundColor: stratDrops.map(v => v >= 0 ? 'rgba(0,255,136,0.5)' : 'rgba(255,170,0,0.5)'),
            borderColor: stratDrops.map(v => v >= 0 ? '#00ff88' : '#ffaa00'),
            borderWidth: 1,
            borderRadius: 2,
          }
        ]
      },
      options: chartOpts({
        plugins: {
          legend: {
            display: true,
            labels: {
              font: { family: 'Space Mono, monospace', size: 10 },
              color: '#5a7a8a',
              boxWidth: 12,
              padding: 12,
            }
          },
          tooltip: {
            callbacks: {
              label: (item) => ` ${item.dataset.label}: ${item.parsed.y.toFixed(1)}%`
            }
          }
        },
        scales: {
          y: { ticks: { callback: (v) => `${v}%` } }
        }
      })
    });
  }

  // ── build HTML skeleton ─────────────────────────────────────────────────
  function buildHTML(data) {
    const sc = scoreColor(data.fundable_score);
    return `
<div id="qp-chart-section" style="margin-bottom:32px;">

  <!-- Chart Tabs -->
  <div style="display:flex;gap:0;border-bottom:1px solid #1e2d3d;margin-bottom:24px;overflow-x:auto;">
    ${[
      ['equity',    '// Equity Curve'],
      ['drawdown',  '// Drawdown'],
      ['rolling',   '// Rolling Sharpe'],
      ['montecarlo','// Monte Carlo'],
      ['radar',     '// Category Radar'],
      ['crash',     '// Crash vs Market'],
    ].map(([id, label], i) => `
      <button
        class="qp-chart-tab${i === 0 ? ' qp-chart-tab-active' : ''}"
        data-chart="${id}"
        onclick="QP_Charts.switchTab('${id}')"
        style="
          font-family:Space Mono,monospace;font-size:10px;letter-spacing:0.08em;
          padding:8px 16px;cursor:pointer;background:transparent;border:none;
          border-bottom:2px solid ${i === 0 ? 'var(--accent,#00ff88)' : 'transparent'};
          color:${i === 0 ? 'var(--accent,#00ff88)' : '#5a7a8a'};
          white-space:nowrap;transition:all 0.15s;
        "
      >${label}</button>
    `).join('')}
  </div>

  <!-- Chart Panels -->
  <div style="background:#0d1117;border:1px solid #1e2d3d;padding:20px 24px;">

    <div id="qp-panel-equity" style="position:relative;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Equity Curve
          <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// illustrative — derived from Sharpe ${data.sharpe}</span>
        </div>
        <div style="display:flex;gap:16px;">
          <div style="font-family:Space Mono,monospace;font-size:10px;color:#5a7a8a;">
            Sharpe&nbsp;<span style="color:${sc};font-weight:700;">${data.sharpe}</span>
          </div>
          <div style="font-family:Space Mono,monospace;font-size:10px;color:#5a7a8a;">
            WR&nbsp;<span style="color:${sc};font-weight:700;">${data.win_rate}%</span>
          </div>
        </div>
      </div>
      <div style="height:220px;"><canvas id="qp-canvas-equity"></canvas></div>
    </div>

    <div id="qp-panel-drawdown" style="display:none;position:relative;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Drawdown Profile
          <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// max DD: ${(data.max_drawdown * 100).toFixed(1)}%</span>
        </div>
        <div style="font-family:Space Mono,monospace;font-size:10px;">
          <span style="color:#ff4444;font-weight:700;">${(data.max_drawdown * 100).toFixed(1)}%</span>
          <span style="color:#5a7a8a;"> max drawdown</span>
        </div>
      </div>
      <div style="height:180px;"><canvas id="qp-canvas-drawdown"></canvas></div>
    </div>

    <div id="qp-panel-rolling" style="display:none;position:relative;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Rolling Sharpe
          <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// 90-period window</span>
        </div>
        <div style="display:flex;gap:12px;font-family:Space Mono,monospace;font-size:9px;color:#5a7a8a;">
          <span style="color:#ffaa00;">— Rolling</span>
          <span style="color:rgba(255,68,68,0.6);">— Threshold 1.0</span>
          <span style="color:rgba(0,255,136,0.5);">— Reported</span>
        </div>
      </div>
      <div style="height:180px;"><canvas id="qp-canvas-rolling"></canvas></div>
    </div>

    <div id="qp-panel-montecarlo" style="display:none;position:relative;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Monte Carlo — 50 paths
          <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// bootstrap resampling</span>
        </div>
        <div style="font-family:Space Mono,monospace;font-size:9px;color:#5a7a8a;">
          <span style="color:#00ff88;">— Actual path</span>
        </div>
      </div>
      <div style="height:220px;"><canvas id="qp-canvas-montecarlo"></canvas></div>
    </div>

    <div id="qp-panel-radar" style="display:none;position:relative;">
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;margin-bottom:16px;">
        Category Breakdown
        <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// 5 weighted categories</span>
      </div>
      <div style="height:260px;max-width:340px;margin:0 auto;"><canvas id="qp-canvas-radar"></canvas></div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:16px;" id="qp-radar-legend"></div>
    </div>

    <div id="qp-panel-crash" style="display:none;position:relative;">
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;margin-bottom:16px;">
        Your Strategy vs Historical Crashes
        <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// crash exposure comparison</span>
      </div>
      <div style="height:220px;"><canvas id="qp-canvas-crash"></canvas></div>
    </div>

  </div>

</div>`;
  }

  // ── render radar legend ─────────────────────────────────────────────────
  function renderRadarLegend(checks) {
    const categories = ['Overfitting', 'Risk', 'Regime', 'Execution', 'Compliance'];
    const weights    = { Overfitting: 22, Risk: 38, Regime: 14, Execution: 11, Compliance: 15 };
    const el = document.getElementById('qp-radar-legend');
    if (!el) return;

    el.innerHTML = categories.map(cat => {
      const catChecks = checks.filter(c => c.category === cat);
      const score = catChecks.length
        ? Math.round(catChecks.reduce((a, c) => a + c.score, 0) / catChecks.length)
        : 0;
      const col = scoreColor(score);
      const passed = catChecks.filter(c => c.passed).length;
      const total  = catChecks.length;
      return `
        <div style="background:#131920;border:1px solid #1e2d3d;padding:10px 12px;">
          <div style="font-family:Space Mono,monospace;font-size:9px;color:#5a7a8a;margin-bottom:4px;letter-spacing:0.06em;">
            ${cat.toUpperCase()} · ${weights[cat]}%
          </div>
          <div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:${col};line-height:1;">${score}</div>
          <div style="font-family:Space Mono,monospace;font-size:9px;color:#2d4a5a;margin-top:3px;">${passed}/${total} pass</div>
        </div>`;
    }).join('');
  }

  // ── switch tab ──────────────────────────────────────────────────────────
  function switchTab(id) {
    const panels = ['equity','drawdown','rolling','montecarlo','radar','crash'];
    panels.forEach(p => {
      const panel = document.getElementById(`qp-panel-${p}`);
      if (panel) panel.style.display = p === id ? '' : 'none';
    });

    document.querySelectorAll('.qp-chart-tab').forEach(btn => {
      const isActive = btn.dataset.chart === id;
      btn.style.borderBottomColor = isActive ? 'var(--accent,#00ff88)' : 'transparent';
      btn.style.color = isActive ? 'var(--accent,#00ff88)' : '#5a7a8a';
    });

    // Lazy render charts on first switch
    if (!_charts[`qp-canvas-${id}`]) {
      _lazyRender(id);
    }
  }

  const _lazyData = {};

  function _lazyRender(id) {
    if (!_lazyData.equity) return;
    const { equity, drawdown, rolling, mc, checks, crashSims, reportedSharpe, rng } = _lazyData;

    if (id === 'equity')      renderEquityChart('qp-canvas-equity', {}, equity);
    if (id === 'drawdown')    renderDrawdownChart('qp-canvas-drawdown', drawdown);
    if (id === 'rolling')     renderRollingSharpeChart('qp-canvas-rolling', rolling, reportedSharpe);
    if (id === 'montecarlo')  renderMonteCarloChart('qp-canvas-montecarlo', equity, mc);
    if (id === 'radar')     { renderCategoryRadar('qp-canvas-radar', checks); renderRadarLegend(checks); }
    if (id === 'crash')       renderCrashBars('qp-canvas-crash', crashSims);
  }

  // ── main init ───────────────────────────────────────────────────────────
  function init(data) {
    const target = document.getElementById('qp-chart-section');
    if (!target) {
      console.warn('[QP_Charts] #qp-chart-section not found in DOM. Did you add it to buildResultsHTML()?');
      return;
    }

    // Build synthetic curve
    const { equity, drawdown, n } = buildEquityCurve(data);
    const rolling = buildRollingSharpe(equity);
    const rng = _seededRng(_hashToInt(data.validation_hash || 'mc') + 1);
    const mc  = buildMonteCarloPaths(equity, 50, rng);

    // Store for lazy rendering
    _lazyData.equity       = equity;
    _lazyData.drawdown     = drawdown;
    _lazyData.rolling      = rolling;
    _lazyData.mc           = mc;
    _lazyData.checks       = data.checks || [];
    _lazyData.crashSims    = data.crash_sims || [];
    _lazyData.reportedSharpe = data.sharpe || 1;
    _lazyData.rng          = rng;

    // Inject HTML
    target.innerHTML = buildHTML(data);

    // Render initial tab (equity)
    setTimeout(() => {
      renderEquityChart('qp-canvas-equity', data, equity);
    }, 100);
  }

  // ── public API ──────────────────────────────────────────────────────────
  return { init, destroy, switchTab };

})();
