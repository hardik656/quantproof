/**
 * QuantProof Chart Layer v2.0
 * Uses REAL equity_curve data from the validator.
 *
 * REQUIRED BACKEND CHANGES (add to validator.py + main.py):
 *
 * validator.py — In ValidationReport dataclass, add:
 *   equity_curve: list = None
 *
 * validator.py — In QuantProofValidator.run(), before the return, add:
 *   equity_curve_raw = [round(v, 4) for v in (np.cumprod(1 + r) / np.cumprod(1 + r)[0] * 100).tolist()]
 *   ...and in ValidationReport(...): equity_curve=equity_curve_raw,
 *
 * main.py — In ValidationResponse Pydantic model, add:
 *   equity_curve: Optional[List[float]] = None
 *
 * main.py — In every ValidationResponse(...) return, add:
 *   equity_curve=report.equity_curve,
 *
 * index.html — 5 line changes (see bottom of this file for exact instructions)
 */

const QP_Charts = (function () {
  'use strict';

  const _charts = {};

  function _destroy(id) {
    if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
  }

  function destroy() {
    Object.keys(_charts).forEach(_destroy);
    const el = document.getElementById('qp-chart-section');
    if (el) el.innerHTML = '';
  }

  function scoreColor(s) {
    if (s >= 80) return '#00ff88';
    if (s >= 65) return '#00cc6e';
    if (s >= 50) return '#ffaa00';
    if (s >= 35) return '#ff6600';
    return '#ff4444';
  }

  function _seededRng(seed) {
    let s = seed;
    return function () {
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

  // ── REAL equity curve from validator ─────────────────────────────────────
  // Uses data.equity_curve (np.cumprod(1+r) normalized to 100, from validator.py)
  // Falls back to synthetic only if backend not yet updated
  function buildEquityCurve(data) {
    if (data.equity_curve && data.equity_curve.length > 5) {
      const equity = data.equity_curve;
      let peak = equity[0];
      const drawdown = equity.map(v => {
        peak = Math.max(peak, v);
        return +((v / peak - 1) * 100).toFixed(2);
      });
      return { equity, drawdown, n: equity.length, isReal: true };
    }

    // Fallback: synthetic
    const n   = Math.min(Math.max(data.total_trades || 60, 40), 200);
    const rng = _seededRng(_hashToInt(data.validation_hash || 'qp'));
    const sharpe = data.sharpe || 1.0;
    const meanR  = sharpe * 0.015 / Math.sqrt(252 / n);
    const stdR   = meanR / (sharpe + 1e-9);
    let equity   = 100;
    const equitySeries   = [equity];
    const drawdownSeries = [0];
    let peak = equity;
    for (let i = 1; i < n; i++) {
      const u1 = Math.max(1e-10, rng());
      const u2 = Math.max(1e-10, rng());
      let r = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      r = Math.max(-0.99, Math.min(0.99, r * stdR + meanR));
      equity *= (1 + r);
      peak    = Math.max(peak, equity);
      equitySeries.push(+equity.toFixed(2));
      drawdownSeries.push(-((peak - equity) / peak * 100));
    }
    return { equity: equitySeries, drawdown: drawdownSeries, n, isReal: false };
  }

  function buildRollingSharpe(equitySeries, windowSize) {
    const w      = windowSize || Math.max(10, Math.floor(equitySeries.length / 8));
    const result = new Array(equitySeries.length).fill(null);
    for (let i = w; i < equitySeries.length; i++) {
      const rets = [];
      for (let j = i - w; j < i; j++) {
        rets.push((equitySeries[j + 1] - equitySeries[j]) / equitySeries[j]);
      }
      const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
      const std  = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length - 1));
      result[i]  = std < 1e-9 ? 0 : +(mean / std * Math.sqrt(252)).toFixed(3);
    }
    return result;
  }

  function buildMonteCarloPaths(equitySeries, nPaths, rng) {
    const n       = equitySeries.length;
    const returns = [];
    for (let i = 1; i < n; i++) {
      returns.push((equitySeries[i] - equitySeries[i - 1]) / equitySeries[i - 1]);
    }
    return Array.from({ length: nPaths }, () => {
      let val    = 100;
      const path = [100];
      for (let i = 0; i < n - 1; i++) {
        val *= (1 + returns[Math.floor(rng() * returns.length)]);
        path.push(+val.toFixed(2));
      }
      return path;
    });
  }

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
        grid:   { color: 'rgba(30,45,61,0.8)', drawTicks: false },
        ticks:  { font: { family: 'Space Mono, monospace', size: 9 }, color: '#2d4a5a', maxTicksLimit: 7, maxRotation: 0 },
        border: { color: '#1e2d3d' }
      },
      y: {
        grid:   { color: 'rgba(30,45,61,0.8)', drawTicks: false },
        ticks:  { font: { family: 'Space Mono, monospace', size: 9 }, color: '#2d4a5a' },
        border: { color: '#1e2d3d' }
      }
    }
  };

  function _merge(t, s) {
    const o = Object.assign({}, t);
    for (const k of Object.keys(s)) {
      o[k] = (s[k] && typeof s[k] === 'object' && !Array.isArray(s[k]))
        ? _merge(t[k] || {}, s[k])
        : s[k];
    }
    return o;
  }
  function chartOpts(ov) { return _merge(CHART_DEFAULTS, ov || {}); }

  function renderEquityChart(canvasId, equitySeries, drawdownSeries) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const labels   = equitySeries.map((_, i) => `T${i + 1}`);
    const eqH = ctx.offsetHeight || 200;
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, eqH);
    gradient.addColorStop(0, 'rgba(0,255,136,0.12)');
    gradient.addColorStop(1, 'rgba(0,255,136,0)');
    const dd = drawdownSeries || equitySeries.map(() => 0);
    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          // Baseline: faint dashed line at starting equity
          {
            data: Array(equitySeries.length).fill(equitySeries[0] || 100),
            borderColor: 'rgba(180,180,180,0.18)',
            borderWidth: 1,
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0,
            order: 2,
          },
          // IMPROVEMENT 3: equity curve coloured by drawdown depth
          {
            data: equitySeries,
            borderColor: '#00ff88',
            borderWidth: 2,
            fill: true,
            backgroundColor: gradient,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: '#00ff88',
            // segment colouring: green=ATH/minor, amber=moderate, red=severe
            segment: {
              borderColor: ctx => {
                const d = Math.abs(dd[ctx.p1DataIndex] || 0);
                const maxDD = Math.abs(Math.min(...dd));
                // Thresholds relative to strategy's own max DD
                if (d < maxDD * 0.15) return '#00ff88';   // near ATH
                if (d < maxDD * 0.50) return '#ffaa00';   // moderate DD
                return '#ff5544';                          // major DD
              }
            }
          }
        ]
      },
      options: chartOpts({
        plugins: {
          tooltip: {
            filter: item => item.datasetIndex === 1,
            callbacks: {
              title: items => `Trade ${items[0].label.replace('T', '')}`,
              label: item  => {
                const d = dd[item.dataIndex] || 0;
                return [` Equity: ${item.parsed.y.toFixed(2)}`, ` DD: ${d.toFixed(2)}%`];
              }
            }
          }
        },
        scales: { y: { ticks: { callback: v => v.toFixed(0) } } }
      })
    });
  }

  function renderDrawdownChart(canvasId, drawdownSeries) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const labels   = drawdownSeries.map((_, i) => `T${i + 1}`);
    const h = ctx.offsetHeight || 80;
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, 'rgba(255,68,68,0.05)');
    gradient.addColorStop(1, 'rgba(255,68,68,0.30)');
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
        plugins: { tooltip: { callbacks: {
          title: items => `Trade ${items[0].label.replace('T','')}`,
          label: item  => ` DD: ${item.parsed.y.toFixed(2)}%`
        }}},
        scales: {
          x: { ticks: { maxTicksLimit: 5, font: { size: 8 }, color: '#1e3040' }, grid: { color: 'rgba(30,45,61,0.4)' } },
          y: { ticks: { maxTicksLimit: 3, font: { size: 8 }, callback: v => `${v.toFixed(0)}%` }, grid: { color: 'rgba(30,45,61,0.4)' } }
        }
      })
    });
  }

  function renderRollingSharpeChart(canvasId, rollingSharpe, reportedSharpe) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const n      = rollingSharpe.length;
    const labels = rollingSharpe.map((_, i) => `T${i + 1}`);
    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { data: rollingSharpe, borderColor: '#ffaa00', borderWidth: 1.5, fill: false, tension: 0.35, pointRadius: 0, spanGaps: true },
          { data: Array(n).fill(1.0), borderColor: 'rgba(255,68,68,0.4)', borderWidth: 1, borderDash: [4, 4], fill: false, pointRadius: 0, tension: 0 },
          { data: Array(n).fill(reportedSharpe), borderColor: 'rgba(0,255,136,0.3)', borderWidth: 1, borderDash: [2, 4], fill: false, pointRadius: 0, tension: 0 },
        ]
      },
      options: chartOpts({
        plugins: {
          tooltip: {
            filter: item => item.datasetIndex === 0,
            callbacks: { label: item => item.parsed.y != null ? ` Rolling SR: ${item.parsed.y.toFixed(2)}` : '' }
          }
        },
        scales: { y: { ticks: { callback: v => v.toFixed(1) } } }
      })
    });
  }

  function renderMonteCarloChart(canvasId, actualPath, mcPaths) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // DEBUG: Check mcPaths structure in detail
    console.log('[DEBUG] renderMonteCarloChart called with:');
    console.log('  - actualPath length:', actualPath.length);
    console.log('  - mcPaths type:', typeof mcPaths);
    console.log('  - mcPaths length:', mcPaths.length);
    console.log('  - mcPaths[0] type:', typeof mcPaths[0]);
    console.log('  - mcPaths[0] length:', mcPaths[0]?.length);
    console.log('  - mcPaths[0] sample:', mcPaths[0]?.slice(0, 5));
    
    const labels = actualPath.map((_, i) => `T${i + 1}`);
    const n = actualPath.length;
    const datasets = [
      ...mcPaths.map(path => ({ data: path.slice(0, n), borderColor: 'rgba(0,255,136,0.45)', borderWidth: 1, fill: false, tension: 0.1, pointRadius: 0 })),
      { data: actualPath, borderColor: '#00ff88', borderWidth: 2.5, fill: false, tension: 0.35, pointRadius: 0 }
    ];
    
    console.log('[DEBUG] Chart datasets:', datasets.length, 'total datasets');
    
    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: datasets
      },
      options: chartOpts({
        clip: false,
        datasets: {
          line: {
            clip: false
          }
        },
        plugins: {
          tooltip: {
            filter: item => item.datasetIndex === mcPaths.length,
            callbacks: { label: item => ` Equity: ${item.parsed.y.toFixed(2)}` }
          }
        },
        scales: {
          x: { clip: false },
          y: { clip: false }
        }
      })
    });
  }

  function renderCategoryRadar(canvasId, checks) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const cats   = ['Overfitting', 'Risk', 'Regime', 'Execution', 'Compliance'];
    const scores = cats.map(cat => {
      const cc = checks.filter(c => c.category === cat);
      return cc.length ? Math.round(cc.reduce((a, c) => a + c.score, 0) / cc.length) : 0;
    });
    _charts[canvasId] = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: cats,
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
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0d1117', borderColor: '#1e2d3d', borderWidth: 1,
            bodyFont: { family: 'Space Mono, monospace', size: 10 }, bodyColor: '#e8edf2',
            callbacks: { label: item => ` ${item.label}: ${item.parsed.r}/100` }
          }
        },
        scales: {
          r: {
            min: 0, max: 100,
            grid: { color: 'rgba(30,45,61,0.8)' }, angleLines: { color: 'rgba(30,45,61,0.6)' },
            pointLabels: { font: { family: 'Space Mono, monospace', size: 10 }, color: '#5a7a8a' },
            ticks: { display: false, stepSize: 25, backdropColor: 'transparent' }
          }
        }
      }
    });
  }

  function renderCrashBars(canvasId, crashSims) {
    _destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const labels      = crashSims.map(s => s.year.split('–')[0].split(' ').slice(-1)[0]);
    const marketDrops = crashSims.map(s => s.market_drop);
    const stratDrops  = crashSims.map(s => +(s.strategy_drop * 100).toFixed(1));
    _charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Market', data: marketDrops, backgroundColor: 'rgba(255,68,68,0.5)', borderColor: '#ff4444', borderWidth: 1, borderRadius: 2 },
          {
            label: 'Your Strategy',
            data: stratDrops,
            backgroundColor: stratDrops.map(v => v >= 0 ? 'rgba(0,255,136,0.5)' : 'rgba(255,170,0,0.5)'),
            borderColor:     stratDrops.map(v => v >= 0 ? '#00ff88' : '#ffaa00'),
            borderWidth: 1, borderRadius: 2
          }
        ]
      },
      options: chartOpts({
        plugins: {
          legend: { display: true, labels: { font: { family: 'Space Mono, monospace', size: 10 }, color: '#5a7a8a', boxWidth: 12, padding: 12 } },
          tooltip: { callbacks: { label: item => ` ${item.dataset.label}: ${item.parsed.y.toFixed(1)}%` } }
        },
        scales: { y: { ticks: { callback: v => `${v}%` } } }
      })
    });
  }

  function renderRadarLegend(checks) {
    const el = document.getElementById('qp-radar-legend');
    if (!el) return;
    const cats    = ['Overfitting', 'Risk', 'Regime', 'Execution', 'Compliance'];
    const weights = { Overfitting: 22, Risk: 38, Regime: 14, Execution: 11, Compliance: 15 };
    el.innerHTML = cats.map(cat => {
      const cc     = checks.filter(c => c.category === cat);
      const score  = cc.length ? Math.round(cc.reduce((a, c) => a + c.score, 0) / cc.length) : 0;
      const passed = cc.filter(c => c.passed).length;
      const col    = scoreColor(score);
      return `
        <div style="background:#131920;border:1px solid #1e2d3d;padding:10px 12px;">
          <div style="font-family:Space Mono,monospace;font-size:9px;color:#5a7a8a;margin-bottom:4px;letter-spacing:0.06em;">${cat.toUpperCase()} · ${weights[cat]}%</div>
          <div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:${col};line-height:1;">${score}</div>
          <div style="font-family:Space Mono,monospace;font-size:9px;color:#2d4a5a;margin-top:3px;">${passed}/${cc.length} pass</div>
        </div>`;
    }).join('');
  }

  function switchTab(id) {
    ['equity', 'rolling', 'montecarlo', 'radar', 'crash'].forEach(p => {
      const el = document.getElementById(`qp-panel-${p}`);
      if (el) el.style.display = p === id ? '' : 'none';
    });
    document.querySelectorAll('.qp-chart-tab').forEach(btn => {
      const active = btn.dataset.chart === id;
      btn.style.borderBottomColor = active ? 'var(--accent,#00ff88)' : 'transparent';
      btn.style.color = active ? 'var(--accent,#00ff88)' : '#5a7a8a';
    });
    if (!_charts[`qp-canvas-${id}`]) _lazyRender(id);
  }

  const _store = {};

  function _lazyRender(id) {
    if (!_store.equity) return;
    const { equity, drawdown, rolling, mc, checks, crashSims, reportedSharpe } = _store;
    if (id === 'equity')    { renderEquityChart('qp-canvas-equity', equity, drawdown); renderDrawdownChart('qp-canvas-drawdown', drawdown); }
    if (id === 'rolling')     renderRollingSharpeChart('qp-canvas-rolling', rolling, reportedSharpe);
    if (id === 'montecarlo')  renderMonteCarloChart('qp-canvas-montecarlo', equity, mc);
    if (id === 'radar')     { renderCategoryRadar('qp-canvas-radar', checks); renderRadarLegend(checks); }
    if (id === 'crash')       renderCrashBars('qp-canvas-crash', crashSims);
  }

  function _buildHTML(data, isReal) {
    const sc        = scoreColor(data.fundable_score);
    const curveNote = isReal ? '// actual trade-by-trade equity curve' : `// illustrative — derived from Sharpe ${data.sharpe}`;
    const mcNote    = isReal ? '// resampled from actual returns' : '// resampled from synthetic returns';
    const tabs = [
      ['equity', '// Equity Curve'],
      ['rolling', '// Rolling Sharpe'], ['montecarlo', '// Monte Carlo'],
      ['radar', '// Category Radar'], ['crash', '// Crash vs Market'],
    ];
    return `
<div id="qp-chart-section" style="margin-bottom:32px;">
  <div style="display:flex;gap:0;border-bottom:1px solid #1e2d3d;margin-bottom:24px;overflow-x:auto;">
    ${tabs.map(([id, label], i) => `
      <button class="qp-chart-tab" data-chart="${id}" onclick="QP_Charts.switchTab('${id}')"
        style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:0.08em;
               padding:8px 16px;cursor:pointer;background:transparent;border:none;
               border-bottom:2px solid ${i === 0 ? 'var(--accent,#00ff88)' : 'transparent'};
               color:${i === 0 ? 'var(--accent,#00ff88)' : '#5a7a8a'};white-space:nowrap;transition:all 0.15s;">
        ${label}</button>`).join('')}
  </div>
  <div style="background:#0d1117;border:1px solid #1e2d3d;padding:20px 24px;">

    <div id="qp-panel-equity">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Equity Curve <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">${curveNote}</span>
        </div>
        <div style="display:flex;gap:16px;font-family:Space Mono,monospace;font-size:10px;color:#5a7a8a;">
          <span>Sharpe&nbsp;<b style="color:${sc}">${data.sharpe}</b></span>
          <span>WR&nbsp;<b style="color:${sc}">${data.win_rate}%</b></span>
          <span>Trades&nbsp;<b style="color:${sc}">${data.total_trades}</b></span>
        </div>
      </div>
      <div style="height:200px;"><canvas id="qp-canvas-equity"></canvas></div>
      <!-- STACKED: Drawdown always visible below equity -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;margin-bottom:4px;">
        <div style="font-family:Space Mono,monospace;font-size:9px;color:#5a7a8a;letter-spacing:0.06em;">DRAWDOWN</div>
        <span style="font-family:Space Mono,monospace;font-size:9px;color:#ff4444;">${(data.max_drawdown * 100).toFixed(1)}% max</span>
      </div>
      <div style="height:72px;"><canvas id="qp-canvas-drawdown"></canvas></div>
    </div>

    <div id="qp-panel-rolling" style="display:none;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Rolling Sharpe <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// 90-trade window</span>
        </div>
        <div style="display:flex;gap:12px;font-family:Space Mono,monospace;font-size:9px;color:#5a7a8a;">
          <span style="color:#ffaa00;">— Rolling</span>
          <span style="color:rgba(255,68,68,0.6);">— Min 1.0</span>
          <span style="color:rgba(0,255,136,0.5);">— Reported ${data.sharpe}</span>
        </div>
      </div>
      <div style="height:180px;"><canvas id="qp-canvas-rolling"></canvas></div>
    </div>

    <div id="qp-panel-montecarlo" style="display:none;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;">
          Monte Carlo — 50 bootstrap paths <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">${mcNote}</span>
        </div>
        <span style="font-family:Space Mono,monospace;font-size:9px;color:#00ff88;">— Actual path</span>
      </div>
      <div style="height:220px;"><canvas id="qp-canvas-montecarlo"></canvas></div>
    </div>

    <div id="qp-panel-radar" style="display:none;">
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;margin-bottom:16px;">
        Category Breakdown <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// scores directly from validator</span>
      </div>
      <div style="height:260px;max-width:340px;margin:0 auto;"><canvas id="qp-canvas-radar"></canvas></div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:16px;" id="qp-radar-legend"></div>
    </div>

    <div id="qp-panel-crash" style="display:none;">
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#e8edf2;font-weight:700;margin-bottom:16px;">
        Your Strategy vs Historical Crashes <span style="font-size:9px;color:#5a7a8a;font-weight:400;margin-left:8px;">// from validator crash simulations</span>
      </div>
      <div style="height:220px;"><canvas id="qp-canvas-crash"></canvas></div>
    </div>

  </div>
</div>`;
  }

  function init(data) {
    const target = document.getElementById('qp-chart-section');
    if (!target) return;

    const { equity, drawdown, n, isReal } = buildEquityCurve(data);
    const rolling = buildRollingSharpe(equity);
    const rng     = _seededRng(_hashToInt(data.validation_hash || 'mc') + 1);
    
    // Use backend Monte Carlo data if available, otherwise generate synthetic
    const mc = data.monte_carlo?.equity_sims || buildMonteCarloPaths(equity, 50, rng);

    // DEBUG: Print Monte Carlo data structure
    console.log('[DEBUG] Frontend Monte Carlo data:');
    console.log('  - data.monte_carlo:', data.monte_carlo);
    console.log('  - mc length:', mc.length);
    console.log('  - mc[0] length:', mc[0]?.length || 'undefined');
    console.log('  - mc[0] sample:', mc[0]?.slice(0, 5) || 'undefined');

    _store.equity         = equity;
    _store.drawdown       = drawdown;
    _store.rolling        = rolling;
    _store.mc             = mc;
    _store.checks         = data.checks || [];
    _store.crashSims      = data.crash_sims || [];
    _store.reportedSharpe = data.sharpe || 1;

    target.innerHTML = _buildHTML(data, isReal);
    setTimeout(() => {
      renderEquityChart('qp-canvas-equity', equity, drawdown);
      renderDrawdownChart('qp-canvas-drawdown', drawdown);
    }, 100);
  }

  return { init, destroy, switchTab };

})();

/*
 * ─────────────────────────────────────────────────────────────────
 * index.html — 5 exact changes (Windsurf / manual)
 * ─────────────────────────────────────────────────────────────────
 *
 * 1. In <head>, after the Google Fonts <link> tag:
 *    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
 *
 * 2. In buildResultsHTML(), directly before <div class="result-tabs">:
 *    <div id="qp-chart-section"></div>
 *
 * 3. In renderResults(), after the second section.classList.add('active');:
 *    setTimeout(() => { if(typeof QP_Charts !== 'undefined') QP_Charts.init(data); }, 50);
 *
 * 4. In resetApp(), after document.getElementById('resultsSection').innerHTML = '';:
 *    if(typeof QP_Charts !== 'undefined') QP_Charts.destroy();
 *
 * 5. Just before </body>:
 *    <script src="quantproof-charts.js"></script>
 * ─────────────────────────────────────────────────────────────────
 */
