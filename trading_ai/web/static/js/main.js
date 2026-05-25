/**
 * IQ Option AI Trading Dashboard – Main JS
 * Handles WebSocket connection, real-time UI updates, and user actions.
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const OTC_ASSETS = [
  'EUR/USD (OTC)',
  'GBP/USD (OTC)',
  'AUD/USD (OTC)',
  'GBP/JPY (OTC)',
];

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  ws:         null,
  connected:  false,
  aiRunning:  false,
  balance:    0,
  stats:      { trades: 0, wins: 0, pnl: 0 },
  prices:     {},   // asset → price
  signals:    {},   // asset → {action, confidence}
};

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    setStatus('เชื่อมต่อ server สำเร็จ', 'success');
  };

  ws.onclose = () => {
    setStatus('การเชื่อมต่อหลุด – กำลังเชื่อมต่อใหม่…', 'warn');
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => {
    setStatus('WebSocket error', 'error');
  };

  ws.onmessage = ({ data }) => {
    try {
      handleMessage(JSON.parse(data));
    } catch (e) {
      console.error('Bad WS message', e);
    }
  };
}

function send(msg) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(msg));
  }
}

// ── Message handler ───────────────────────────────────────────────────────────
function handleMessage(msg) {
  switch (msg.type) {

    case 'init':
      applyInit(msg);
      break;

    case 'connection':
      applyConnection(msg);
      break;

    case 'price':
      applyPrice(msg);
      break;

    case 'signal':
      applySignal(msg);
      break;

    case 'check':
      applyChecks(msg);
      break;

    case 'brain':
      applyBrain(msg);
      break;

    case 'candle_history':
      applyCandleHistory(msg);
      break;

    case 'trade':
      applyTrade(msg);
      break;

    case 'status':
      setStatus(msg.message, msg.level || 'info');
      break;

    case 'training':
      applyTraining(msg);
      break;
  }
}

function applyTraining(msg) {
  const el = document.getElementById('training-metrics');
  if (!el) return;
  el.innerHTML = `
    <span class="metric-chip">📉 Policy: ${msg.policy_loss?.toFixed(4) || '-'}</span>
    <span class="metric-chip">📊 Value: ${msg.value_loss?.toFixed(4) || '-'}</span>
    <span class="metric-chip">🔀 Entropy: ${msg.entropy?.toFixed(4) || '-'}</span>
    <span class="metric-chip">⚡ LR: ${msg.lr?.toFixed(6) || '-'}</span>
    <span class="metric-chip">🔄 Updates: ${msg.updates || 0}</span>
    <span class="metric-chip">👣 Steps: ${msg.steps || 0}</span>
  `;
}

// ── Apply functions ───────────────────────────────────────────────────────────
function applyInit(msg) {
  applyConnection({ connected: msg.connected, balance: msg.balance, account: '' });
  updateStats(msg.stats || {});
  if (msg.checks && msg.checks.length) applyChecks({ results: msg.checks });
  (msg.history || []).forEach(addTradeToHistory);
  applySettings(msg.settings || {});
  setAiState(msg.ai_running || false);
}

function applyConnection(msg) {
  state.connected = msg.connected;
  state.balance   = msg.balance || 0;

  const dot  = document.getElementById('conn-dot');
  const text = document.getElementById('conn-text');
  const bal  = document.getElementById('balance-val');

  dot.className  = 'dot ' + (msg.connected ? 'green' : 'red');
  text.textContent = msg.connected
    ? `เชื่อมต่อแล้ว (${msg.account || ''})`
    : 'ไม่ได้เชื่อมต่อ';

  if (bal) bal.textContent = `$${state.balance.toLocaleString('en', { minimumFractionDigits: 2 })}`;

  // Enable/disable trade buttons
  document.querySelectorAll('.btn-trade').forEach(btn => {
    btn.disabled = !msg.connected;
  });
}

function applyPrice(msg) {
  const { asset, price, change_pct, candle } = msg;
  state.prices[asset] = price;

  const card = document.querySelector(`[data-asset="${CSS.escape(asset)}"]`);
  if (!card) return;

  const priceEl  = card.querySelector('.price-val');
  const changeEl = card.querySelector('.price-change');
  const prev     = parseFloat(priceEl.dataset.prev || price);

  priceEl.textContent   = formatPrice(asset, price);
  priceEl.dataset.prev  = price;
  priceEl.className     = 'price-val ' + (price >= prev ? 'up' : 'down');

  const sign = change_pct >= 0 ? '+' : '';
  changeEl.textContent = `${sign}${change_pct.toFixed(3)}%`;
  changeEl.className   = 'price-change ' + (change_pct >= 0 ? 'up' : 'down');

  // Push candle to chart
  if (candle && window.charts && window.charts[asset]) {
    window.charts[asset].update(candle);
  }
}

function applySignal(msg) {
  const { asset, action, confidence } = msg;
  state.signals[asset] = { action, confidence };

  const card = document.querySelector(`[data-asset="${CSS.escape(asset)}"]`);
  if (!card) return;

  // Signal tag
  const tag = card.querySelector('.signal-tag');
  tag.className = `signal-tag ${action.toLowerCase()}`;
  const icons = { BUY: '▲', SELL: '▼', HOLD: '──' };
  tag.textContent = `${icons[action] || ''} ${action}  ${(confidence * 100).toFixed(0)}%`;

  // Confidence bar
  const fill  = card.querySelector('.conf-fill');
  const pct   = card.querySelector('.conf-pct');
  fill.style.width = `${(confidence * 100).toFixed(1)}%`;
  fill.className   = `conf-fill ${action.toLowerCase()}`;
  pct.textContent  = `${(confidence * 100).toFixed(0)}%`;

  // Card border
  card.className = `chart-card signal-${action.toLowerCase()}`;

  // Highlight active button
  const btnBuy  = card.querySelector('.btn-buy');
  const btnSell = card.querySelector('.btn-sell');
  btnBuy.style.opacity  = action === 'BUY'  ? '1' : '0.65';
  btnSell.style.opacity = action === 'SELL' ? '1' : '0.65';
}

function applyChecks(msg) {
  const { results, all_passed } = msg;
  if (!results) return;

  // Update banner
  const banner = document.getElementById('check-banner');
  const pills  = document.getElementById('check-pills');
  banner.className = all_passed ? 'pass' : 'fail';
  document.getElementById('banner-text').textContent = all_passed
    ? '✅ ผ่านการตรวจสอบทั้งหมด – AI พร้อมเทรด'
    : '❌ ยังไม่ผ่านการตรวจสอบบางข้อ';

  pills.innerHTML = '';
  results.forEach(r => {
    const pill = document.createElement('span');
    pill.className = `check-pill ${r.passed ? 'pass' : 'fail'}`;
    pill.textContent = (r.passed ? '✅ ' : '❌ ') + r.name;
    pill.onclick = () => showChecksModal(results);
    pills.appendChild(pill);
  });

  // Enable trade buttons if passed
  if (all_passed) {
    document.querySelectorAll('.btn-trade').forEach(b => b.disabled = false);
  }
}

function applyBrain(msg) {
  // ── Raw stats ──────────────────────────────────────────────────────────
  document.getElementById('brain-nodes').textContent    = msg.graph_nodes   || 0;
  document.getElementById('brain-branches').textContent = msg.graph_branches || 0;
  document.getElementById('brain-conf').textContent     = ((msg.avg_confidence || 0) * 100).toFixed(1) + '%';
  document.getElementById('brain-wr').textContent       = ((msg.recent_win_rate || 0) * 100).toFixed(1) + '%';
  document.getElementById('brain-episodes').textContent = msg.episodic_memories || 0;

  const wr = msg.recent_win_rate || 0;
  document.getElementById('brain-wr').style.color =
    wr > 0.55 ? 'var(--green)' : wr < 0.45 ? 'var(--red)' : 'var(--text)';

  // Type chips
  const chipContainer = document.getElementById('brain-types');
  chipContainer.innerHTML = '';
  const types = msg.knowledge_by_type || {};
  Object.entries(types).forEach(([t, n]) => {
    const chip = document.createElement('span');
    chip.className = 'type-chip';
    chip.textContent = `${t}: ${n}`;
    chipContainer.appendChild(chip);
  });

  // ── Brain Age ──────────────────────────────────────────────────────────
  if (msg.brain_age !== undefined) {
    applyBrainAge(msg);
  }

  // Navbar badges
  document.getElementById('brain-badge').textContent = `🧠 ${msg.graph_nodes || 0} nodes`;
}

function applyBrainAge(msg) {
  const age    = msg.brain_age   || 0;
  const score  = msg.brain_score || 0;
  const stage  = msg.brain_stage || 'ทารก';
  const emoji  = msg.brain_emoji || '👶';
  const desc   = msg.brain_desc  || '';
  const next   = msg.brain_next  || '';
  const pctNext = msg.brain_pct_next || 0;
  const bd     = msg.brain_breakdown || {};

  // Big age display
  document.getElementById('age-emoji').textContent = emoji;
  document.getElementById('age-num').textContent   = age.toFixed(1);
  document.getElementById('age-stage').textContent = `${stage} – ${desc}`;

  // Timeline fill (map age 0-80 → 0-100%)
  const fill = Math.min(age / 80, 1) * 100;
  document.getElementById('age-fill').style.width = fill + '%';

  // Stage pills
  document.querySelectorAll('.stage-pill').forEach(pill => {
    pill.classList.toggle('active', pill.dataset.stage === stage);
  });

  // Next milestone
  document.getElementById('age-next').textContent = next;

  // Breakdown bars (each out of 30, 30, 25, 10, 10 max)
  const maxes = { knowledge: 25, performance: 30, experience: 25, learning: 10, confidence: 10 };
  Object.entries(bd).forEach(([key, val]) => {
    if (key === 'total') {
      document.getElementById('score-total').textContent = val.toFixed(0);
      return;
    }
    const fillEl = document.getElementById(`bd-${key}`);
    const valEl  = document.getElementById(`bv-${key}`);
    if (fillEl) {
      const pct = Math.min(val / (maxes[key] || 25), 1) * 100;
      fillEl.style.width = pct + '%';
    }
    if (valEl) valEl.textContent = val.toFixed(0);
  });

  // Navbar brain age badge
  document.getElementById('nav-age-emoji').textContent = emoji;
  document.getElementById('nav-age-val').textContent   = age.toFixed(0);
  document.getElementById('nav-age-stage').textContent = stage;
}

// ── Candle history (initial load from IQ Option OTC data) ─────────────────────
function applyCandleHistory(msg) {
  const data = msg.data || {};
  Object.entries(data).forEach(([asset, candles]) => {
    if (!candles || !candles.length) return;
    if (window.charts && window.charts[asset]) {
      // Feed all historical candles to the chart at once
      if (typeof window.charts[asset].setHistory === 'function') {
        window.charts[asset].setHistory(candles);
      } else {
        // Fallback: feed one by one
        candles.forEach(c => window.charts[asset].update(c));
      }
    }
  });
}

function applyTrade(msg) {
  updateStats({ trades: msg.trades, wins: msg.wins, pnl: msg.total_pnl });
  if (msg.balance) {
    document.getElementById('balance-val').textContent =
      `$${msg.balance.toLocaleString('en', { minimumFractionDigits: 2 })}`;
  }
  if (msg.entry) addTradeToHistory(msg.entry);

  const color = msg.pnl >= 0 ? 'success' : 'error';
  const sign  = msg.pnl >= 0 ? '+' : '';
  showToast(`${msg.action} ${msg.asset}  ${sign}$${msg.pnl.toFixed(2)}`, color);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(s) {
  if (s.trades !== undefined) {
    document.getElementById('stat-trades').textContent = s.trades;
    document.getElementById('stat-pnl').textContent =
      `${s.pnl >= 0 ? '+' : ''}$${(s.pnl || 0).toFixed(2)}`;
    document.getElementById('stat-pnl').className =
      'stat-val ' + (s.pnl >= 0 ? 'green' : 'red');

    const wr = s.trades > 0 ? (s.wins / s.trades * 100).toFixed(1) : '0.0';
    document.getElementById('stat-wr').textContent = `${wr}%`;
    document.getElementById('stat-wr').className =
      'stat-val ' + (parseFloat(wr) >= 55 ? 'green' : 'red');
  }
}

// ── Trade history ─────────────────────────────────────────────────────────────
function addTradeToHistory(entry) {
  const list = document.getElementById('trade-list');
  const item = document.createElement('div');
  item.className = `trade-item ${entry.win ? 'win' : 'loss'}`;
  const sign = entry.pnl >= 0 ? '+' : '';
  item.innerHTML = `
    <span class="trade-time">${entry.time || '--:--'}</span>
    <span class="trade-action ${entry.action.toLowerCase()}">${entry.action}</span>
    <span class="trade-asset">${(entry.asset || '').replace(' (OTC)', '')}</span>
    <span class="trade-pnl ${entry.pnl >= 0 ? 'pos' : 'neg'}">${sign}$${entry.pnl.toFixed(2)}</span>
  `;
  list.insertBefore(item, list.firstChild);

  // Keep max 50 items
  while (list.children.length > 50) list.removeChild(list.lastChild);
}

// ── Settings ──────────────────────────────────────────────────────────────────
function applySettings(s) {
  if (s.timeframe) document.getElementById('set-tf').value      = s.timeframe;
  if (s.duration)  document.getElementById('set-dur').value     = s.duration;
  if (s.amount)    document.getElementById('set-amount').value  = s.amount;
  updateSettingsBadge(s.timeframe || '─', s.duration || '─');
}

function updateSettingsBadge(tf, dur) {
  document.getElementById('settings-badge').textContent = `TF: ${tf} | DUR: ${dur}`;
}

function saveSettings() {
  const tf  = document.getElementById('set-tf').value;
  const dur = document.getElementById('set-dur').value;
  const amt = document.getElementById('set-amount').value;
  send({ type: 'settings', timeframe: tf, duration: dur, amount: parseFloat(amt) });
  updateSettingsBadge(tf, dur);
  showToast('Settings saved', 'success');
}

// ── AI toggle ─────────────────────────────────────────────────────────────────
function toggleAI() {
  state.aiRunning = !state.aiRunning;
  setAiState(state.aiRunning);
  send({ type: 'ai', running: state.aiRunning });
}

function setAiState(running) {
  state.aiRunning = running;
  const btn = document.getElementById('btn-ai');
  btn.textContent = running ? '⏹  STOP AI' : '▶  START AI';
  btn.className   = running ? 'running' : '';
}

// ── Manual trade ──────────────────────────────────────────────────────────────
function manualTrade(asset, direction) {
  send({ type: 'trade', asset, direction });
  showToast(`${direction.toUpperCase()} ${asset} sent…`, 'info');
}

// ── Sidebar tabs ──────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
}

// ── Checks modal ──────────────────────────────────────────────────────────────
function showChecksModal(results) {
  const body = document.getElementById('modal-body');
  body.innerHTML = '';
  results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'check-row';
    row.innerHTML = `
      <span class="check-icon">${r.passed ? '✅' : '❌'}</span>
      <div>
        <div class="check-name">${r.name}</div>
        ${!r.passed && r.message ? `<div class="check-hint">→ ${r.message}</div>` : ''}
      </div>
    `;
    body.appendChild(row);
  });
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(text, level = 'info') {
  setStatus(text, level);

  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${level}`;
  toast.textContent = text;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

// ── Status bar ────────────────────────────────────────────────────────────────
function setStatus(msg, level = 'info') {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  const colors = { success: 'var(--green)', error: 'var(--red)',
                   warn: 'var(--gold)', info: 'var(--text-muted)' };
  el.style.color = colors[level] || colors.info;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatPrice(asset, price) {
  // JPY pairs have 3 decimal places; others have 5
  const decimals = asset.includes('JPY') ? 3 : 5;
  return price.toFixed(decimals);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Init chart instances (defined in charts.js)
  window.charts = {};
  OTC_ASSETS.forEach(asset => {
    const container = document.querySelector(
      `[data-asset="${CSS.escape(asset)}"] .chart-container`
    );
    if (container) {
      window.charts[asset] = createMiniChart(container, asset);
    }
  });

  connectWS();
  setStatus('กำลังเชื่อมต่อ…', 'info');

  // Auto-ping keepalive every 25s
  setInterval(() => send({ type: 'ping' }), 25000);
});
