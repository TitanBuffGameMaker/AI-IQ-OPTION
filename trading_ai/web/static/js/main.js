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
  'EUR/JPY (OTC)',
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

    case 'open_trade':
      if (msg.entry) addActiveOrder(msg.entry);
      break;

    case 'close_trade':
      if (msg.order_id != null) removeActiveOrder(msg.order_id);
      break;

    case 'stats_update':
      updateStats(msg.stats || {});
      if (msg.history && msg.history.length) {
        document.getElementById('trade-list').innerHTML = '';
        const empty = document.getElementById('empty-history');
        if (empty) empty.style.display = 'none';
        msg.history.forEach(addTradeToHistory);
      }
      break;

    case 'status':
      setStatus(msg.message, msg.level || 'info');
      break;

    case 'training':
      applyTraining(msg);
      break;

    case 'strategy':
      applyStrategy(msg);
      break;

    case 'market_mode':
      if (msg.mode) _applyMarketModeUI(msg.mode);
      break;

    case 'capital_guard':
      applyCapitalGuard(msg);
      break;

    case 'chat_ai':
      appendChatBubble('ai', msg.message, msg.time || '');
      break;

    case 'chat_history':
      if (Array.isArray(msg.entries)) {
        msg.entries.forEach(e => appendChatBubble(e.role, e.message, e.time || '', true));
        scrollChatToBottom();
      }
      break;

    case 'log':
      addLogEntry(msg);
      break;

    case 'log_history':
      if (Array.isArray(msg.entries)) {
        msg.entries.forEach(e => addLogEntry(e, /*batch=*/true));
        const ll = document.getElementById('log-list');
        if (ll) ll.scrollTop = ll.scrollHeight;
      }
      break;

    case 'otp_required':
      showOTPModal(msg.message);
      break;

    case 'ssid_required':        // legacy — redirect to new modal
    case 'login_required':
      showLoginModal(msg.reason || msg.message || '');
      break;
  }
}

// ── Login Modal (Email/Password + SSID) ───────────────────────────────────────
function showLoginModal(reason) {
  const overlay = document.getElementById('login-overlay');
  if (!overlay) return;
  const reasonEl = document.getElementById('login-reason');
  if (reasonEl && reason) {
    // Show human-readable hint based on error content
    if (reason.toLowerCase().includes('exceeded') || reason.toLowerCase().includes('number of requests')) {
      reasonEl.textContent = '⏳ IQ Option บล็อก login ชั่วคราว — ลอง SSID Token แทน';
    } else if (reason.toLowerCase().includes('10060') || reason.toLowerCase().includes('timed out')) {
      reasonEl.textContent = '🌐 เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ — ตรวจสอบ internet หรือลอง login ใหม่';
    } else if (reason) {
      reasonEl.textContent = reason.substring(0, 120);
    }
  }
  document.getElementById('login-error').style.display = 'none';
  showLoginTab('creds');   // default to email/password tab
  overlay.style.display = 'flex';
  setTimeout(() => document.getElementById('login-email').focus(), 150);
}

function showLoginTab(tab) {
  const isCreds = tab === 'creds';
  document.getElementById('login-panel-creds').style.display = isCreds ? 'block' : 'none';
  document.getElementById('login-panel-ssid').style.display  = isCreds ? 'none'  : 'block';

  const btnC = document.getElementById('tab-btn-creds');
  const btnS = document.getElementById('tab-btn-ssid');
  if (isCreds) {
    btnC.style.cssText += ';border:2px solid #2962ff;background:#2962ff22;color:#5c8fff;';
    btnS.style.cssText += ';border:1px solid #444;background:transparent;color:#666;';
    setTimeout(() => document.getElementById('login-email').focus(), 80);
  } else {
    btnS.style.cssText += ';border:2px solid #00c87a;background:#00c87a22;color:#00c87a;';
    btnC.style.cssText += ';border:1px solid #444;background:transparent;color:#666;';
    setTimeout(() => document.getElementById('ssid-input').focus(), 80);
  }
}

function submitLoginCreds() {
  const email    = (document.getElementById('login-email').value    || '').trim();
  const password = (document.getElementById('login-password').value || '').trim();
  const errEl    = document.getElementById('login-error');
  if (!email || !email.includes('@')) {
    errEl.textContent = 'กรุณากรอก Email ให้ถูกต้อง';
    errEl.style.display = 'block';
    return;
  }
  if (!password) {
    errEl.textContent = 'กรุณากรอก Password';
    errEl.style.display = 'block';
    return;
  }
  send({ type: 'credentials', email, password });
  document.getElementById('login-overlay').style.display = 'none';
  setStatus('กำลังเชื่อมต่อ…', 'info');
}

function submitSSID() {
  const ssid  = (document.getElementById('ssid-input').value || '').trim();
  const errEl = document.getElementById('login-error');
  if (!ssid || ssid.length < 10) {
    errEl.textContent = 'SSID ไม่ถูกต้อง (ต้องยาวกว่า 10 ตัวอักษร)';
    errEl.style.display = 'block';
    return;
  }
  send({ type: 'otp', code: ssid });
  document.getElementById('login-overlay').style.display = 'none';
  setStatus('กำลังเชื่อมต่อด้วย SSID…', 'info');
}

// ── OTP Modal ─────────────────────────────────────────────────────────────────
function showOTPModal(message) {
  const overlay = document.getElementById('otp-overlay');
  if (!overlay) return;
  if (message) document.getElementById('otp-msg').textContent = message;
  document.getElementById('otp-error').style.display = 'none';
  document.getElementById('otp-input').value = '';
  overlay.style.display = 'flex';
  setTimeout(() => document.getElementById('otp-input').focus(), 100);
}

function submitOTP() {
  const code = document.getElementById('otp-input').value.trim();
  if (code.length !== 5 || !/^\d+$/.test(code)) {
    document.getElementById('otp-error').style.display = 'block';
    document.getElementById('otp-error').textContent = 'OTP ต้องเป็นตัวเลข 5 หลัก';
    return;
  }
  send({ type: 'otp', code });
  document.getElementById('otp-overlay').style.display = 'none';
  setStatus('ส่ง OTP แล้ว กำลังยืนยัน…', 'info');
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
  applyConnection({ connected: msg.connected, balance: msg.balance, account: msg.account || '' });
  updateStats(msg.stats || {});
  if (msg.checks && msg.checks.length) {
    applyChecks({ results: msg.checks, all_passed: !!msg.all_passed });
  }
  const history = msg.history || [];
  if (history.length === 0) {
    const empty = document.getElementById('empty-history');
    if (empty) empty.style.display = '';
  } else {
    history.forEach(addTradeToHistory);
  }
  // Replay any orders that were already open when this client connected
  (msg.open_orders || []).forEach(addActiveOrder);
  applySettings(msg.settings || {});
  setAiState(msg.ai_running || false);
  _applyMarketModeUI(msg.market_mode || 'OTC');
  if (msg.capital_guard) applyCapitalGuard(msg.capital_guard);
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

// ── Strategy status ───────────────────────────────────────────────────────────
function applyStrategy(msg) {
  // Update active strategy name
  const stratEl = document.getElementById('active-strategy');
  if (stratEl) {
    stratEl.textContent = msg.name || '-';
  }

  // Update strategy win rate
  const stratWrEl = document.getElementById('strategy-win-rate');
  if (stratWrEl) {
    const wr = (msg.win_rate || 0) * 100;
    stratWrEl.textContent = wr.toFixed(1) + '%';
    stratWrEl.style.color = wr >= 55 ? 'var(--green)' : wr < 45 ? 'var(--red)' : 'var(--text)';
  }

  // Update improvement tips
  const tipsEl = document.getElementById('improvement-tips');
  if (tipsEl && msg.tips && msg.tips.length) {
    tipsEl.innerHTML = msg.tips
      .map(t => `<div class="tip-item">💡 ${t}</div>`)
      .join('');
  }

  // Show/hide pause warning
  const pauseEl = document.getElementById('pause-warning');
  if (pauseEl) {
    pauseEl.style.display = msg.should_pause ? 'block' : 'none';
  }
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
  const list  = document.getElementById('trade-list');
  const empty = document.getElementById('empty-history');
  if (empty) empty.style.display = 'none';
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
  updateClosedCount();
}

function updateClosedCount() {
  const list  = document.getElementById('trade-list');
  const count = document.getElementById('closed-count');
  if (count && list) count.textContent = list.children.length;
}

// ── Active orders (open positions with live countdown) ────────────────────────
const _activeOrders = new Map();   // order_id → {entry, el}

function addActiveOrder(entry) {
  if (!entry || entry.order_id == null) return;
  const list = document.getElementById('active-list');
  const oid  = entry.order_id;
  if (_activeOrders.has(oid)) return;   // already shown

  const action = (entry.action || '').toUpperCase();
  const cls    = action === 'BUY' || action === 'CALL' ? 'buy'  :
                 action === 'SELL'|| action === 'PUT'  ? 'sell' : '';
  const arrow  = cls === 'buy' ? '▲' : cls === 'sell' ? '▼' : '•';
  const tag    = entry.manual ? '🖐 Manual' : '🤖 AI';

  const el = document.createElement('div');
  el.className   = `active-order ${cls}`;
  el.dataset.oid = String(oid);
  el.innerHTML = `
    <span class="ao-time">${entry.open_time || '--:--'}</span>
    <span class="ao-main">
      <span class="ao-asset">${(entry.asset || '').replace(' (OTC)', '')} <span class="ao-action ${cls}">${arrow} ${action}</span></span>
      <span class="ao-info">$${(entry.amount ?? 0).toFixed(2)} · ${tag}</span>
    </span>
    <span class="ao-countdown" data-expiry="${entry.expiry_ts || 0}">
      <span class="ao-cd-main">--:--</span>
      <span class="ao-cd-sub">เหลือ</span>
    </span>
  `;
  list.insertBefore(el, list.firstChild);
  _activeOrders.set(oid, { entry, el });

  const empty = document.getElementById('empty-active');
  if (empty) empty.style.display = 'none';
  updateActiveCount();
  tickCountdown(el);  // first paint immediately
}

function removeActiveOrder(oid) {
  if (oid == null) return;
  const rec = _activeOrders.get(oid);
  if (!rec) return;
  rec.el.remove();
  _activeOrders.delete(oid);
  if (_activeOrders.size === 0) {
    const empty = document.getElementById('empty-active');
    if (empty) empty.style.display = '';
  }
  updateActiveCount();
}

function updateActiveCount() {
  const c = document.getElementById('active-count');
  if (c) c.textContent = _activeOrders.size;
}

function tickCountdown(el) {
  const cd   = el.querySelector('.ao-countdown');
  if (!cd) return;
  const exp  = parseFloat(cd.dataset.expiry) || 0;
  const left = Math.floor(exp - Date.now() / 1000);
  const main = cd.querySelector('.ao-cd-main');
  const sub  = cd.querySelector('.ao-cd-sub');
  if (left > 0) {
    const m = Math.floor(left / 60), s = left % 60;
    main.textContent = `${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
    sub.textContent  = 'เหลือ';
    el.classList.remove('expired');
  } else {
    main.textContent = '⏳';
    sub.textContent  = 'กำลังรอผล…';
    el.classList.add('expired');
  }
}

// One global tick loop for all active orders
setInterval(() => {
  _activeOrders.forEach(rec => tickCountdown(rec.el));
}, 1000);



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

// ── Market Mode toggle ────────────────────────────────────────────────────────
function setMarketMode(mode) {
  send({ type: 'settings', market_mode: mode });
  _applyMarketModeUI(mode);
}

function _applyMarketModeUI(mode) {
  const otcBtn  = document.getElementById('btn-mode-otc');
  const realBtn = document.getElementById('btn-mode-real');
  const desc    = document.getElementById('mode-desc');
  const badge   = document.getElementById('market-mode-badge');
  const icon    = document.getElementById('market-mode-icon');
  const text    = document.getElementById('market-mode-text');

  if (!otcBtn) return;

  if (mode === 'OTC') {
    otcBtn.className  = 'mode-btn active otc';
    realBtn.className = 'mode-btn';
    desc.textContent  = '🎲 OTC — pattern learning, news/calendar disabled (OTC ไม่เกี่ยวกับข่าว)';
    if (badge) { badge.className = 'nav-badge otc'; icon.textContent = '🎲'; text.textContent = 'OTC'; }
  } else {
    otcBtn.className  = 'mode-btn';
    realBtn.className = 'mode-btn active real';
    desc.textContent  = '📈 Real Market — news, calendar, sentiment active (ดูแนวโน้มตลาดจริง)';
    if (badge) { badge.className = 'nav-badge real'; icon.textContent = '📈'; text.textContent = 'Real'; }
  }
}

// ── Capital Guard ─────────────────────────────────────────────────────────────
function applyCapitalGuard(msg) {
  const panel = document.getElementById('capital-guard-panel');
  if (!panel) return;

  const isReal = (msg.account_type === 'REAL');
  panel.style.display = isReal ? 'block' : 'none';
  if (!isReal) return;

  // Daily P&L
  const pnl = msg.session_pnl || 0;
  const pnlEl = document.getElementById('cg-daily-pnl');
  if (pnlEl) {
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
    pnlEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
  }

  // Kelly bet
  const kellyEl = document.getElementById('cg-kelly-amount');
  if (kellyEl && msg.kelly_amount > 0) kellyEl.textContent = '$' + msg.kelly_amount.toFixed(2);

  // Min confidence (derived from account type label — will be updated via signal messages)
  // Profit bar
  const profitPct = Math.min((msg.profit_pct || 0) / (msg.profit_target || 0.10), 1) * 100;
  const profitBar = document.getElementById('cg-profit-bar');
  const profitTxt = document.getElementById('cg-profit-pct-text');
  if (profitBar) profitBar.style.width = profitPct.toFixed(1) + '%';
  if (profitTxt) profitTxt.textContent = ((msg.profit_pct || 0) * 100).toFixed(1) + '%';

  // Target line position
  const targetLine = document.getElementById('cg-target-line');
  if (targetLine) targetLine.style.left = '100%';   // at 100% of bar = target

  // Loss bar
  const lossPct = Math.min((msg.loss_pct || 0) / (msg.daily_loss_limit || 0.20), 1) * 100;
  const lossBar = document.getElementById('cg-loss-bar');
  const lossTxt = document.getElementById('cg-loss-pct-text');
  if (lossBar) lossBar.style.width = lossPct.toFixed(1) + '%';
  if (lossTxt) lossTxt.textContent = ((msg.loss_pct || 0) * 100).toFixed(1) + '%';

  // Labels
  const targetPctEl = document.getElementById('cg-target-pct');
  const limitPctEl  = document.getElementById('cg-limit-pct');
  if (targetPctEl) targetPctEl.textContent = ((msg.profit_target || 0.10) * 100).toFixed(0) + '%';
  if (limitPctEl)  limitPctEl.textContent  = ((msg.daily_loss_limit || 0.20) * 100).toFixed(0) + '%';

  // Stop banner
  const stopBanner = document.getElementById('cg-stop-banner');
  const stopReason = document.getElementById('cg-stop-reason');
  if (stopBanner && stopReason) {
    stopBanner.style.display = msg.stopped ? 'block' : 'none';
    stopReason.textContent   = msg.stop_reason || '';
  }

  // Input sync
  const lossInput   = document.getElementById('cg-loss-limit-input');
  const targetInput = document.getElementById('cg-target-input');
  if (lossInput && msg.daily_loss_limit) lossInput.value = Math.round(msg.daily_loss_limit * 100);
  if (targetInput && msg.profit_target)  targetInput.value = Math.round(msg.profit_target * 100);
}

function sendCapitalGuardSettings() {
  const lossInput   = document.getElementById('cg-loss-limit-input');
  const targetInput = document.getElementById('cg-target-input');
  send({
    type:               'settings',
    loss_limit_pct:     (parseInt(lossInput?.value || 20) / 100),
    profit_target_pct:  (parseInt(targetInput?.value || 10) / 100),
  });
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

// ── Floating Chat Widget ──────────────────────────────────────────────────────
let _chatOpen    = false;
let _chatUnread  = 0;

function toggleChat() {
  const panel = document.getElementById('chat-panel');
  const btn   = document.getElementById('chat-float-btn');
  if (!panel) return;

  _chatOpen = !_chatOpen;
  panel.style.display = _chatOpen ? 'flex' : 'none';
  btn.textContent = _chatOpen ? '✕' : '💬';

  if (_chatOpen) {
    // Clear unread badge
    _chatUnread = 0;
    const notif = document.getElementById('chat-notif');
    if (notif) notif.style.display = 'none';
    scrollChatToBottom();
    setTimeout(() => document.getElementById('chat-input')?.focus(), 80);
    // Add back emoji to button when closing
    btn.style.fontSize = _chatOpen ? '16px' : '22px';
  } else {
    btn.textContent = '💬';
    btn.style.fontSize = '22px';
  }
}

function sendChatMessage() {
  const input = document.getElementById('chat-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  appendChatBubble('user', text, _chatNow());
  send({ type: 'chat_user', message: text });
}

function appendChatBubble(role, message, time, batch) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  const wrap = document.createElement('div');
  wrap.className = `chat-bubble ${role}`;

  const textEl = document.createElement('div');
  textEl.className = 'chat-text';
  textEl.innerHTML = _chatMarkdown(message);

  const tsEl = document.createElement('div');
  tsEl.className = 'chat-time';
  tsEl.textContent = time || _chatNow();

  wrap.appendChild(textEl);
  wrap.appendChild(tsEl);
  container.appendChild(wrap);

  if (!batch) {
    scrollChatToBottom();
    // Increment unread badge when chat is closed
    if (!_chatOpen && role === 'ai') {
      _chatUnread++;
      const notif = document.getElementById('chat-notif');
      if (notif) {
        notif.textContent = _chatUnread > 9 ? '9+' : _chatUnread;
        notif.style.display = 'flex';
      }
    }
  }
}

function scrollChatToBottom() {
  const container = document.getElementById('chat-messages');
  if (container) {
    // Use requestAnimationFrame to ensure DOM has updated
    requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
  }
}

function _chatNow() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

function _chatMarkdown(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

// Inject greeting bubble once on page load
window.addEventListener('DOMContentLoaded', () => {
  appendChatBubble(
    'ai',
    'สวัสดีครับ! 🤖 ผมคือ AI Trading Brain\n\nถามผมได้ทุกเรื่อง — สถานะ, win rate, กลยุทธ์, ความคิดของผม หรืออะไรก็ได้\n\nผมจะรายงานผลเทรดให้ทราบด้วยโดยอัตโนมัติ',
    _chatNow(),
    true
  );
});

// ── Live Logs ─────────────────────────────────────────────────────────────────
let _logFilter  = 'all';
let _logUnread  = 0;
const _LOG_MAX  = 300;     // max entries to keep in DOM

function setLogFilter(lvl) {
  _logFilter = lvl;
  document.querySelectorAll('.log-filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.lvl === lvl);
  });
  // Show/hide existing entries
  document.querySelectorAll('#log-list .log-entry').forEach(el => {
    el.style.display = (lvl === 'all' || el.dataset.lvl === lvl) ? '' : 'none';
  });
}

function clearLogs() {
  const ll = document.getElementById('log-list');
  if (ll) ll.innerHTML = '';
  _logUnread = 0;
  const badge = document.querySelector('#tab-btn-logs .log-badge-count');
  if (badge) { badge.style.display = 'none'; badge.textContent = '0'; }
}

function addLogEntry(entry, batch) {
  const ll = document.getElementById('log-list');
  if (!ll) return;

  const lvl  = entry.level || 'info';
  const show = (_logFilter === 'all' || _logFilter === lvl);
  const row  = document.createElement('div');
  row.className = `log-entry ${lvl}`;
  row.dataset.lvl = lvl;
  row.style.display = show ? '' : 'none';
  row.innerHTML =
    `<span class="log-time">${entry.time || ''}</span>` +
    `<span class="log-badge">${lvl.toUpperCase()}</span>` +
    `<span class="log-msg">[${entry.name || ''}] ${entry.message || ''}</span>`;
  ll.appendChild(row);

  // Trim old entries
  while (ll.children.length > _LOG_MAX) ll.removeChild(ll.firstChild);

  // Auto-scroll only if we're within 60px of bottom
  if (!batch) {
    const atBottom = ll.scrollHeight - ll.scrollTop - ll.clientHeight < 60;
    if (atBottom) ll.scrollTop = ll.scrollHeight;

    // Unread badge when logs tab is NOT active
    const logsTabActive = document.getElementById('tab-logs')?.classList.contains('active');
    if (!logsTabActive && (lvl === 'warn' || lvl === 'error')) {
      _logUnread++;
      const badge = document.querySelector('#tab-btn-logs .log-badge-count');
      if (badge) { badge.style.display = 'inline-block'; badge.textContent = _logUnread; }
    }
  }
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
