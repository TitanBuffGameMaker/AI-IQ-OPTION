/**
 * Mini candlestick chart using TradingView Lightweight Charts.
 * Falls back to canvas bar chart if LWC fails to load.
 */

'use strict';

/**
 * Create a chart for one asset panel.
 * Returns an object with an `update(candle)` method.
 */
function createMiniChart(container, asset) {
  // Try TradingView Lightweight Charts first (loaded via CDN in HTML)
  if (typeof LightweightCharts !== 'undefined') {
    return _createLWChart(container, asset);
  }
  // Fallback: canvas bar chart
  return _createCanvasChart(container, asset);
}

// ── Lightweight Charts (TradingView) ─────────────────────────────────────────
function _createLWChart(container, asset) {
  const isJPY = asset.includes('JPY');

  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth  || 300,
    height: container.clientHeight || 120,
    layout: {
      background:  { color: '#1c2235' },
      textColor:   '#8892b0',
      fontSize:    10,
      fontFamily:  "'Inter', sans-serif",
    },
    grid: {
      vertLines:  { color: '#2a3150' },
      horzLines:  { color: '#2a3150' },
    },
    crosshair: {
      vertLine:   { color: '#4a5568', style: 3 },
      horzLine:   { color: '#4a5568', style: 3 },
    },
    rightPriceScale: {
      borderColor: '#2a3150',
      scaleMargins: { top: 0.1, bottom: 0.1 },
    },
    timeScale: {
      borderColor:     '#2a3150',
      timeVisible:     true,
      secondsVisible:  false,
    },
    handleScroll: false,
    handleScale:  false,
  });

  const series = chart.addCandlestickSeries({
    upColor:          '#00c87a',
    downColor:        '#f23645',
    borderUpColor:    '#00c87a',
    borderDownColor:  '#f23645',
    wickUpColor:      '#00804d',
    wickDownColor:    '#8b1e28',
    priceFormat: {
      type:      'price',
      precision: isJPY ? 3 : 5,
      minMove:   isJPY ? 0.001 : 0.00001,
    },
  });

  // Add baseline with dummy candles so chart is not empty
  const now  = Math.floor(Date.now() / 1000);
  const stub = [];
  for (let i = 29; i >= 0; i--) {
    stub.push({ time: now - i * 60, open: 1, high: 1, low: 1, close: 1 });
  }
  series.setData(stub);

  // Responsive resize
  const ro = new ResizeObserver(() => {
    chart.resize(container.clientWidth, container.clientHeight);
  });
  ro.observe(container);

  return {
    /** Feed a single new candle (real-time tick) */
    update(candle) {
      series.update(candle);
    },
    /** Load full OTC candle history from IQ Option in one shot */
    setHistory(candles) {
      if (!candles || !candles.length) return;
      // Lightweight Charts requires candles sorted by time asc, unique times
      const seen  = new Set();
      const clean = candles
        .filter(c => { if (seen.has(c.time)) return false; seen.add(c.time); return true; })
        .sort((a, b) => a.time - b.time);
      series.setData(clean);
    },
  };
}

// ── Canvas fallback chart ─────────────────────────────────────────────────────
function _createCanvasChart(container, asset) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:100%;display:block;';
  container.appendChild(canvas);

  const ctx    = canvas.getContext('2d');
  const candles = [];
  const MAX    = 40;

  function resize() {
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    draw();
  }

  function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (candles.length < 2) return;

    const hi  = Math.max(...candles.map(c => c.high));
    const lo  = Math.min(...candles.map(c => c.low));
    const rng = hi - lo || 1e-8;

    const bw = Math.max(4, (W / candles.length) - 2);

    candles.forEach((c, i) => {
      const x      = i * (bw + 2) + 1;
      const yOpen  = H - ((c.open  - lo) / rng) * (H - 8) - 4;
      const yClose = H - ((c.close - lo) / rng) * (H - 8) - 4;
      const yHigh  = H - ((c.high  - lo) / rng) * (H - 8) - 4;
      const yLow   = H - ((c.low   - lo) / rng) * (H - 8) - 4;
      const bull   = c.close >= c.open;
      const color  = bull ? '#00c87a' : '#f23645';

      // Wick
      ctx.strokeStyle = bull ? '#00804d' : '#8b1e28';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(x + bw / 2, yHigh);
      ctx.lineTo(x + bw / 2, yLow);
      ctx.stroke();

      // Body
      ctx.fillStyle = color;
      ctx.fillRect(
        x,
        Math.min(yOpen, yClose),
        bw,
        Math.max(1, Math.abs(yClose - yOpen)),
      );
    });

    // Last price line
    const last = candles[candles.length - 1];
    const ly   = H - ((last.close - lo) / rng) * (H - 8) - 4;
    ctx.strokeStyle = '#ffc107';
    ctx.lineWidth   = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, ly);
    ctx.lineTo(W, ly);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  new ResizeObserver(resize).observe(container);
  resize();

  return {
    update(candle) {
      candles.push(candle);
      if (candles.length > MAX) candles.shift();
      draw();
    },
    setHistory(history) {
      candles.length = 0;
      history.slice(-MAX).forEach(c => candles.push(c));
      draw();
    },
  };
}
