/* ==========================================
   AI FORECAST DASHBOARD - JAVASCRIPT
   ========================================== */

const API_BASE = '';

// ==========================================
// TAB NAVIGATION
// ==========================================

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Lazy load data for the tab
    if (tabName === 'forecast') loadForecastTab();
    if (tabName === 'restaurants') loadRestaurants();
    if (tabName === 'trends') loadTrends();
    if (tabName === 'models') loadModels();
    if (tabName === 'alerts') loadAlerts();
}

// ==========================================
// DATA LOADING
// ==========================================

async function fetchAPI(endpoint) {
    try {
        const res = await fetch(`${API_BASE}/api/${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API Error (${endpoint}):`, err);
        return null;
    }
}

async function loadAll() {
    setStatus('loading');

    try {
        await Promise.all([
            loadOverview(),
            loadWeekday(),
            loadHourly(),
        ]);
        setStatus('online');
    } catch (err) {
        setStatus('error');
    }
}

function setStatus(state) {
    const dot = document.querySelector('#status-dot .dot');
    const text = document.getElementById('status-text');

    dot.className = 'dot';
    if (state === 'online') {
        dot.classList.add('online');
        text.textContent = 'Connected';
    } else if (state === 'error') {
        dot.classList.add('error');
        text.textContent = 'Error';
    } else {
        text.textContent = 'Loading...';
    }
}

// ==========================================
// OVERVIEW
// ==========================================

async function loadOverview() {
    const data = await fetchAPI('overview');
    if (!data || data.error) {
        showEmpty('kpi-grid');
        return;
    }

    document.getElementById('system-date').textContent = data.date || '--';

    const o = data.overall || {};
    setValue('v-mae', formatNum(o.MAE));
    setValue('v-mape', formatNum(o.MAPE));
    setValue('v-hit', formatNum(o.Hit_Rate));
    setValue('v-bias', formatNum(Math.abs(o.Bias || 0)));
    setValue('v-bias-dir', (o.Bias || 0) >= 0 ? '↑ over-predict' : '↓ under-predict');
    setValue('v-restaurants', data.restaurants || 0);
    setValue('v-samples', formatK(o.N_samples));

    // Color code KPI cards
    const mapeCard = document.getElementById('kpi-mape');
    mapeCard.classList.remove('good', 'bad');
    if (o.MAPE < 25) mapeCard.classList.add('good');
    else if (o.MAPE > 50) mapeCard.classList.add('bad');

    const hitCard = document.getElementById('kpi-hit');
    hitCard.classList.remove('good', 'bad');
    if (o.Hit_Rate > 70) hitCard.classList.add('good');
    else if (o.Hit_Rate < 50) hitCard.classList.add('bad');

    // Rolling accuracy
    const rolling = data.rolling || {};
    const rgrid = document.getElementById('rolling-grid');
    rgrid.innerHTML = '';

    const windowOrder = ['7d', '14d', '30d'];
    const windowLabels = { '7d': '7 Days', '14d': '14 Days', '30d': '30 Days' };

    for (const w of windowOrder) {
        const m = rolling[w] || {};
        if (!m.MAE && m.MAE !== 0) continue;

        const el = document.createElement('div');
        el.className = 'rolling-item';
        el.innerHTML = `
            <div class="rolling-title">${windowLabels[w] || w}</div>
            <div class="rolling-metrics">
                <div class="rolling-metric">
                    <span class="label">MAE</span>
                    <span class="value">${formatNum(m.MAE)}</span>
                </div>
                <div class="rolling-metric">
                    <span class="label">MAPE</span>
                    <span class="value">${formatNum(m.MAPE)}%</span>
                </div>
                <div class="rolling-metric">
                    <span class="label">Hit Rate</span>
                    <span class="value">${formatNum(m.Hit_Rate)}%</span>
                </div>
                <div class="rolling-metric">
                    <span class="label">Bias</span>
                    <span class="value">${formatNum(m.Bias)}</span>
                </div>
            </div>
        `;
        rgrid.appendChild(el);
    }

    if (rgrid.children.length === 0) {
        rgrid.innerHTML = '<div class="empty-state"><div class="empty-icon">📊</div><p>No rolling data available</p></div>';
    }
}

// ==========================================
// WEEKDAY CHART
// ==========================================

async function loadWeekday() {
    const data = await fetchAPI('weekday');
    const container = document.getElementById('weekday-chart');

    if (!data || !data.length) {
        container.innerHTML = '<div class="empty-state"><p>No weekday data</p></div>';
        return;
    }

    // Sort by weekday order
    const order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    data.sort((a, b) => order.indexOf(a.Weekday) - order.indexOf(b.Weekday));

    const maxMape = Math.max(...data.map(d => d.MAPE || 0), 1);

    container.innerHTML = data.map(d => {
        const pct = Math.min(((d.MAPE || 0) / maxMape) * 100, 100);
        const fillClass = d.MAPE > 50 ? 'danger' : d.MAPE > 30 ? 'warning' : 'success';
        const shortDay = d.Weekday ? d.Weekday.substring(0, 3) : '?';

        return `
            <div class="bar-row">
                <div class="bar-label">${shortDay}</div>
                <div class="bar-track">
                    <div class="bar-fill ${fillClass}" style="width: ${pct}%">
                        ${formatNum(d.MAPE)}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ==========================================
// HOURLY CHART
// ==========================================

async function loadHourly() {
    const data = await fetchAPI('hourly');
    const container = document.getElementById('hourly-chart');

    if (!data || !data.length) {
        container.innerHTML = '<div class="empty-state"><p>No hourly data</p></div>';
        return;
    }

    data.sort((a, b) => a.Hour - b.Hour);
    const maxMape = Math.max(...data.map(d => d.MAPE || 0), 1);

    container.innerHTML = data.map(d => {
        const pct = Math.min(((d.MAPE || 0) / maxMape) * 100, 100);
        const fillClass = d.MAPE > 50 ? 'danger' : d.MAPE > 30 ? 'warning' : 'success';

        return `
            <div class="bar-row">
                <div class="bar-label">${d.Hour}:00</div>
                <div class="bar-track">
                    <div class="bar-fill ${fillClass}" style="width: ${pct}%">
                        ${formatNum(d.MAPE)}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ==========================================
// RESTAURANTS TABLE
// ==========================================

let restaurantData = [];

async function loadRestaurants() {
    const data = await fetchAPI('restaurants');
    const tbody = document.getElementById('restaurant-tbody');

    if (!data || !data.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No restaurant data</td></tr>';
        return;
    }

    restaurantData = data;
    renderRestaurantTable(data);
}

function renderRestaurantTable(data) {
    const tbody = document.getElementById('restaurant-tbody');

    tbody.innerHTML = data.map(r => {
        const mape = r.MAPE || 0;
        let statusClass, statusText;

        if (mape <= 20) { statusClass = 'good'; statusText = '✅ Good'; }
        else if (mape <= 40) { statusClass = 'warning'; statusText = '⚠️ Fair'; }
        else { statusClass = 'danger'; statusText = '❌ Poor'; }

        const biasArrow = (r.Bias || 0) >= 0 ? '↑' : '↓';

        return `
            <tr>
                <td><strong>${r.Restaurant_Code || ''}</strong></td>
                <td>${formatNum(r.MAE)}</td>
                <td>${formatNum(r.MAPE)}%</td>
                <td>${biasArrow} ${formatNum(Math.abs(r.Bias || 0))}</td>
                <td>${formatNum(r.Hit_Rate)}%</td>
                <td>${formatK(r.N_samples)}</td>
                <td><span class="status-badge ${statusClass}">${statusText}</span></td>
            </tr>
        `;
    }).join('');
}

function filterRestaurants() {
    const query = document.getElementById('restaurant-search').value.toUpperCase();
    const filtered = restaurantData.filter(r =>
        (r.Restaurant_Code || '').toUpperCase().includes(query)
    );
    renderRestaurantTable(filtered);
}

// ==========================================
// TRENDS
// ==========================================

async function loadTrends() {
    const [daily, history] = await Promise.all([
        fetchAPI('daily'),
        fetchAPI('history'),
    ]);

    renderDailyTrend(daily);
    renderHistory(history);
}

function renderDailyTrend(data) {
    const container = document.getElementById('daily-trend-chart');

    if (!data || !data.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📉</div><p>No daily trend data</p></div>';
        return;
    }

    // Show MAPE per day as bars
    const sorted = data.sort((a, b) => String(a.Date).localeCompare(String(b.Date)));
    const maxVal = Math.max(...sorted.map(d => d.MAPE || 0), 1);

    container.innerHTML = sorted.map(d => {
        const h = Math.max(((d.MAPE || 0) / maxVal) * 180, 4);
        const color = (d.MAPE || 0) > 40 ? 'var(--accent-red)' :
            (d.MAPE || 0) > 25 ? 'var(--accent-amber)' : 'var(--accent-green)';

        return `
            <div class="trend-bar" style="height: ${h}px; background: linear-gradient(to top, ${color}, ${color}88)">
                <div class="tooltip">${d.Date}<br>MAPE: ${formatNum(d.MAPE)}%<br>MAE: ${formatNum(d.MAE)}</div>
            </div>
        `;
    }).join('');
}

function renderHistory(data) {
    const container = document.getElementById('history-chart');

    if (!data || !data.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📜</div><p>No accuracy history yet. Run the pipeline to start tracking.</p></div>';
        return;
    }

    const maxVal = Math.max(...data.map(d => (d.metrics || {}).MAPE || 0), 1);

    container.innerHTML = data.map(d => {
        const m = d.metrics || {};
        const h = Math.max(((m.MAPE || 0) / maxVal) * 180, 4);

        return `
            <div class="trend-bar" style="height: ${h}px;">
                <div class="tooltip">${d.date}<br>MAPE: ${formatNum(m.MAPE)}%<br>Hit: ${formatNum(m.Hit_Rate)}%</div>
            </div>
        `;
    }).join('');
}

// ==========================================
// MODELS
// ==========================================

async function loadModels() {
    const data = await fetchAPI('comparison');
    const container = document.getElementById('model-grid');

    if (!data || (!data.ensemble && !data.ai_raw)) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">🤖</div><p>No model comparison data available</p></div>';
        return;
    }

    const ens = data.ensemble || {};
    const ai = data.ai_raw || {};
    const winner = data.winner || '';
    const improvement = data.improvement || 0;

    container.innerHTML = `
        <div class="model-card ${winner === 'ensemble' ? 'winner' : ''}">
            <div class="model-name">🧠 Ensemble</div>
            <div class="model-stat">
                <span class="ms-label">MAE</span>
                <span class="ms-value">${formatNum(ens.MAE)}</span>
            </div>
            <div class="model-stat">
                <span class="ms-label">MAPE</span>
                <span class="ms-value">${formatNum(ens.MAPE)}%</span>
            </div>
            ${winner === 'ensemble' ? `<div class="winner-badge">🏆 Winner (+${formatNum(improvement)}%)</div>` : ''}
        </div>
        <div class="model-card ${winner === 'ai_raw' ? 'winner' : ''}">
            <div class="model-name">🤖 AI Raw</div>
            <div class="model-stat">
                <span class="ms-label">MAE</span>
                <span class="ms-value">${formatNum(ai.MAE)}</span>
            </div>
            <div class="model-stat">
                <span class="ms-label">MAPE</span>
                <span class="ms-value">${formatNum(ai.MAPE)}%</span>
            </div>
            ${winner === 'ai_raw' ? `<div class="winner-badge">🏆 Winner (+${formatNum(improvement)}%)</div>` : ''}
        </div>
    `;
}

// ==========================================
// ALERTS
// ==========================================

async function loadAlerts() {
    const [drift, problems] = await Promise.all([
        fetchAPI('drift'),
        fetchAPI('problems'),
    ]);

    renderDrift(drift);
    renderProblems(problems);
}

function renderDrift(data) {
    const container = document.getElementById('drift-content');

    if (!data) {
        container.innerHTML = '<div class="empty-state"><p>No drift data</p></div>';
        return;
    }

    let html = '';

    // Alerts
    const alerts = data.alerts || [];
    if (alerts.length > 0) {
        html += alerts.map(a => {
            const cls = a.level === 'CRITICAL' ? 'critical' : '';
            const icon = a.level === 'CRITICAL' ? '🚨' : '⚠️';
            return `
                <div class="alert-item ${cls}">
                    <div class="alert-icon">${icon}</div>
                    <div class="alert-body">
                        <div class="alert-title">${a.level || 'ALERT'}</div>
                        <div class="alert-desc">${a.message || ''}</div>
                    </div>
                </div>
            `;
        }).join('');
    } else {
        html += `
            <div class="alert-item success">
                <div class="alert-icon">✅</div>
                <div class="alert-body">
                    <div class="alert-title">No Drift Detected</div>
                    <div class="alert-desc">All metrics are within normal range</div>
                </div>
            </div>
        `;
    }

    // Week-over-week changes
    const changes = data.changes || {};
    if (Object.keys(changes).length > 0) {
        html += '<div class="change-grid">';
        for (const [metric, info] of Object.entries(changes)) {
            const pctClass = info.change_pct > 0 ? 'positive' : 'negative';
            const arrow = info.change_pct > 0 ? '↑' : '↓';

            html += `
                <div class="change-item">
                    <div class="change-metric">${metric}</div>
                    <div class="change-values">${formatNum(info.last_week)} → ${formatNum(info.this_week)}</div>
                    <div class="change-pct ${pctClass}">${arrow} ${formatNum(Math.abs(info.change_pct))}%</div>
                </div>
            `;
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

function renderProblems(data) {
    const container = document.getElementById('problems-content');

    if (!data) {
        container.innerHTML = '<div class="empty-state"><p>No problem data</p></div>';
        return;
    }

    let html = '';

    // Needs retune
    const retune = data.needs_retune || [];
    if (retune.length > 0) {
        html += '<h3 style="color: var(--accent-red); margin-bottom: 12px; font-size: 14px;">🔧 Needs Retuning</h3>';
        html += retune.map(r => `
            <div class="alert-item critical">
                <div class="alert-icon">🏪</div>
                <div class="alert-body">
                    <div class="alert-title">${r.Restaurant_Code}</div>
                    <div class="alert-desc">MAPE: ${formatNum(r.MAPE)}% | MAE: ${formatNum(r.MAE)} | Hit Rate: ${formatNum(r.Hit_Rate)}%</div>
                </div>
            </div>
        `).join('');
    }

    // High bias
    const bias = data.high_bias || [];
    if (bias.length > 0) {
        html += '<h3 style="color: var(--accent-amber); margin: 16px 0 12px; font-size: 14px;">⚖️ High Bias</h3>';
        html += bias.slice(0, 10).map(r => {
            const dir = (r.Bias || 0) >= 0 ? 'Over-predicting' : 'Under-predicting';
            return `
                <div class="alert-item">
                    <div class="alert-icon">📊</div>
                    <div class="alert-body">
                        <div class="alert-title">${r.Restaurant_Code} - ${dir}</div>
                        <div class="alert-desc">Bias: ${formatNum(r.Bias)} | MAE: ${formatNum(r.MAE)}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    if (!retune.length && !bias.length) {
        html = `
            <div class="alert-item success">
                <div class="alert-icon">🎉</div>
                <div class="alert-body">
                    <div class="alert-title">All Clear</div>
                    <div class="alert-desc">No problem restaurants detected</div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

// ==========================================
// UTILITIES
// ==========================================

function formatNum(n) {
    if (n === null || n === undefined || isNaN(n)) return '--';
    return Number(n).toLocaleString('en', { maximumFractionDigits: 1 });
}

function formatK(n) {
    if (n === null || n === undefined || isNaN(n)) return '--';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return Number(n).toLocaleString('en');
}

function setValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ==========================================
// FORECAST VIEWER
// ==========================================

let forecastMode = 'summary';  // 'summary' or 'hourly'
let forecastScope = 'future';  // 'future' or 'all'
let restaurantListLoaded = false;

async function loadForecastTab() {
    if (!restaurantListLoaded) {
        await loadRestaurantList();
        restaurantListLoaded = true;
    }
    loadForecastView();
}

async function loadRestaurantList() {
    const codes = await fetchAPI('restaurants/list');
    const select = document.getElementById('forecast-restaurant-select');

    if (!codes || !codes.length) return;

    // Keep first option
    select.innerHTML = '<option value="">-- T\u1ea5t c\u1ea3 --</option>';
    codes.forEach(code => {
        const opt = document.createElement('option');
        opt.value = code;
        opt.textContent = code;
        select.appendChild(opt);
    });
}

function setForecastMode(mode) {
    forecastMode = mode;
    document.getElementById('btn-mode-summary').classList.toggle('active', mode === 'summary');
    document.getElementById('btn-mode-hourly').classList.toggle('active', mode === 'hourly');
    loadForecastView();
}

function setForecastScope(scope) {
    forecastScope = scope;
    document.getElementById('btn-scope-future').classList.toggle('active', scope === 'future');
    document.getElementById('btn-scope-all').classList.toggle('active', scope === 'all');
    loadForecastView();
}

async function loadForecastView() {
    const restaurant = document.getElementById('forecast-restaurant-select').value;

    if (forecastMode === 'summary') {
        await loadForecastSummary(restaurant);
    } else {
        await loadForecastHourly(restaurant);
    }
}

async function loadForecastSummary(restaurant) {
    let url = `forecast/summary?scope=${forecastScope}`;
    if (restaurant) url += `&restaurant=${restaurant}`;

    const data = await fetchAPI(url);
    const thead = document.getElementById('forecast-thead');
    const tbody = document.getElementById('forecast-tbody');
    const empty = document.getElementById('forecast-empty');

    thead.innerHTML = `
        <tr>
            <th>Restaurant</th>
            <th>Date</th>
            <th>Predicted Total</th>
            <th>Actual Total</th>
            <th>Accuracy</th>
            <th>Confidence</th>
            <th>Hours</th>
        </tr>
    `;

    if (!data || !data.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }

    empty.style.display = 'none';

    tbody.innerHTML = data.slice(0, 200).map(r => {
        const hasActual = r.actual_total !== null && r.actual_total !== undefined;
        const actualStr = hasActual ? formatNum(r.actual_total) : '<span style="color:var(--text-muted)">\u2014</span>';

        let accStr = '<span style="color:var(--text-muted)">Pending</span>';
        if (r.accuracy !== undefined) {
            const accClass = r.accuracy >= 80 ? 'good' : r.accuracy >= 60 ? 'warning' : 'danger';
            accStr = `<span class="status-badge ${accClass}">${formatNum(r.accuracy)}%</span>`;
        }

        const confHtml = renderConfidence(r.confidence);

        return `
            <tr>
                <td><strong>${r.restaurant}</strong></td>
                <td>${r.date}</td>
                <td style="font-weight:600; color:var(--accent-cyan)">${formatNum(r.predicted_total)}</td>
                <td>${actualStr}</td>
                <td>${accStr}</td>
                <td>${confHtml}</td>
                <td>${r.hours}</td>
            </tr>
        `;
    }).join('');
}

async function loadForecastHourly(restaurant) {
    let url = `forecast/upcoming`;
    if (restaurant) url += `?restaurant=${restaurant}`;

    const data = await fetchAPI(url);
    const thead = document.getElementById('forecast-thead');
    const tbody = document.getElementById('forecast-tbody');
    const empty = document.getElementById('forecast-empty');

    thead.innerHTML = `
        <tr>
            <th>Restaurant</th>
            <th>Date</th>
            <th>Weekday</th>
            <th>Hour</th>
            <th>Predicted Guests</th>
            <th>Confidence</th>
            <th>Strategy</th>
        </tr>
    `;

    if (!data || !data.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }

    empty.style.display = 'none';

    tbody.innerHTML = data.slice(0, 500).map(r => {
        const confHtml = renderConfidence(r.confidence);
        const holiday = r.is_holiday ? ' \ud83c\udf89' : '';

        return `
            <tr>
                <td><strong>${r.restaurant}</strong></td>
                <td>${r.date}${holiday}</td>
                <td>${r.weekday || ''}</td>
                <td>${r.hour}:00</td>
                <td style="font-weight:700; color:var(--accent-cyan); font-size:16px;">${formatNum(r.predicted)}</td>
                <td>${confHtml}</td>
                <td><span class="badge">${r.strategy || ''}</span></td>
            </tr>
        `;
    }).join('');
}

function renderConfidence(conf) {
    if (conf === null || conf === undefined) {
        return '<span style="color:var(--text-muted)">\u2014</span>';
    }
    const pct = Math.round(conf * 100);
    const cls = pct >= 70 ? '' : pct >= 40 ? 'medium' : 'low';
    return `
        <div class="confidence-bar">
            <div class="confidence-track">
                <div class="confidence-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <span style="font-size:12px;color:var(--text-secondary)">${pct}%</span>
        </div>
    `;
}

// ==========================================
// INIT
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    loadAll();

    // Auto-refresh every 5 minutes
    setInterval(loadAll, 5 * 60 * 1000);
});
