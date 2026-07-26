/**
 * ============================================================
 * TRAFFIC CONGESTION PREDICTION AI - PREMIUM FRONTEND
 * Advanced JavaScript Engine
 * ============================================================
 */

const API_BASE = '';
let autoRefreshInterval = null;
let chartInstances = {};

// ============================================================
// CONGESTION UTILITIES
// ============================================================
const CongestionUtils = {
    levels: ['Low', 'Moderate', 'High', 'Severe'],
    colors: ['#10b981', '#eab308', '#f59e0b', '#dc2626'],
    bgColors: ['rgba(16,185,129,0.15)', 'rgba(234,179,8,0.15)', 'rgba(245,158,11,0.15)', 'rgba(220,38,38,0.15)'],
    
    getSeverityClass(level) {
        return ['congestion-low', 'congestion-moderate', 'congestion-high', 'congestion-severe'][Math.min(Math.floor(level), 3)] || 'congestion-low';
    },
    
    getColor(level) {
        return this.colors[Math.min(Math.floor(level), 3)] || this.colors[0];
    },
    
    getLabel(level) {
        return this.levels[Math.min(Math.floor(level), 3)] || 'Unknown';
    },
    
    getBgColor(level) {
        return this.bgColors[Math.min(Math.floor(level), 3)] || this.bgColors[0];
    }
};

// ============================================================
// FORMATTING UTILITIES
// ============================================================
const FormatUtils = {
    number(num) { return (num || 0).toLocaleString(); },
    percent(num, decimals = 1) { return ((num || 0) * 100).toFixed(decimals) + '%'; },
    date(dateStr) { return new Date(dateStr).toLocaleString(); },
    time(dateStr) { return new Date(dateStr).toLocaleTimeString(); },
    decimal(num, places = 2) { return (num || 0).toFixed(places); },
    kmh(speed) { return Math.round(speed || 0) + ' km/h'; },
    volume(vol) { return this.number(Math.round(vol || 0)); }
};

// ============================================================
// API CLIENT - Premium
// ============================================================
const ApiClient = {
    async get(endpoint) {
        try {
            const res = await fetch(`${API_BASE}${endpoint}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            return await res.json();
        } catch (err) {
            console.error(`API GET ${endpoint} failed:`, err);
            throw err;
        }
    },
    
    async post(endpoint, data) {
        try {
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            return await res.json();
        } catch (err) {
            console.error(`API POST ${endpoint} failed:`, err);
            throw err;
        }
    },
    
    async healthCheck() {
        return this.get('/api/health');
    },
    
    async getModels() {
        return this.get('/api/models');
    },
    
    async predict(data) {
        return this.post('/api/predict', data);
    },
    
    async getSampleData(n = 500) {
        return this.get(`/api/sample-data?n=${n}`);
    },
    
    async getTimeAnalysis() {
        return this.get('/api/time-analysis');
    },
    
    async getWeatherAnalysis() {
        return this.get('/api/weather-analysis');
    },
    
    async getLocationRanking() {
        return this.get('/api/location-ranking');
    },
    
    async getOptimalTimes() {
        return this.get('/api/optimal-times');
    },

    async getTrends() {
        return this.get('/api/trends');
    }
};

// ============================================================
// CHART MANAGER - Premium Plotly Integration
// ============================================================
const ChartManager = {
    darkLayout(title = '', xlabel = '', ylabel = '') {
        return {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'Inter, sans-serif' },
            title: { text: title, font: { color: '#e2e8f0', size: 16, family: 'Inter' } },
            xaxis: { 
                title: xlabel, 
                gridcolor: 'rgba(255,255,255,0.04)',
                zerolinecolor: 'rgba(255,255,255,0.04)',
                tickfont: { size: 11 }
            },
            yaxis: { 
                title: ylabel, 
                gridcolor: 'rgba(255,255,255,0.04)',
                zerolinecolor: 'rgba(255,255,255,0.04)',
                tickfont: { size: 11 }
            },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            hovermode: 'closest',
            showlegend: true,
            legend: { font: { color: '#94a3b8', size: 11 }, orientation: 'h', y: -0.15 }
        };
    },
    
    create(elementId, traces, layout, config = {}) {
        const defaultConfig = { responsive: true, displayModeBar: false, displaylogo: false };
        Plotly.newPlot(elementId, traces, layout, { ...defaultConfig, ...config });
        chartInstances[elementId] = true;
    },
    
    update(elementId, traces, layout) {
        if (document.getElementById(elementId)) {
            Plotly.react(elementId, traces, layout, { responsive: true, displayModeBar: false });
        }
    },
    
    destroy(elementId) {
        if (chartInstances[elementId]) {
            Plotly.purge(elementId);
            delete chartInstances[elementId];
        }
    }
};

// ============================================================
// UI COMPONENTS - Premium
// ============================================================
const UI = {
    showLoading(containerId, message = 'Loading...') {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = `
            <div class="d-flex justify-content-center align-items-center" style="min-height: 250px;">
                <div class="text-center">
                    <div class="spinner-premium mb-3 mx-auto"></div>
                    <div class="text-muted small">${message}</div>
                </div>
            </div>`;
    },
    
    showError(containerId, message = 'Failed to load data', detail = '') {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = `
            <div class="d-flex justify-content-center align-items-center" style="min-height: 250px;">
                <div class="text-center">
                    <div class="text-danger mb-3" style="font-size: 3rem;">
                        <i class="fas fa-exclamation-circle"></i>
                    </div>
                    <div class="fw-semibold mb-1">${message}</div>
                    ${detail ? `<div class="text-muted small">${detail}</div>` : ''}
                </div>
            </div>`;
    },
    
    showToast(message, type = 'success') {
        const colors = { success: '#10b981', error: '#ef4444', warning: '#f59e0b', info: '#3b82f6' };
        const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };
        
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 20px; right: 20px; z-index: 10000;
            background: var(--card-bg); border: 1px solid ${colors[type] || colors.info};
            border-radius: 12px; padding: 1rem 1.5rem;
            display: flex; align-items: center; gap: 0.8rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            animation: slideInRight 0.3s ease;
            max-width: 400px;
        `;
        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info}" style="color: ${colors[type] || colors.info}; font-size: 1.3rem;"></i>
            <span style="color: var(--text); font-size: 0.9rem;">${message}</span>
            <button onclick="this.parentElement.remove()" style="background:none;border:none;color:var(--text-muted);cursor:pointer;margin-left:0.5rem;">
                <i class="fas fa-times"></i>
            </button>`;
        document.body.appendChild(toast);
        setTimeout(() => { if (toast.parentElement) toast.remove(); }, 5000);
    },
    
    updateStat(id, value, animate = true) {
        const el = document.getElementById(id);
        if (!el) return;
        if (animate) {
            el.style.opacity = '0';
            el.style.transform = 'translateY(10px)';
            setTimeout(() => {
                el.textContent = value;
                el.style.transition = 'all 0.5s ease';
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, 100);
        } else {
            el.textContent = value;
        }
    }
};

// ============================================================
// AUTO-REFRESH MANAGER
// ============================================================
const AutoRefresh = {
    start(callback, interval = 15000) {
        this.stop();
        autoRefreshInterval = setInterval(callback, interval);
        const indicator = document.getElementById('autoRefreshIndicator');
        if (indicator) indicator.innerHTML = '<span class="live-dot"></span> Live';
        UI.showToast('Auto-refresh enabled', 'info');
    },
    
    stop() {
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
        const indicator = document.getElementById('autoRefreshIndicator');
        if (indicator) indicator.innerHTML = '<span style="color: var(--text-muted)">●</span> Paused';
    },
    
    toggle(callback, interval = 15000) {
        if (autoRefreshInterval) {
            this.stop();
            UI.showToast('Auto-refresh paused', 'warning');
        } else {
            this.start(callback, interval);
        }
    }
};

// ============================================================
// DASHBOARD DATA LOADER - Premium
// ============================================================
const DashboardLoader = {
    async loadStats() {
        try {
            const data = await ApiClient.getSampleData(500);
            if (!data.stats) return;
            
            UI.updateStat('avgVolume', FormatUtils.volume(data.stats.avg_traffic_volume));
            UI.updateStat('avgSpeed', FormatUtils.kmh(data.stats.avg_speed));
            UI.updateStat('avgCongestion', FormatUtils.decimal(data.stats.avg_congestion));
            UI.updateStat('totalLocations', data.stats.locations);
            
            return data;
        } catch (err) {
            console.error('Failed to load stats:', err);
        }
    },
    
    async loadTimeSeries(containerId, n = 500) {
        try {
            UI.showLoading(containerId);
            const data = await ApiClient.getSampleData(n);
            if (!data.data || data.data.length === 0) return;
            
            const timeData = data.data.slice(0, 500);
            const trace = {
                x: timeData.map((_, i) => i),
                y: timeData.map(d => d.congestion_level),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Congestion Level',
                line: { color: '#2563eb', width: 2.5, shape: 'spline' },
                marker: { 
                    size: 4, 
                    color: timeData.map(d => CongestionUtils.getColor(d.congestion_level)),
                    symbol: 'circle'
                },
                hovertemplate: 'Sample %{x}<br>Level: %{y:.2f}<extra></extra>'
            };
            
            const layout = ChartManager.darkLayout('', 'Sample', 'Congestion Level (0-3)');
            layout.yaxis.range = [-0.3, 3.3];
            layout.margin.t = 10;
            layout.showlegend = false;
            
            ChartManager.create(containerId, [trace], layout);
            return data;
        } catch (err) {
            UI.showError(containerId, 'Failed to load time series data');
        }
    },
    
    async loadDistribution(containerId) {
        try {
            UI.showLoading(containerId);
            const data = await ApiClient.getSampleData(500);
            if (!data.data) return;
            
            const dist = {};
            data.data.forEach(d => {
                const level = Math.round(d.congestion_level);
                dist[level] = (dist[level] || 0) + 1;
            });
            
            const labels = ['Low', 'Moderate', 'High', 'Severe'];
            const values = [0,1,2,3].map(i => dist[i] || 0);
            const colors = ['#10b981', '#eab308', '#f59e0b', '#dc2626'];
            
            const trace = {
                labels, values,
                type: 'pie',
                marker: { colors },
                textinfo: 'label+percent',
                textposition: 'outside',
                hole: 0.45,
                hoverinfo: 'label+value+percent',
                textfont: { size: 11, color: '#94a3b8' }
            };
            
            const layout = ChartManager.darkLayout();
            layout.margin = { t: 10, r: 10, b: 10, l: 10 };
            layout.showlegend = false;
            layout.annotations = [{
                text: '<b>Congestion<br>Distribution</b>',
                showarrow: false,
                font: { size: 13, color: '#e2e8f0' }
            }];
            
            ChartManager.create(containerId, [trace], layout);
        } catch (err) {
            UI.showError(containerId, 'Failed to load distribution');
        }
    },
    
    async loadHourly(containerId) {
        try {
            UI.showLoading(containerId);
            const data = await ApiClient.getTimeAnalysis();
            if (!data.hourly_avg) return;
            
            const hours = Object.keys(data.hourly_avg).map(Number);
            const values = Object.values(data.hourly_avg);
            
            const trace = {
                x: hours,
                y: values,
                type: 'bar',
                marker: {
                    color: values.map(v => CongestionUtils.getColor(v)),
                    line: { color: values.map(v => CongestionUtils.getColor(v)), width: 1 }
                },
                hovertemplate: 'Hour: %{x}:00<br>Avg: %{y:.2f}<extra></extra>'
            };
            
            const layout = ChartManager.darkLayout('', 'Hour of Day', 'Avg Congestion');
            layout.xaxis.dtick = 2;
            layout.margin.t = 10;
            layout.showlegend = false;
            
            // Add rush hour markers
            const shapes = [
                { type: 'rect', x0: 6.5, x1: 9.5, y0: 0, y1: 1, yref: 'paper', fillcolor: 'rgba(37,99,235,0.08)', line: { width: 0 } },
                { type: 'rect', x0: 16.5, x1: 19.5, y0: 0, y1: 1, yref: 'paper', fillcolor: 'rgba(37,99,235,0.08)', line: { width: 0 } }
            ];
            
            layout.shapes = shapes;
            layout.annotations = [
                { x: 8, y: 1.08, yref: 'paper', text: '🚗 Morning Rush', showarrow: false, font: { size: 10, color: '#3b82f6' } },
                { x: 18, y: 1.08, yref: 'paper', text: '🚗 Evening Rush', showarrow: false, font: { size: 10, color: '#3b82f6' } }
            ];
            
            ChartManager.create(containerId, [trace], layout);
        } catch (err) {
            UI.showError(containerId, 'Failed to load hourly data');
        }
    },
    
    async loadWeatherImpact(containerId) {
        try {
            UI.showLoading(containerId);
            const data = await ApiClient.getWeatherAnalysis();
            if (!data.weather_impact) return;
            
            const weatherMap = { '0': 'Clear', '1': 'Cloudy', '2': 'Rainy', '3': 'Stormy', '4': 'Foggy', '5': 'Snowy' };
            const weatherIcons = { '0': '☀️', '1': '☁️', '2': '🌧️', '3': '⛈️', '4': '🌫️', '5': '❄️' };
            
            const conditions = Object.keys(data.weather_impact);
            const labels = conditions.map(k => `${weatherIcons[k] || ''} ${weatherMap[k] || k}`);
            const values = conditions.map(k => data.weather_impact[k].avg_congestion);
            const counts = conditions.map(k => data.weather_impact[k].samples);
            
            const trace = {
                x: labels,
                y: values,
                type: 'bar',
                marker: {
                    color: ['#60a5fa', '#94a3b8', '#6366f1', '#8b5cf6', '#a78bfa', '#e879f9'],
                    line: { color: 'rgba(255,255,255,0.1)', width: 1 }
                },
                text: values.map(v => v.toFixed(2)),
                textposition: 'outside',
                textfont: { size: 10 },
                hovertemplate: '%{x}<br>Avg Congestion: %{y:.2f}<br>Samples: %{customdata}<extra></extra>',
                customdata: counts
            };
            
            const layout = ChartManager.darkLayout('', 'Weather Condition', 'Avg Congestion');
            layout.margin.t = 10;
            layout.showlegend = false;
            
            ChartManager.create(containerId, [trace], layout);
        } catch (err) {
            UI.showError(containerId, 'Failed to load weather data');
        }
    },
    
    async loadLocationRanking(containerId) {
        try {
            UI.showLoading(containerId);
            const data = await ApiClient.getLocationRanking();
            if (!data.rankings) return;
            
            const locations = data.rankings.map(r => r.location);
            const avgCongestion = data.rankings.map(r => r.avg_congestion);
            const colors = avgCongestion.map(v => CongestionUtils.getColor(v));
            
            const trace = {
                x: avgCongestion,
                y: locations,
                type: 'bar',
                orientation: 'h',
                marker: { color: colors, line: { color: 'rgba(255,255,255,0.1)', width: 1 } },
                text: avgCongestion.map(v => v.toFixed(2)),
                textposition: 'outside',
                textfont: { size: 10 },
                hovertemplate: '%{y}<br>Avg Congestion: %{x:.2f}<extra></extra>'
            };
            
            const layout = ChartManager.darkLayout('', 'Avg Congestion Level', '');
            layout.margin = { t: 10, r: 60, b: 40, l: 130 };
            layout.showlegend = false;
            layout.xaxis.range = [0, Math.max(...avgCongestion) + 0.5];
            
            ChartManager.create(containerId, [trace], layout);
        } catch (err) {
            UI.showError(containerId, 'Failed to load location data');
        }
    },
    
    async loadOptimalTimes(containerId) {
        try {
            UI.showLoading(containerId);
            const data = await ApiClient.getOptimalTimes();
            if (!data.recommendations) return;
            
            const recommendations = data.recommendations;
            
            let html = '<div class="row g-3">';
            recommendations.forEach(r => {
                const severity = CongestionUtils.getLabel(r.congestion_level);
                const color = CongestionUtils.getColor(r.congestion_level);
                const bg = CongestionUtils.getBgColor(r.congestion_level);
                html += `
                    <div class="col-md-6">
                        <div class="card-premium" style="border-left: 3px solid ${color};">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h6 class="fw-semibold mb-0"><i class="fas fa-map-marker-alt me-2" style="color: ${color}"></i>${r.location}</h6>
                                <span class="badge-premium badge-premium-${severity === 'Low' ? 'success' : severity === 'Moderate' ? 'warning' : 'danger'}">${severity}</span>
                            </div>
                            <div class="small text-muted mb-2">
                                <i class="far fa-clock me-1"></i>Best: ${r.best_time}
                            </div>
                            <div class="d-flex justify-content-between text-muted small">
                                <span><i class="fas fa-car me-1"></i>${FormatUtils.volume(r.traffic_volume)}</span>
                                <span><i class="fas fa-tachometer-alt me-1"></i>${FormatUtils.kmh(r.avg_speed)}</span>
                            </div>
                            <div class="mt-2">
                                <div class="progress-custom">
                                    <div class="progress-fill" style="width: ${(r.congestion_level / 3 * 100).toFixed(0)}%; background: ${color};"></div>
                                </div>
                            </div>
                        </div>
                    </div>`;
            });
            html += '</div>';
            
            document.getElementById(containerId).innerHTML = html;
        } catch (err) {
            UI.showError(containerId, 'Failed to load recommendations');
        }
    }
};

// ============================================================
// PREDICTION FORM - Premium
// ============================================================
const PredictionForm = {
    init() {
        const form = document.getElementById('predictForm');
        if (!form) return;
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.submit();
        });
        
        // Auto-detect rush hour
        const hourInput = document.getElementById('hour');
        const dayOfWeekInput = document.getElementById('dayOfWeek');
        if (hourInput && dayOfWeekInput) {
            hourInput.addEventListener('change', () => this.updateRushHour());
            dayOfWeekInput.addEventListener('change', () => this.updateRushHour());
        }
        
        // Load example presets
        this.loadPresets();
    },
    
    updateRushHour() {
        const hour = parseInt(document.getElementById('hour')?.value) || 12;
        const day = parseInt(document.getElementById('dayOfWeek')?.value) || 0;
        const isRush = (7 <= hour && hour <= 9 || 17 <= hour && hour <= 19) && day < 5;
        const el = document.getElementById('rushHourIndicator');
        if (el) {
            el.innerHTML = isRush 
                ? '<span class="badge-premium badge-premium-warning"><i class="fas fa-clock me-1"></i>Rush Hour</span>'
                : '<span class="badge-premium badge-premium-success"><i class="fas fa-clock me-1"></i>Off-Peak</span>';
        }
    },
    
    loadPresets() {
        const presets = [
            { name: '🌆 Evening Rush', hour: 17, day: 1, month: 6, temp: 32, humidity: 65, precip: 0.2, wind: 12, volume: 850, speed: 35, weather: 'Rainy', location: 'Downtown' },
            { name: '🌅 Morning Commute', hour: 8, day: 2, month: 3, temp: 18, humidity: 70, precip: 0.5, wind: 15, volume: 720, speed: 30, weather: 'Cloudy', location: 'Highway_A' },
            { name: '🌙 Late Night', hour: 23, day: 5, month: 12, temp: 10, humidity: 55, precip: 0, wind: 8, volume: 120, speed: 65, weather: 'Clear', location: 'Residential_A' },
            { name: '⛈️ Stormy Day', hour: 14, day: 3, month: 7, temp: 22, humidity: 85, precip: 15, wind: 35, volume: 950, speed: 20, weather: 'Stormy', location: 'Bridge_1' }
        ];
        
        const container = document.getElementById('presetButtons');
        if (!container) return;
        
        presets.forEach((preset, i) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-sm btn-glass me-2 mb-2';
            btn.textContent = preset.name;
            btn.onclick = () => this.applyPreset(preset);
            container.appendChild(btn);
        });
    },
    
    applyPreset(p) {
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
        set('hour', p.hour);
        set('dayOfWeek', p.day);
        set('month', p.month);
        set('temperature', p.temp);
        set('humidity', p.humidity);
        set('precipitation', p.precip);
        set('windSpeed', p.wind);
        set('trafficVolume', p.volume);
        set('avgSpeed', p.speed);
        set('weatherCondition', p.weather);
        set('location', p.location);
        this.updateRushHour();
        UI.showToast(`Loaded preset: ${p.name}`, 'info');
    },
    
    async submit() {
        const btn = document.getElementById('predictBtn');
        if (!btn) return;
        
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-premium" style="width:20px;height:20px;border-width:2px;display:inline-block;margin-right:8px;vertical-align:middle;"></span> Analyzing...';
        
        const data = {
            hour: parseInt(document.getElementById('hour')?.value) || 12,
            day_of_week: parseInt(document.getElementById('dayOfWeek')?.value) || 0,
            month: parseInt(document.getElementById('month')?.value) || 1,
            is_weekend: parseInt(document.getElementById('dayOfWeek')?.value) >= 5 ? 1 : 0,
            temperature: parseFloat(document.getElementById('temperature')?.value) || 25,
            humidity: parseFloat(document.getElementById('humidity')?.value) || 60,
            precipitation: parseFloat(document.getElementById('precipitation')?.value) || 0,
            wind_speed: parseFloat(document.getElementById('windSpeed')?.value) || 10,
            traffic_volume: parseInt(document.getElementById('trafficVolume')?.value) || 500,
            avg_speed: parseInt(document.getElementById('avgSpeed')?.value) || 50,
            weather_condition: document.getElementById('weatherCondition')?.value || 'Clear',
            location: document.getElementById('location')?.value || 'Downtown',
            model: document.getElementById('modelSelect')?.value || 'ensemble'
        };
        
        try {
            const result = await ApiClient.predict(data);
            this.displayResult(result);
            UI.showToast('Prediction completed successfully!', 'success');
        } catch (err) {
            UI.showToast('Prediction failed. Check server & models.', 'error');
            console.error(err);
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    },
    
    displayResult(result) {
        const card = document.getElementById('resultCard');
        const welcome = document.getElementById('welcomeCard');
        if (card) { card.classList.remove('d-none'); card.style.display = 'block'; }
        if (welcome) welcome.style.display = 'none';
        
        const level = result.congestion_level;
        const label = result.congestion_label;
        const value = result.congestion_value;
        const confidence = result.confidence || 0.5;
        const color = CongestionUtils.getColor(level);
        const bgColor = CongestionUtils.getBgColor(level);
        
        // Update indicator ring
        const indicator = document.getElementById('congestionIndicator');
        if (indicator) {
            indicator.style.borderColor = color;
            indicator.style.background = bgColor;
            indicator.innerHTML = `<i class="fas fa-traffic-light" style="color: ${color}; font-size: 2.5rem;"></i>`;
        }
        
        // Update text
        const labelEl = document.getElementById('congestionLabel');
        if (labelEl) { labelEl.textContent = label; labelEl.style.color = color; }
        
        const valEl = document.getElementById('congestionValue');
        if (valEl) { valEl.textContent = value.toFixed(2); valEl.style.color = color; }
        
        // Confidence
        const confidencePct = Math.min(Math.round(confidence * 100), 100);
        const confidenceText = document.getElementById('confidenceText');
        if (confidenceText) confidenceText.textContent = `${confidencePct}%`;
        
        const fill = document.getElementById('confidenceFill');
        if (fill) {
            fill.style.width = `${confidencePct}%`;
            fill.style.background = confidence >= 0.7 ? color : '#f59e0b';
        }
        
        // Model predictions
        const modelPredictions = document.getElementById('modelPredictions');
        if (modelPredictions && result.model_predictions) {
            const modelNames = { random_forest: '🌲 Random Forest', xgboost: '⚡ XGBoost', lstm: '🧠 LSTM', linear_regression: '📈 Linear Reg.' };
            modelPredictions.innerHTML = Object.entries(result.model_predictions)
                .filter(([_, v]) => v !== null)
                .map(([name, pred]) => `
                    <div class="d-flex justify-content-between align-items-center py-2 px-3 mb-1" 
                         style="background: rgba(0,0,0,0.2); border-radius: 8px;">
                        <span class="small">${modelNames[name] || name}</span>
                        <span class="fw-semibold small" style="color: ${color}">${pred.toFixed(4)}</span>
                    </div>`).join('');
        }
        
        // Timestamp
        const timeEl = document.getElementById('predictionTime');
        if (timeEl) timeEl.textContent = `Predicted at ${new Date().toLocaleTimeString()}`;
    }
};

// ============================================================
// PARTICLES BACKGROUND GENERATOR
// ============================================================
function initParticles() {
    const container = document.getElementById('particlesBg');
    if (!container) return;
    
    for (let i = 0; i < 40; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = (Math.random() * 3 + 1) + 'px';
        particle.style.height = particle.style.width;
        particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
        particle.style.animationDelay = (Math.random() * 10) + 's';
        particle.style.opacity = Math.random() * 0.3 + 0.1;
        container.appendChild(particle);
    }
}

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            navbar.classList.toggle('scrolled', window.scrollY > 50);
        });
    }
    
    // Initialize particles
    initParticles();
    
    // Initialize prediction form
    PredictionForm.init();
    
    // Initialize tooltips
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
    
    console.log('🚦 TrafficAI Premium Frontend initialized');
});

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CongestionUtils, FormatUtils, ApiClient, ChartManager, UI, DashboardLoader, PredictionForm, AutoRefresh };
}

