/* ============================================
   ParkVision AI — Main Application Logic
   ============================================ */

const API = '';
let currentResult = null;
let doughnutChart = null;
let barChart = null;

// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initParticles();
    initUpload();
    initTabs();
    initCompareSlider();
    initButtons();
    loadHeroStats();
    loadHistory();
});

// ==================== THEME ====================
function initTheme() {
    const saved = localStorage.getItem('pv-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    document.getElementById('themeToggle').addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('pv-theme', next);
        if (doughnutChart) updateChartColors();
    });
}

// ==================== PARTICLES ====================
function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 60; i++) {
        particles.push({
            x: Math.random() * canvas.width, y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4,
            r: Math.random() * 2 + 0.5, alpha: Math.random() * 0.4 + 0.1
        });
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const theme = document.documentElement.getAttribute('data-theme');
        const color = theme === 'dark' ? '99,102,241' : '99,102,241';
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${color},${p.alpha})`;
            ctx.fill();
        });
        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(${color},${0.08 * (1 - dist / 120)})`;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
}

// ==================== UPLOAD ====================
function initUpload() {
    const area = document.getElementById('uploadArea');
    const input = document.getElementById('fileInput');
    const content = document.getElementById('uploadContent');
    const preview = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('previewImage');
    const changeBtn = document.getElementById('changeImageBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Click to upload
    area.addEventListener('click', (e) => {
        if (e.target === changeBtn || e.target.closest('#changeImageBtn')) return;
        if (!preview.classList.contains('hidden')) return;
        input.click();
    });

    // Drag & drop
    ['dragenter', 'dragover'].forEach(evt => {
        area.addEventListener(evt, (e) => { e.preventDefault(); area.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        area.addEventListener(evt, (e) => { e.preventDefault(); area.classList.remove('drag-over'); });
    });
    area.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        input.value = '';
        input.click();
    });

    function handleFile(file) {
        const valid = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!valid.includes(file.type)) { showToast('Invalid file type. Use JPG, PNG, or WebP.', 'error'); return; }
        if (file.size > 20 * 1024 * 1024) { showToast('File too large. Maximum 20MB.', 'error'); return; }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            content.classList.add('hidden');
            preview.classList.remove('hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.dataset.file = 'ready';
        };
        reader.readAsDataURL(file);
        area._file = file;
    }

    analyzeBtn.addEventListener('click', () => {
        if (!area._file) return;
        runDetection(area._file);
    });
}

// ==================== DETECTION ====================
async function runDetection(file) {
    const progressDiv = document.getElementById('uploadProgress');
    const content = document.getElementById('uploadContent');
    const preview = document.getElementById('uploadPreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const progressCircle = document.getElementById('progressCircle');
    const progressText = document.getElementById('progressText');
    const progressLabel = document.getElementById('progressLabel');

    // Show progress
    content.classList.add('hidden');
    preview.classList.add('hidden');
    progressDiv.classList.remove('hidden');
    analyzeBtn.disabled = true;

    const circumference = 2 * Math.PI * 54;
    let progress = 0;

    const fakeProgress = setInterval(() => {
        progress = Math.min(progress + Math.random() * 8, 90);
        const offset = circumference - (progress / 100) * circumference;
        progressCircle.style.strokeDashoffset = offset;
        progressText.textContent = Math.round(progress) + '%';

        if (progress < 30) progressLabel.textContent = 'Uploading image...';
        else if (progress < 60) progressLabel.textContent = 'Running AI detection...';
        else if (progress < 80) progressLabel.textContent = 'Analyzing parking spaces...';
        else progressLabel.textContent = 'Generating visual report...';
    }, 200);

    try {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${API}/api/detect`, { method: 'POST', body: formData });
        const data = await response.json();

        clearInterval(fakeProgress);

        if (!data.success) throw new Error(data.error || 'Detection failed');

        // Complete progress
        progressCircle.style.strokeDashoffset = 0;
        progressText.textContent = '100%';
        progressLabel.textContent = 'Analysis complete!';

        await new Promise(r => setTimeout(r, 600));

        currentResult = data;
        showResults(data);
        showToast(`Detected ${data.occupied} vehicles in ${data.processing_time}s`, 'success');
        loadHistory();
        loadHeroStats();

    } catch (err) {
        clearInterval(fakeProgress);
        showToast(err.message, 'error');
        resetUpload();
    }
}

function resetUpload() {
    document.getElementById('uploadContent').classList.remove('hidden');
    document.getElementById('uploadPreview').classList.add('hidden');
    document.getElementById('uploadProgress').classList.add('hidden');
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadArea')._file = null;
}

// ==================== RESULTS ====================
function showResults(data) {
    const section = document.getElementById('resultsSection');
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Animate stats
    animateCounter('totalSpaces', data.total_spaces);
    animateCounter('occupiedSpaces', data.occupied);
    animateCounter('availableSpaces', data.available);
    animateCounter('confidenceValue', data.confidence_avg, '%');

    // Gauge
    updateGauge(data.occupancy_rate);

    // Processing info
    document.getElementById('procTime').textContent = data.processing_time + 's';
    document.getElementById('imgDims').textContent = `${data.image_dimensions.width} × ${data.image_dimensions.height}`;
    document.getElementById('vehicleCount').textContent = data.occupied;
    document.getElementById('detectionId').textContent = '#' + data.id;
    document.getElementById('confThresh').textContent = (data.model_info.confidence_threshold * 100) + '%';
    document.getElementById('modelName').textContent = data.model_info.name || 'YOLOv8';

    // Images
    document.getElementById('annotatedImg').src = data.annotated_image;
    document.getElementById('heatmapImg').src = data.heatmap_image;
    document.getElementById('originalImg').src = data.original_image;
    document.getElementById('compareBefore').src = data.original_image;
    document.getElementById('compareAfter').src = data.annotated_image;

    // Charts
    renderCharts(data);

    // Zones
    renderZones(data.zone_analysis);

    // Detections table
    renderDetectionsTable(data.detections);

    // Reset upload area
    resetUpload();
}

function animateCounter(elementId, target, suffix = '') {
    const el = document.getElementById(elementId);
    const duration = 1200;
    const start = performance.now();
    const startVal = 0;

    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(startVal + (target - startVal) * eased);
        el.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function updateGauge(rate) {
    const fill = document.getElementById('gaugeFill');
    const value = document.getElementById('gaugeValue');
    const label = document.getElementById('gaugeLabel');
    const circumference = 2 * Math.PI * 85;
    const offset = circumference - (rate / 100) * circumference;

    setTimeout(() => {
        fill.style.strokeDashoffset = offset;
        if (rate < 50) { fill.style.stroke = 'var(--green)'; label.textContent = 'Low Occupancy'; }
        else if (rate < 80) { fill.style.stroke = 'var(--yellow)'; label.textContent = 'Moderate'; }
        else { fill.style.stroke = 'var(--red)'; label.textContent = 'High Occupancy'; }
    }, 200);

    animateCounter('gaugeValue', rate, '%');
}

// ==================== CHARTS ====================
function renderCharts(data) {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#9595b5' : '#555577';

    // Doughnut
    const dCtx = document.getElementById('doughnutChart').getContext('2d');
    if (doughnutChart) doughnutChart.destroy();
    doughnutChart = new Chart(dCtx, {
        type: 'doughnut',
        data: {
            labels: ['Occupied', 'Available'],
            datasets: [{
                data: [data.occupied, data.available],
                backgroundColor: [
                    isDark ? 'rgba(239,68,68,0.8)' : 'rgba(220,38,38,0.8)',
                    isDark ? 'rgba(34,197,94,0.8)' : 'rgba(22,163,74,0.8)'
                ],
                borderWidth: 0,
                borderRadius: 6,
                spacing: 4
            }]
        },
        options: {
            responsive: true, cutout: '70%',
            plugins: {
                legend: { position: 'bottom', labels: { color: textColor, padding: 16, font: { family: 'Inter', weight: 600 } } }
            },
            animation: { animateRotate: true, duration: 1500 }
        }
    });

    // Bar
    const bCtx = document.getElementById('barChart').getContext('2d');
    if (barChart) barChart.destroy();
    const vt = data.vehicle_types;
    barChart = new Chart(bCtx, {
        type: 'bar',
        data: {
            labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
            datasets: [{
                label: 'Count',
                data: [vt.car || 0, vt.motorcycle || 0, vt.bus || 0, vt.truck || 0],
                backgroundColor: [
                    'rgba(59,130,246,0.7)', 'rgba(245,158,11,0.7)',
                    'rgba(168,85,247,0.7)', 'rgba(6,182,212,0.7)'
                ],
                borderRadius: 8, borderSkipped: false, barPercentage: 0.6
            }]
        },
        options: {
            responsive: true, indexAxis: 'x',
            scales: {
                x: { grid: { display: false }, ticks: { color: textColor, font: { family: 'Inter' } } },
                y: { grid: { color: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)' }, ticks: { color: textColor, stepSize: 1, font: { family: 'Inter' } }, beginAtZero: true }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 1200 }
        }
    });
}

function updateChartColors() {
    if (currentResult) renderCharts(currentResult);
}

// ==================== ZONES ====================
function renderZones(zones) {
    const grid = document.getElementById('zonesGrid');
    grid.innerHTML = '';
    const colors = ['#6366f1', '#a855f7', '#3b82f6', '#06b6d4'];
    let i = 0;
    for (const [key, zone] of Object.entries(zones)) {
        const color = colors[i % 4];
        const el = document.createElement('div');
        el.className = 'zone-item animate-in';
        el.style.borderLeftColor = color;
        el.style.animationDelay = `${i * 0.1}s`;
        el.innerHTML = `
            <div class="zone-name">${zone.name}</div>
            <div class="zone-stats">
                <div class="zone-stat"><span class="zone-stat-val" style="color:${color}">${zone.occupied}</span><span class="zone-stat-lbl">Occupied</span></div>
                <div class="zone-stat"><span class="zone-stat-val" style="color:var(--green)">${zone.available}</span><span class="zone-stat-lbl">Available</span></div>
                <div class="zone-stat"><span class="zone-stat-val">${zone.total}</span><span class="zone-stat-lbl">Total</span></div>
                <div class="zone-stat"><span class="zone-stat-val">${zone.occupancy}%</span><span class="zone-stat-lbl">Occupancy</span></div>
            </div>
            <div class="zone-bar"><div class="zone-bar-fill" style="width:${zone.occupancy}%;background:${zone.occupancy > 80 ? 'var(--red)' : zone.occupancy > 50 ? 'var(--yellow)' : 'var(--green)'}"></div></div>`;
        grid.appendChild(el);
        i++;
    }
}

// ==================== DETECTIONS TABLE ====================
function renderDetectionsTable(detections) {
    const tbody = document.getElementById('detectionsBody');
    tbody.innerHTML = '';
    detections.forEach((det, idx) => {
        const conf = det.confidence * 100;
        const confClass = conf >= 70 ? 'conf-high' : conf >= 40 ? 'conf-med' : 'conf-low';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td><span class="type-badge">${det.class_name.toUpperCase()}</span></td>
            <td><span class="conf-badge ${confClass}">${conf.toFixed(1)}%</span></td>
            <td style="font-family:var(--mono);font-size:0.8rem;color:var(--text-secondary)">(${det.center[0]}, ${det.center[1]})</td>
            <td style="font-family:var(--mono);font-size:0.8rem;color:var(--text-secondary)">${det.area.toLocaleString()}px²</td>`;
        tbody.appendChild(tr);
    });
    if (!detections.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:24px;">No vehicles detected</td></tr>';
    }
}

// ==================== TABS ====================
function initTabs() {
    document.querySelectorAll('.comp-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.comp-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.comp-pane').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('pane-' + tab.dataset.tab).classList.add('active');
        });
    });
}

// ==================== COMPARE SLIDER ====================
function initCompareSlider() {
    const slider = document.getElementById('sliderCompare');
    const handle = document.getElementById('compareHandle');
    const afterWrap = document.getElementById('compareAfterWrap');
    if (!slider) return;

    let dragging = false;
    const move = (clientX) => {
        const rect = slider.getBoundingClientRect();
        let x = clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        const pct = (x / rect.width) * 100;
        afterWrap.style.width = pct + '%';
        handle.style.left = pct + '%';
    };

    slider.addEventListener('mousedown', (e) => { dragging = true; move(e.clientX); });
    window.addEventListener('mousemove', (e) => { if (dragging) move(e.clientX); });
    window.addEventListener('mouseup', () => { dragging = false; });
    slider.addEventListener('touchstart', (e) => { dragging = true; move(e.touches[0].clientX); });
    window.addEventListener('touchmove', (e) => { if (dragging) move(e.touches[0].clientX); });
    window.addEventListener('touchend', () => { dragging = false; });
}

// ==================== BUTTONS ====================
function initButtons() {
    document.getElementById('downloadAnnotatedBtn').addEventListener('click', () => {
        if (currentResult) downloadImage(currentResult.annotated_image, 'parkvision_annotated.jpg');
    });
    document.getElementById('downloadHeatmapBtn').addEventListener('click', () => {
        if (currentResult) downloadImage(currentResult.heatmap_image, 'parkvision_heatmap.jpg');
    });
    document.getElementById('newAnalysisBtn').addEventListener('click', () => {
        document.getElementById('resultsSection').classList.add('hidden');
        resetUpload();
        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
    });
}

function downloadImage(url, filename) {
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
}

// ==================== HISTORY ====================
async function loadHistory() {
    try {
        const res = await fetch(`${API}/api/history`);
        const data = await res.json();
        if (!data.success) return;

        const grid = document.getElementById('historyGrid');
        const empty = document.getElementById('historyEmpty');

        if (!data.detections.length) { empty.classList.remove('hidden'); return; }
        empty.classList.add('hidden');

        // Keep only history cards
        grid.querySelectorAll('.history-card').forEach(c => c.remove());

        data.detections.forEach(det => {
            const card = document.createElement('div');
            card.className = 'history-card animate-in';
            const date = new Date(det.timestamp).toLocaleString();
            card.innerHTML = `
                <img class="history-thumb" src="${det.annotated_image}" alt="Detection ${det.id}" loading="lazy">
                <div class="history-info">
                    <div class="history-date">📅 ${date}</div>
                    <div class="history-stats-row">
                        <span class="history-stat hs-total">🅿️ ${det.total_spaces} total</span>
                        <span class="history-stat hs-occ">🚗 ${det.occupied} occupied</span>
                        <span class="history-stat hs-avail">✅ ${det.available} free</span>
                    </div>
                </div>
                <div class="history-actions">
                    <button class="btn btn-secondary btn-sm" onclick="viewHistoryItem(${det.id})">View</button>
                    <button class="btn btn-secondary btn-sm" onclick="deleteHistoryItem(${det.id}, this)" style="color:var(--red)">Delete</button>
                </div>`;
            grid.appendChild(card);
        });
    } catch (e) { /* silent */ }
}

async function viewHistoryItem(id) {
    try {
        const res = await fetch(`${API}/api/history/${id}`);
        const data = await res.json();
        if (data.success) {
            currentResult = data.detection;
            currentResult.model_info = currentResult.model_info || { confidence_threshold: 0.3 };
            currentResult.image_dimensions = currentResult.image_dimensions || { width: 0, height: 0 };
            showResults(data.detection);
        }
    } catch (e) { showToast('Failed to load detection', 'error'); }
}

async function deleteHistoryItem(id, btn) {
    if (!confirm('Delete this detection?')) return;
    try {
        await fetch(`${API}/api/history/${id}`, { method: 'DELETE' });
        const card = btn.closest('.history-card');
        card.style.opacity = '0'; card.style.transform = 'scale(0.9)';
        setTimeout(() => { card.remove(); loadHistory(); }, 300);
        showToast('Detection deleted', 'info');
    } catch (e) { showToast('Failed to delete', 'error'); }
}

// ==================== HERO STATS ====================
async function loadHeroStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const data = await res.json();
        if (data.success) {
            animateCounter('statScans', data.stats.total_scans);
            animateCounter('statVehicles', data.stats.total_vehicles_detected);
            document.getElementById('statAccuracy').textContent = data.stats.avg_confidence + '%';
        }
    } catch (e) { /* silent */ }
}

// ==================== TOAST ====================
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(60px)'; setTimeout(() => toast.remove(), 300); }, 4000);
}
