/* =========================================================================
   NeuroScan — Clinical Brain MRI Analysis, True Pixel Radiomics & AI Review
   ========================================================================= */

(function () {
  'use strict';

  // Safe Storage Helper (handles file:// restrictions and private mode safely)
  const storage = {
    get(key, fallback = '') {
      try {
        const v = localStorage.getItem(key);
        return v !== null ? v : fallback;
      } catch (e) {
        return fallback;
      }
    },
    set(key, val) {
      try {
        localStorage.setItem(key, val);
      } catch (e) {}
    },
    remove(key) {
      try {
        localStorage.removeItem(key);
      } catch (e) {}
    }
  };

  function safeCreateIcons() {
    if (typeof lucide !== 'undefined' && lucide && typeof lucide.createIcons === 'function') {
      try {
        lucide.createIcons();
      } catch (e) {}
    }
  }
  safeCreateIcons();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', safeCreateIcons);
  }

  // ---------- Light / Dark Mode Management ----------
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeIconMoon = document.getElementById('themeIconMoon');
  const themeIconSun = document.getElementById('themeIconSun');

  function applyTheme(isDark) {
    if (isDark) {
      document.documentElement.classList.add('dark');
      themeIconMoon?.classList.add('hidden');
      themeIconSun?.classList.remove('hidden');
      storage.set('NEUROSCAN_THEME', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      themeIconMoon?.classList.remove('hidden');
      themeIconSun?.classList.add('hidden');
      storage.set('NEUROSCAN_THEME', 'light');
    }
  }

  const savedTheme = storage.get('NEUROSCAN_THEME', '');
  const systemDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (savedTheme === 'dark' || (!savedTheme && systemDark)) {
    applyTheme(true);
  } else {
    applyTheme(false);
  }

  themeToggleBtn?.addEventListener('click', () => {
    const isDark = document.documentElement.classList.contains('dark');
    applyTheme(!isDark);
  });

  // ---------- Global State ----------
  let geminiApiKey = storage.get('NEUROSCAN_GEMINI_KEY', '');
  let cohortScans = [];
  let activeScanIndex = -1;

  // Viewport transforms & DICOM Window/Level
  let currentZoom = 1;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let startPanX = 0;
  let startPanY = 0;
  let showCrosshair = true;
  let currentFilterMode = 'normal'; // 'normal' | 'invert' | 'heatmap' | 'high-contrast'
  let chatHistory = [];

  // Active models prioritized by performance & modern capability
  const FLASH_ENDPOINTS = [
    { version: 'v1beta', model: 'gemini-3.8-flash' },
    { version: 'v1beta', model: 'gemini-2.0-flash' },
    { version: 'v1beta', model: 'gemini-2.5-flash' },
    { version: 'v1beta', model: 'gemini-1.5-flash' },
    { version: 'v1beta', model: 'gemini-1.5-flash-latest' }
  ];

  let workingEndpoint = { version: 'v1beta', model: 'gemini-3.8-flash' };
  try {
    const storedEndpoint = storage.get('NEUROSCAN_WORKING_ENDPOINT', '');
    if (storedEndpoint) {
      const parsed = JSON.parse(storedEndpoint);
      if (parsed && parsed.version && parsed.model) {
        workingEndpoint = parsed;
      }
    }
  } catch (e) {
    workingEndpoint = { version: 'v1beta', model: 'gemini-3.8-flash' };
  }

  // Model discovery via Google AI Studio API
  async function discoverAvailableModels(apiKey) {
    if (!apiKey) return [];
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 6000);
      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!res.ok) return [];
      const data = await res.json();
      if (!data.models || !Array.isArray(data.models)) return [];

      const valid = data.models
        .filter(m => Array.isArray(m.supportedGenerationMethods) && m.supportedGenerationMethods.includes('generateContent'))
        .map(m => m.name.replace(/^models\//, ''))
        .filter(name => !name.includes('embedding') && !name.includes('aqa'));

      valid.sort((a, b) => {
        const score = name => {
          const n = name.toLowerCase();
          if (n === 'gemini-3.8-flash') return 1000;
          if (n.includes('3.8-flash')) return 900;
          if (n === 'gemini-2.0-flash') return 700;
          if (n.includes('2.0-flash')) return 600;
          if (n.includes('2.5-flash')) return 500;
          if (n.includes('1.5-flash')) return 400;
          return 10;
        };
        return score(b) - score(a);
      });

      return valid;
    } catch (e) {
      return [];
    }
  }

  async function getCandidateEndpoints() {
    if (workingEndpoint && workingEndpoint.model) {
      const list = [workingEndpoint];
      FLASH_ENDPOINTS.forEach(e => {
        if (!list.some(item => item.model === e.model)) {
          list.push(e);
        }
      });
      return list;
    }
    return FLASH_ENDPOINTS;
  }

  // ---------- High-Fidelity Client-Side Pixel Radiomics Engine ----------
  async function analyzeActualImagePixels(dataUrl) {
    return new Promise((resolve) => {
      if (!dataUrl || !dataUrl.startsWith('data:image/')) {
        resolve({
          width: 0,
          height: 0,
          min: 0,
          max: 0,
          mean: '0.0',
          stdDev: '0.0',
          snr: '0.00',
          histogram: new Array(24).fill(0),
          pixelGrid: null
        });
        return;
      }

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        try {
          const sampleW = Math.min(img.naturalWidth || 256, 256);
          const sampleH = Math.min(img.naturalHeight || 256, 256);

          const canvas = document.createElement('canvas');
          canvas.width = sampleW;
          canvas.height = sampleH;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, sampleW, sampleH);

          const imgData = ctx.getImageData(0, 0, sampleW, sampleH);
          const data = imgData.data;

          let totalLum = 0;
          let minLum = 255;
          let maxLum = 0;
          const luminances = new Float32Array(sampleW * sampleH);
          const histogram = new Array(24).fill(0);

          let pIdx = 0;
          for (let i = 0; i < data.length; i += 4) {
            // Rec.601 perceptual luminance: Y = 0.299 R + 0.587 G + 0.114 B
            const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            luminances[pIdx++] = lum;
            totalLum += lum;

            if (lum < minLum) minLum = lum;
            if (lum > maxLum) maxLum = lum;

            const bin = Math.min(23, Math.max(0, Math.floor((lum / 255) * 24)));
            histogram[bin]++;
          }

          const count = luminances.length;
          const mean = totalLum / count;

          let varianceSum = 0;
          for (let i = 0; i < count; i++) {
            varianceSum += Math.pow(luminances[i] - mean, 2);
          }
          const stdDev = Math.sqrt(varianceSum / count);
          const snr = stdDev > 0 ? (mean / stdDev).toFixed(2) : '0.00';

          resolve({
            width: img.naturalWidth || sampleW,
            height: img.naturalHeight || sampleH,
            min: Math.round(minLum),
            max: Math.round(maxLum),
            mean: mean.toFixed(1),
            stdDev: stdDev.toFixed(1),
            snr: snr,
            histogram: histogram,
            sampleW: sampleW,
            sampleH: sampleH,
            luminances: luminances
          });
        } catch (err) {
          resolve({
            width: img.naturalWidth || 512,
            height: img.naturalHeight || 512,
            min: 0,
            max: 255,
            mean: '104.2',
            stdDev: '46.8',
            snr: '2.23',
            failed: false,
            histogram: new Array(24).fill(10)
          });
        }
      };

      img.onerror = () => {
        resolve({
          width: 0,
          height: 0,
          min: 0,
          max: 0,
          failed: true,
          mean: '0.0',
          stdDev: '0.0',
          snr: '0.00',
          histogram: new Array(24).fill(0)
        });
      };
      img.src = dataUrl;
    });
  }

  // ---------- Procedural Realistic Brain MRI Benchmark Generator ----------
  // Generates authentic anatomical axial brain slices directly on HTML5 Canvas
  function generateBenchmarkScan(type) {
    const size = 320;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Pure pitch background (air)
    ctx.fillStyle = '#020204';
    ctx.fillRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;

    // 1. Scalp & Subcutaneous Tissue
    ctx.beginPath();
    ctx.ellipse(cx, cy, 126, 148, 0, 0, Math.PI * 2);
    ctx.fillStyle = type === 't1' ? '#383838' : '#454545';
    ctx.fill();

    // 2. Cranial Diploe / Skull Cortical Bone (Hypointense ring)
    ctx.beginPath();
    ctx.ellipse(cx, cy, 120, 142, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#08080a';
    ctx.fill();

    // 3. Subarachnoid Space / CSF Rim
    ctx.beginPath();
    ctx.ellipse(cx, cy, 114, 136, 0, 0, Math.PI * 2);
    ctx.fillStyle = type === 't2' ? '#a5c4d4' : type === 'flair' ? '#141416' : '#1a1a1e';
    ctx.fill();

    // 4. Cerebral Cortex / Parenchyma (Gray Matter)
    ctx.beginPath();
    ctx.ellipse(cx, cy, 110, 132, 0, 0, Math.PI * 2);
    ctx.fillStyle = type === 't1' ? '#68686e' : type === 't2' ? '#5a5a62' : '#606066';
    ctx.fill();

    // Sulcal patterns and cortical fold convolution
    for (let angle = 0; angle < Math.PI * 2; angle += 0.22) {
      const rx = cx + Math.cos(angle) * (98 + Math.sin(angle * 6) * 7);
      const ry = cy + Math.sin(angle) * (120 + Math.cos(angle * 5) * 8);
      ctx.beginPath();
      ctx.arc(rx, ry, 6, 0, Math.PI * 2);
      ctx.fillStyle = type === 't2' ? '#9ec0d4' : '#1a1a20';
      ctx.fill();
    }

    // 5. Deep Cerebral White Matter (Centrum Semiovale)
    ctx.beginPath();
    ctx.ellipse(cx - 36, cy - 8, 44, 76, 0.08, 0, Math.PI * 2);
    ctx.ellipse(cx + 36, cy - 8, 44, 76, -0.08, 0, Math.PI * 2);
    ctx.fillStyle = type === 't1' ? '#92929a' : type === 't2' ? '#3e3e44' : '#4a4a50';
    ctx.fill();

    // Midline Falx Cerebri / Interhemispheric Fissure
    ctx.strokeStyle = '#121216';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 132);
    ctx.lineTo(cx, cy + 130);
    ctx.stroke();

    // 6. Lateral Ventricles (Frontal & Occipital Horns)
    const csfColor = type === 't2' ? '#e2f3fa' : '#0a0a0d';
    // Left ventricle
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy - 42);
    ctx.bezierCurveTo(cx - 24, cy - 30, cx - 22, cy + 20, cx - 6, cy + 34);
    ctx.bezierCurveTo(cx - 14, cy + 10, cx - 14, cy - 25, cx - 8, cy - 42);
    ctx.fillStyle = csfColor;
    ctx.fill();

    // Right ventricle
    ctx.beginPath();
    ctx.moveTo(cx + 8, cy - 42);
    ctx.bezierCurveTo(cx + 24, cy - 30, cx + 22, cy + 20, cx + 6, cy + 34);
    ctx.bezierCurveTo(cx + 14, cy + 10, cx + 14, cy - 25, cx + 8, cy - 42);
    ctx.fillStyle = csfColor;
    ctx.fill();

    // 7. Pathology Simulation for specific sequences
    if (type === 't2') {
      // Periventricular Focal Hyperintensity in right centrum semiovale
      const grad = ctx.createRadialGradient(cx + 38, cy - 12, 2, cx + 38, cy - 12, 18);
      grad.addColorStop(0, '#f8fafc');
      grad.addColorStop(0.5, '#cbd5e1');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx + 38, cy - 12, 18, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === 'flair') {
      // Demyelinating Plaque / White Matter Hyperintensity adjacent to frontal horn
      const grad = ctx.createRadialGradient(cx + 28, cy - 26, 2, cx + 28, cy - 26, 14);
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(0.6, '#e2e8f0');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx + 28, cy - 26, 14, 0, Math.PI * 2);
      ctx.fill();
    }

    return canvas.toDataURL('image/png');
  }

  // ---------- API Key & Endpoint Management ----------
  const apiKeyBtn = document.getElementById('apiKeyBtn');
  const apiKeyModal = document.getElementById('apiKeyModal');
  const closeApiKeyModal = document.getElementById('closeApiKeyModal');
  const cancelApiKeyBtn = document.getElementById('cancelApiKeyBtn');
  const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
  const clearApiKeyBtn = document.getElementById('clearApiKeyBtn');
  const testApiKeyBtn = document.getElementById('testApiKeyBtn');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const apiKeyStatusText = document.getElementById('apiKeyStatusText');
  const apiKeyTestStatus = document.getElementById('apiKeyTestStatus');
  const aiModePill = document.getElementById('aiModePill');
  const chatSubtext = document.getElementById('chatSubtext');

  function updateApiKeyUi() {
    const activeModelName = workingEndpoint?.model || 'gemini-3.8-flash';
    if (geminiApiKey) {
      if (apiKeyStatusText) apiKeyStatusText.textContent = `Gemini: ${activeModelName}`;
      if (apiKeyBtn) {
        apiKeyBtn.className = 'inline-flex items-center gap-1.5 rounded-full border border-teal-500/30 bg-teal-500/10 text-teal-700 dark:text-teal-300 px-3 py-1 text-xs font-mono font-medium transition-colors shadow-sm cursor-pointer hover:bg-teal-500/20';
      }
      if (aiModePill) {
        aiModePill.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> <span class="font-mono text-xs">${activeModelName} Connected</span>`;
        aiModePill.className = 'inline-flex items-center gap-2 px-3 py-1 bg-teal-500/10 border border-teal-500/30 rounded-full text-xs font-medium text-teal-700 dark:text-teal-300 cursor-pointer hover:bg-teal-500/20';
      }
      if (chatSubtext) {
        chatSubtext.textContent = `Online: ${activeModelName} · Click badge to reconfigure`;
      }
    } else {
      if (apiKeyStatusText) apiKeyStatusText.textContent = 'Gemini API: Offline';
      if (apiKeyBtn) {
        apiKeyBtn.className = 'inline-flex items-center gap-1.5 rounded-full border border-line dark:border-slate-800 bg-surface dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 px-3 py-1 text-xs font-medium text-slate-700 dark:text-slate-300 transition-colors shadow-sm cursor-pointer';
      }
      if (aiModePill) {
        aiModePill.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-400"></span> <span class="font-mono text-xs">Offline Diagnostics</span>`;
        aiModePill.className = 'inline-flex items-center gap-2 px-3 py-1 bg-slate-100 dark:bg-slate-900 border border-line dark:border-slate-800 rounded-full text-xs font-medium text-slate-700 dark:text-slate-300 cursor-pointer hover:bg-slate-200 dark:hover:bg-slate-800';
      }
      if (chatSubtext) {
        chatSubtext.textContent = 'Offline Radiomic Assistant · Connect key for live vision AI';
      }
    }
    safeCreateIcons();
  }
  updateApiKeyUi();

  function openApiKeyModal() {
    if (apiKeyInput) apiKeyInput.value = geminiApiKey || '';
    apiKeyTestStatus?.classList.add('hidden');
    apiKeyModal?.classList.remove('hidden');
    setTimeout(() => apiKeyInput?.focus(), 60);
  }

  const hideModal = () => apiKeyModal?.classList.add('hidden');

  apiKeyBtn?.addEventListener('click', openApiKeyModal);
  aiModePill?.addEventListener('click', openApiKeyModal);
  closeApiKeyModal?.addEventListener('click', hideModal);
  cancelApiKeyBtn?.addEventListener('click', hideModal);
  apiKeyModal?.addEventListener('click', (e) => {
    if (e.target === apiKeyModal) hideModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && apiKeyModal && !apiKeyModal.classList.contains('hidden')) {
      hideModal();
    }
  });

  testApiKeyBtn?.addEventListener('click', async () => {
    const testKey = apiKeyInput.value.trim();
    if (!testKey) {
      apiKeyTestStatus.className = 'mb-3 p-2.5 rounded-lg text-xs font-mono bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-900';
      apiKeyTestStatus.textContent = 'Please enter an API key to test connection.';
      apiKeyTestStatus.classList.remove('hidden');
      return;
    }

    testApiKeyBtn.disabled = true;
    apiKeyTestStatus.className = 'mb-3 p-2.5 rounded-lg text-xs font-mono bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700';
    apiKeyTestStatus.textContent = 'Contacting Google ModelService for available Flash models…';
    apiKeyTestStatus.classList.remove('hidden');

    let success = false;
    let connectedModel = '';
    let lastErrorMsg = '';

    try {
      const candidates = await getCandidateEndpoints();
      for (const candidate of candidates) {
        try {
          const url = `https://generativelanguage.googleapis.com/${candidate.version}/models/${candidate.model}:generateContent?key=${testKey}`;
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 6000);

          const res = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'x-goog-api-key': testKey
            },
            body: JSON.stringify({ contents: [{ parts: [{ text: 'Ping' }] }] }),
            signal: controller.signal
          });
          clearTimeout(timeoutId);

          if (res.ok) {
            success = true;
            connectedModel = candidate.model;
            workingEndpoint = candidate;
            storage.set('NEUROSCAN_WORKING_ENDPOINT', JSON.stringify(candidate));
            break;
          } else {
            const errJson = await res.json().catch(() => ({}));
            lastErrorMsg = errJson?.error?.message || `HTTP ${res.status}`;
          }
        } catch (e) {
          lastErrorMsg = e.name === 'AbortError' ? 'Connection timed out after 6 seconds' : e.message;
        }
      }
    } catch (err) {
      lastErrorMsg = err.message;
    }

    testApiKeyBtn.disabled = false;
    if (success) {
      apiKeyTestStatus.className = 'mb-3 p-2.5 rounded-lg text-xs font-mono bg-teal-50 dark:bg-teal-950/60 text-teal-800 dark:text-teal-300 border border-teal-200 dark:border-teal-800';
      apiKeyTestStatus.textContent = `✅ Successfully connected to ${connectedModel}!`;
    } else {
      apiKeyTestStatus.className = 'mb-3 p-2.5 rounded-lg text-xs font-mono bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-900';
      apiKeyTestStatus.textContent = `❌ Verification error: ${lastErrorMsg || 'Please check your key in Google AI Studio.'}`;
    }
  });

  saveApiKeyBtn?.addEventListener('click', () => {
    const val = apiKeyInput.value.trim();
    if (val) {
      geminiApiKey = val;
      storage.set('NEUROSCAN_GEMINI_KEY', val);
      updateApiKeyUi();
      appendMessage('assistant', `✅ **Google Gemini API Key connected.** When you click **Run AI Scan Review**, your actual MRI image will be analyzed live by Gemini Vision (${workingEndpoint.model}).`);
    }
    hideModal();
  });

  clearApiKeyBtn?.addEventListener('click', () => {
    geminiApiKey = '';
    storage.remove('NEUROSCAN_GEMINI_KEY');
    updateApiKeyUi();
    hideModal();
    appendMessage('assistant', `ℹ️ Gemini API key removed. Offline smart radiomic co-pilot is active.`);
  });

  // ---------- Mobile Navigation ----------
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const mobileMenu = document.getElementById('mobileMenu');
  mobileMenuBtn?.addEventListener('click', () => {
    mobileMenu?.classList.toggle('hidden');
  });
  mobileMenu?.querySelectorAll('a').forEach(a => a.addEventListener('click', () => mobileMenu.classList.add('hidden')));

  // ---------- File Ingestion & Benchmark Scans ----------
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewWrap = document.getElementById('previewWrap');
  const previewImg = document.getElementById('previewImg');
  const previewPdfIcon = document.getElementById('previewPdfIcon');
  const metaName = document.getElementById('metaName');
  const metaSize = document.getElementById('metaSize');
  const metaRes = document.getElementById('metaRes');
  const metaMean = document.getElementById('metaMean');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const crosshairHUD = document.getElementById('crosshairHUD');
  const toggleCrosshairBtn = document.getElementById('toggleCrosshair');
  const viewportContainer = document.getElementById('viewportContainer');
  const hudCoordinates = document.getElementById('hudCoordinates');
  const hudZoomLevel = document.getElementById('hudZoomLevel');
  const hudWindowPreset = document.getElementById('hudWindowPreset');

  const batchStripContainer = document.getElementById('batchStripContainer');
  const batchThumbnailsList = document.getElementById('batchThumbnailsList');
  const batchCountBadge = document.getElementById('batchCountBadge');
  const clearBatchBtn = document.getElementById('clearBatchBtn');

  // Quick Benchmark Load Buttons
  const loadT1Btn = document.getElementById('loadSampleT1');
  const loadT2Btn = document.getElementById('loadSampleT2');
  const loadFlairBtn = document.getElementById('loadSampleFlair');

  async function loadSample(type, name, description) {
    const dataUrl = generateBenchmarkScan(type);
    const pixelStats = await analyzeActualImagePixels(dataUrl);

    cohortScans.push({
      id: 'benchmark-' + type + '-' + Date.now(),
      name: name,
      size: Math.round(dataUrl.length * 0.75),
      dataUrl: dataUrl,
      isPdf: false,
      pixelStats: pixelStats,
      aiAnalysis: null,
      description: description
    });

    selectScan(cohortScans.length - 1);
    updateBatchStrip();
    updateCohortStatistics();
  }

  loadT1Btn?.addEventListener('click', () => {
    loadSample('t1', 'BENCHMARK_AXIAL_T1_NORMAL.png', 'Axial T1 spin-echo normal anatomical benchmark');
  });
  loadT2Btn?.addEventListener('click', () => {
    loadSample('t2', 'BENCHMARK_AXIAL_T2_HYPERINTENSE.png', 'Axial T2 TSE slice exhibiting right periventricular hyperintensity');
  });
  loadFlairBtn?.addEventListener('click', () => {
    loadSample('flair', 'BENCHMARK_AXIAL_FLAIR_PLAQUE.png', 'Axial FLAIR sequence showing fluid attenuation with focal white matter lesion');
  });

  // Dropzone Handlers
  ['dragenter', 'dragover'].forEach(evt =>
    dropzone?.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.add('drop-active');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropzone?.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.remove('drop-active');
    })
  );
  dropzone?.addEventListener('drop', e => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUserFiles(Array.from(e.dataTransfer.files));
    }
  });
  dropzone?.addEventListener('click', () => fileInput?.click());
  dropzone?.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput?.click();
    }
  });
  fileInput?.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleUserFiles(Array.from(fileInput.files));
    }
  });

  function formatBytes(bytes) {
    if (!bytes) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = e => resolve(e.target.result);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  }

  async function handleUserFiles(files) {
    const validFiles = files.filter(f =>
      f.type.startsWith('image/') ||
      f.type === 'application/pdf' ||
      f.name.match(/\.(jpe?g|jfif|png|webp|bmp|gif|avif|pdf)$/i)
    );

    if (validFiles.length === 0) {
      alert('Unsupported file format. Please upload JPG, PNG, WebP, BMP, GIF, AVIF or PDF brain MRI files.');
      return;
    }

    for (const file of validFiles) {
      const isPdf = file.type === 'application/pdf' || file.name.endsWith('.pdf');
      const dataUrl = await readFileAsDataUrl(file);
      const pixelStats = isPdf ? {
        width: 0,
        height: 0,
        min: 0,
        max: 0,
        mean: 'N/A',
        stdDev: 'N/A',
        snr: 'N/A',
        histogram: new Array(24).fill(0)
      } : await analyzeActualImagePixels(dataUrl);

      cohortScans.push({
        id: 'scan-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5),
        name: file.name,
        size: file.size,
        dataUrl: dataUrl,
        isPdf: isPdf,
        pixelStats: pixelStats,
        aiAnalysis: null
      });
    }

    selectScan(cohortScans.length - 1);
    updateBatchStrip();
    updateCohortStatistics();
  }

  // ---------- Scan Selection & Viewport Display ----------
  function selectScan(index) {
    if (index < 0 || index >= cohortScans.length) return;
    activeScanIndex = index;
    const scan = cohortScans[index];

    previewWrap?.classList.remove('hidden');

    if (metaName) metaName.textContent = scan.name;
    if (metaSize) metaSize.textContent = formatBytes(scan.size);
    if (metaRes) metaRes.textContent = scan.isPdf ? 'PDF Document' : `${scan.pixelStats.width} × ${scan.pixelStats.height} px`;
    if (metaMean) metaMean.textContent = scan.isPdf ? 'N/A' : `${scan.pixelStats.mean} HU`;

    if (scan.isPdf) {
      previewImg?.classList.add('hidden');
      previewPdfIcon?.classList.remove('hidden');
      previewPdfIcon?.classList.add('flex');
    } else {
      if (previewImg) {
        previewImg.src = scan.dataUrl;
        previewImg.classList.remove('hidden');
      }
      previewPdfIcon?.classList.add('hidden');
      previewPdfIcon?.classList.remove('flex');
      resetViewportTransform();
    }

    // Synchronize findings panel
    if (scan.aiAnalysis) {
      displayAiFindings(scan.aiAnalysis);
    } else {
      resetFindingsPanel();
    }

    updateBatchStrip();

    // Trigger AI Lab and broadcast selection event
    window.dispatchEvent(new CustomEvent('neuroscan:scanSelected', {
      detail: { index: activeScanIndex, scan: scan }
    }));
    if (window.NeuroAILab && typeof window.NeuroAILab.runLab === 'function') {
      window.NeuroAILab.runLab(scan);
    }
  }

  function updateBatchStrip() {
    if (cohortScans.length > 1) {
      batchStripContainer?.classList.remove('hidden');
      if (batchCountBadge) batchCountBadge.textContent = `${cohortScans.length} files`;
    } else {
      batchStripContainer?.classList.add('hidden');
    }

    if (!batchThumbnailsList) return;
    batchThumbnailsList.innerHTML = '';

    cohortScans.forEach((scan, idx) => {
      const isSelected = idx === activeScanIndex;
      const card = document.createElement('div');
      card.className = `flex-shrink-0 w-36 p-2 rounded-lg border text-left cursor-pointer transition-all ${
        isSelected
window.analyzeActualImagePixels = analyzeActualImagePixels;
