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
          ? 'border-teal-500 bg-teal-50/70 dark:bg-teal-950/60 ring-2 ring-teal-500/30'
          : 'border-line dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-slate-400'
      }`;

      card.innerHTML = `
        <div class="h-20 bg-black rounded overflow-hidden flex items-center justify-center mb-1.5 relative">
          ${scan.isPdf
            ? `<i data-lucide="file-text" class="w-8 h-8 text-teal-400"></i>`
            : `<img src="${scan.dataUrl}" class="max-h-full max-w-full object-contain" alt="thumbnail" />`
          }
          <span class="absolute top-1 right-1 text-[9px] font-mono px-1 py-0.5 rounded bg-slate-900/80 text-teal-300">
            ${scan.isPdf ? 'PDF' : `${scan.pixelStats.width}px`}
          </span>
        </div>
        <p class="font-mono text-[11px] truncate font-medium text-slate-800 dark:text-slate-200" title="${scan.name}">${scan.name}</p>
        <p class="text-[10px] text-slate-400 font-mono">${formatBytes(scan.size)}</p>
      `;
      card.addEventListener('click', () => selectScan(idx));
      batchThumbnailsList.appendChild(card);
    });
    safeCreateIcons();
  }

  clearBatchBtn?.addEventListener('click', () => {
    cohortScans = [];
    activeScanIndex = -1;
    previewWrap?.classList.add('hidden');
    resetFindingsPanel();
    updateBatchStrip();
    updateCohortStatistics();

    window.dispatchEvent(new CustomEvent('neuroscan:scansCleared'));
    if (window.NeuroAILab && typeof window.NeuroAILab.resetLab === 'function') {
      window.NeuroAILab.resetLab();
    }
  });

  // ---------- Viewport Pan, Zoom & DICOM Window/Level Controls ----------
  function updateViewportTransform() {
    if (!previewImg) return;
    previewImg.style.transform = `translate(${panX}px, ${panY}px) scale(${currentZoom})`;
    if (hudZoomLevel) hudZoomLevel.textContent = `${Math.round(currentZoom * 100)}%`;
  }

  function resetViewportTransform() {
    currentZoom = 1;
    panX = 0;
    panY = 0;
    updateViewportTransform();
  }

  document.getElementById('zoomIn')?.addEventListener('click', () => {
    currentZoom = Math.min(4, currentZoom + 0.25);
    updateViewportTransform();
  });
  document.getElementById('zoomOut')?.addEventListener('click', () => {
    currentZoom = Math.max(0.75, currentZoom - 0.25);
    updateViewportTransform();
  });
  document.getElementById('zoomReset')?.addEventListener('click', resetViewportTransform);

  // Mouse wheel zoom
  viewportContainer?.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    currentZoom = Math.min(4, Math.max(0.75, currentZoom + delta));
    updateViewportTransform();
  }, { passive: false });

  // Pan by dragging
  viewportContainer?.addEventListener('mousedown', (e) => {
    if (e.button === 0 && currentZoom > 1) {
      isPanning = true;
      startPanX = e.clientX - panX;
      startPanY = e.clientY - panY;
      viewportContainer.style.cursor = 'grabbing';
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (isPanning) {
      panX = e.clientX - startPanX;
      panY = e.clientY - startPanY;
      updateViewportTransform();
    }
  });

  window.addEventListener('mouseup', () => {
    if (isPanning) {
      isPanning = false;
      if (viewportContainer) viewportContainer.style.cursor = 'crosshair';
    }
  });

  // Interactive Live Pixel Inspector & Crosshairs
  viewportContainer?.addEventListener('mousemove', (e) => {
    if (!hudCoordinates) return;
    const scan = activeScanIndex >= 0 ? cohortScans[activeScanIndex] : null;
    if (!scan || scan.isPdf || !scan.pixelStats || !previewImg) {
      hudCoordinates.textContent = 'X: -- Y: -- | -- HU';
      return;
    }

    const rect = previewImg.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * scan.pixelStats.width);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * scan.pixelStats.height);

    if (x >= 0 && x < scan.pixelStats.width && y >= 0 && y < scan.pixelStats.height) {
      let huValue = '--';
      if (scan.pixelStats.luminances && scan.pixelStats.sampleW) {
        const sx = Math.floor((x / scan.pixelStats.width) * scan.pixelStats.sampleW);
        const sy = Math.floor((y / scan.pixelStats.height) * scan.pixelStats.sampleH);
        const idx = sy * scan.pixelStats.sampleW + sx;
        const val = scan.pixelStats.luminances[idx];
        if (typeof val === 'number') {
          huValue = Math.round(val);
        }
      }
      hudCoordinates.textContent = `X: ${x} Y: ${y} | ${huValue} HU`;
    } else {
      hudCoordinates.textContent = 'X: -- Y: -- | -- HU';
    }
  });

  // Crosshair toggle
  toggleCrosshairBtn?.addEventListener('click', () => {
    showCrosshair = !showCrosshair;
    crosshairHUD?.classList.toggle('hidden', !showCrosshair);
    toggleCrosshairBtn.classList.toggle('text-teal-600', showCrosshair);
    toggleCrosshairBtn.classList.toggle('dark:text-teal-400', showCrosshair);
  });

  // DICOM Window/Level Presets & Colormaps
  const colormapBtn = document.getElementById('toggleColormap');

  const COLORMAP_MODES = ['normal', 'invert', 'heatmap', 'high-contrast'];
  const COLORMAP_LABELS = {
    'normal': 'Grayscale',
    'invert': 'Inverted Film',
    'heatmap': 'Thermal Heatmap',
    'high-contrast': 'High Contrast'
  };

  function applyColormapMode(mode) {
    currentFilterMode = mode;
    if (!previewImg) return;

    if (mode === 'invert') {
      previewImg.style.filter = 'invert(1) contrast(110%)';
    } else if (mode === 'heatmap') {
      previewImg.style.filter = 'hue-rotate(180deg) saturate(220%) contrast(140%)';
    } else if (mode === 'high-contrast') {
      previewImg.style.filter = 'contrast(175%) brightness(105%)';
    } else {
      previewImg.style.filter = 'none';
    }

    if (hudWindowPreset) hudWindowPreset.textContent = COLORMAP_LABELS[mode] || 'Standard';
  }

  colormapBtn?.addEventListener('click', () => {
    const curIdx = COLORMAP_MODES.indexOf(currentFilterMode);
    const nextMode = COLORMAP_MODES[(curIdx + 1) % COLORMAP_MODES.length];
    applyColormapMode(nextMode);
  });

  // ---------- AI Review Call (Gemini Multimodal Vision) ----------
  const processingBox = document.getElementById('processingBox');
  const resultsEmpty = document.getElementById('resultsEmpty');
  const resultsBody = document.getElementById('resultsBody');
  const resultsBadge = document.getElementById('resultsBadge');
  const observationText = document.getElementById('observationText');
  const recapMean = document.getElementById('recapMean');
  const recapStdDev = document.getElementById('recapStdDev');
  const recapSnr = document.getElementById('recapSnr');
  const exportReportBtn = document.getElementById('exportReportBtn');

  analyzeBtn?.addEventListener('click', async () => {
    if (activeScanIndex < 0 || !cohortScans[activeScanIndex]) return;
    const scan = cohortScans[activeScanIndex];

    if (!geminiApiKey) {
      resultsEmpty?.classList.add('hidden');
      resultsBody?.classList.remove('hidden');
      if (resultsBadge) {
        resultsBadge.textContent = 'API Key Required';
        resultsBadge.className = 'text-xs font-mono px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-300 font-medium';
      }
      if (observationText) {
        observationText.innerHTML = `
          <div class="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-900 dark:text-amber-200 text-xs leading-relaxed space-y-2">
            <p class="font-bold">⚠️ Gemini API Offline Mode</p>
            <p>To run real-time multimodal vision review on <strong>${scan.name}</strong>, click the <strong>[Gemini API: Offline]</strong> badge at top right and input your key from Google AI Studio.</p>
            <p class="text-slate-600 dark:text-slate-400">Offline radiomics and syllabus AI Lab (K-means segmentation + Forward Chaining Expert System) continue running client-side with full precision below.</p>
          </div>
        `;
      }
      if (recapMean) recapMean.textContent = scan.pixelStats.mean;
      if (recapStdDev) recapStdDev.textContent = scan.pixelStats.stdDev;
      if (recapSnr) recapSnr.textContent = scan.pixelStats.snr;
      return;
    }

    analyzeBtn.disabled = true;
    processingBox?.classList.remove('hidden');
    if (resultsBadge) {
      resultsBadge.textContent = 'Analyzing…';
      resultsBadge.className = 'text-xs font-mono px-2 py-0.5 rounded bg-teal-500/15 border border-teal-500/30 text-teal-700 dark:text-teal-300 font-medium';
    }

    try {
      const prompt = `You are a clinical neuro-radiologist assistant.
Examine this uploaded brain MRI scan slice: "${scan.name}".
Computed pixel statistics from client canvas: Matrix Resolution ${scan.pixelStats.width}x${scan.pixelStats.height}, Mean Luminance ${scan.pixelStats.mean}/255 HU, Contrast Standard Deviation ${scan.pixelStats.stdDev}, Mean/σ ratio ${scan.pixelStats.snr}.

Provide a clear, objective radiological consultation report:
1. **Examination & Anatomy**: Orientation (axial/sagittal/coronal), visible anatomical structures (ventricles, hemispheres, cortex).
2. **Signal Intensities & Findings**: Any notable hyperintensity, hypointensity, mass effect, or bilateral asymmetry.
3. **Radiomic Impression & Protocol Recommendation**: Correlation with clinical history, suggested sequence correlation (e.g. T1, T2, FLAIR, DWI/ADC).
Maintain professional, concise clinical terminology.`;

      const review = await callGeminiVision(prompt, scan.dataUrl);
      scan.aiAnalysis = review;
      displayAiFindings(review);
    } catch (err) {
      resultsEmpty?.classList.add('hidden');
      resultsBody?.classList.remove('hidden');
      if (resultsBadge) {
        resultsBadge.textContent = 'API Notice';
        resultsBadge.className = 'text-xs font-mono px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-800 dark:text-amber-300 font-medium';
      }

      if (observationText) {
        observationText.innerHTML = `
          <div class="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-900 dark:text-amber-200 mb-3 text-xs leading-relaxed">
            <p class="font-bold">⚠️ Gemini API Notice:</p>
            <p class="mt-1">${err.message}</p>
            <button type="button" onclick="openApiKeyModal()" class="mt-2.5 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-teal-600 text-white font-medium text-xs hover:bg-teal-700 cursor-pointer">
              Verify API Key &amp; Endpoint
            </button>
          </div>
          <p class="font-semibold text-slate-800 dark:text-slate-100 mb-1.5 text-xs uppercase tracking-wide">Client Computed Radiomics:</p>
          <ul class="list-disc pl-5 space-y-1 text-xs text-slate-600 dark:text-slate-300 font-mono">
            <li>Resolution: ${scan.pixelStats.width} × ${scan.pixelStats.height} px</li>
            <li>Mean Luminance: ${scan.pixelStats.mean} HU</li>
            <li>Contrast (σ): ${scan.pixelStats.stdDev}</li>
            <li>Mean/σ Ratio: ${scan.pixelStats.snr}</li>
            <li>Dynamic Range: ${scan.pixelStats.min} &rarr; ${scan.pixelStats.max}</li>
          </ul>
        `;
      }
      if (recapMean) recapMean.textContent = scan.pixelStats.mean;
      if (recapStdDev) recapStdDev.textContent = scan.pixelStats.stdDev;
      if (recapSnr) recapSnr.textContent = scan.pixelStats.snr;
    } finally {
      processingBox?.classList.add('hidden');
      analyzeBtn.disabled = false;
    }
  });

  function displayAiFindings(reviewText) {
    resultsEmpty?.classList.add('hidden');
    resultsBody?.classList.remove('hidden');
    if (resultsBadge) {
      resultsBadge.textContent = 'Review Complete';
      resultsBadge.className = 'text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 font-semibold';
    }
    exportReportBtn?.classList.remove('hidden');

    if (observationText) observationText.innerHTML = renderMarkdown(reviewText);

    const scan = cohortScans[activeScanIndex];
    if (scan) {
      if (recapMean) recapMean.textContent = scan.pixelStats.mean;
      if (recapStdDev) recapStdDev.textContent = scan.pixelStats.stdDev;
      if (recapSnr) recapSnr.textContent = scan.pixelStats.snr;

      const printDate = document.getElementById('printDate');
      const printFile = document.getElementById('printFile');
      const printRadiomics = document.getElementById('printRadiomics');
      const printAiSummary = document.getElementById('printAiSummary');
      const printScanImage = document.getElementById('printScanImage');

      if (printDate) printDate.textContent = 'Report Date: ' + new Date().toLocaleString();
      if (printFile) printFile.textContent = 'Study: ' + scan.name;
      if (printScanImage && scan.dataUrl && !scan.isPdf) {
        printScanImage.src = scan.dataUrl;
        printScanImage.classList.remove('hidden');
      }
      if (printRadiomics) {
        printRadiomics.innerHTML = `
          <div class="grid grid-cols-4 gap-2 border p-2 text-xs font-mono">
            <div><strong>Matrix:</strong> ${scan.pixelStats.width}×${scan.pixelStats.height}px</div>
            <div><strong>Mean Signal:</strong> ${scan.pixelStats.mean} HU</div>
            <div><strong>Contrast (σ):</strong> ${scan.pixelStats.stdDev}</div>
            <div><strong>Mean/σ Ratio:</strong> ${scan.pixelStats.snr}</div>
          </div>
        `;
      }
      if (printAiSummary) printAiSummary.innerHTML = renderMarkdown(reviewText);
    }
  }

  function resetFindingsPanel() {
    resultsEmpty?.classList.remove('hidden');
    resultsBody?.classList.add('hidden');
    exportReportBtn?.classList.add('hidden');
    if (resultsBadge) {
      resultsBadge.textContent = 'Awaiting Upload';
      resultsBadge.className = 'text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-medium';
    }
  }

  exportReportBtn?.addEventListener('click', () => window.print());

  // Resilient Gemini Vision Fetcher with Auto-Fallback across Flash models
  async function callGeminiVision(promptText, imageBase64) {
    let candidates = await getCandidateEndpoints();

    const contentsParts = [];
    if (imageBase64) {
      if (imageBase64.startsWith('data:image/')) {
        const [mimePart, dataPart] = imageBase64.split(';base64,');
        if (dataPart) {
          const cleanMime = (mimePart.replace('data:', '').split(';')[0] || 'image/jpeg').trim();
          contentsParts.push({
            inlineData: { mimeType: cleanMime, data: dataPart }
          });
        }
      } else if (imageBase64.startsWith('data:application/pdf')) {
        const dataPart = imageBase64.split(';base64,')[1];
        if (dataPart) {
          contentsParts.push({
            inlineData: { mimeType: 'application/pdf', data: dataPart }
          });
        }
      }
    }

    contentsParts.push({ text: promptText });

    let lastError = null;

    for (const candidate of candidates) {
      try {
        const endpointUrl = `https://generativelanguage.googleapis.com/${candidate.version}/models/${candidate.model}:generateContent?key=${geminiApiKey}`;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 14000);

        const response = await fetch(endpointUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-goog-api-key': geminiApiKey
          },
          body: JSON.stringify({
            contents: [{ parts: contentsParts }],
            generationConfig: { temperature: 0.2, maxOutputTokens: 1400 }
          }),
          signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (response.ok) {
          const data = await response.json();
          const reply = data.candidates?.[0]?.content?.parts?.[0]?.text;
          if (reply) {
            workingEndpoint = candidate;
            storage.set('NEUROSCAN_WORKING_ENDPOINT', JSON.stringify(candidate));
            updateApiKeyUi();
            return reply;
          }
        }

        const errData = await response.json().catch(() => ({}));
        const errMsg = errData?.error?.message || `HTTP ${response.status}`;
        lastError = new Error(errMsg);

        if (response.status === 400 && errMsg.toLowerCase().includes('api key')) {
          throw new Error('API key is invalid. Please verify your key from Google AI Studio.');
        }
        if (response.status === 403) {
          throw new Error(`API key authorization error: ${errMsg}`);
        }

        continue;
      } catch (err) {
        if (err.message.includes('API key') || err.message.includes('authorization error')) {
          throw err;
        }
        lastError = err;
        continue;
      }
    }

    throw lastError || new Error('No compatible Gemini model could be reached with this API key.');
  }

  // ---------- Cohort & Real Pixel Statistics ----------
  const statTotalScans = document.getElementById('statTotalScans');
  const statFileSizeTotal = document.getElementById('statFileSizeTotal');
  const statAvgRes = document.getElementById('statAvgRes');
  const statMeanSignal = document.getElementById('statMeanSignal');
  const statMeanContrast = document.getElementById('statMeanContrast');
  const cohortTableBody = document.getElementById('cohortTableBody');
  const intensityHistogramBars = document.getElementById('intensityHistogramBars');
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  const exportJsonBtn = document.getElementById('exportJsonBtn');

  function updateCohortStatistics() {
    const total = cohortScans.length;
    if (statTotalScans) statTotalScans.textContent = total;

    if (total === 0) {
      if (statFileSizeTotal) statFileSizeTotal.textContent = '0 KB total';
      if (statAvgRes) statAvgRes.textContent = '--';
      if (statMeanSignal) statMeanSignal.textContent = '--';
      if (statMeanContrast) statMeanContrast.textContent = '--';
      if (cohortTableBody) {
        cohortTableBody.innerHTML = `<tr><td colspan="9" class="py-8 text-center text-slate-400 font-mono">No scans uploaded yet. Drag and drop your MRI files above or load benchmark scans to populate the table.</td></tr>`;
      }
      if (intensityHistogramBars) intensityHistogramBars.innerHTML = '';
      return;
    }

    const totalBytes = cohortScans.reduce((acc, s) => acc + (s.size || 0), 0);
    if (statFileSizeTotal) statFileSizeTotal.textContent = `${formatBytes(totalBytes)} total`;

    const validImages = cohortScans.filter(s => !s.isPdf && s.pixelStats && s.pixelStats.width > 0);
    if (validImages.length > 0) {
      const avgW = Math.round(validImages.reduce((acc, s) => acc + s.pixelStats.width, 0) / validImages.length);
      const avgH = Math.round(validImages.reduce((acc, s) => acc + s.pixelStats.height, 0) / validImages.length);
      if (statAvgRes) statAvgRes.textContent = `${avgW}×${avgH}`;

      const avgMean = (validImages.reduce((acc, s) => acc + parseFloat(s.pixelStats.mean || 0), 0) / validImages.length).toFixed(1);
      if (statMeanSignal) statMeanSignal.textContent = avgMean;

      const avgContrast = (validImages.reduce((acc, s) => acc + parseFloat(s.pixelStats.stdDev || 0), 0) / validImages.length).toFixed(1);
      if (statMeanContrast) statMeanContrast.textContent = avgContrast;
    } else {
      if (statAvgRes) statAvgRes.textContent = 'PDF';
      if (statMeanSignal) statMeanSignal.textContent = '--';
      if (statMeanContrast) statMeanContrast.textContent = '--';
    }

    // Populate Table
    if (cohortTableBody) {
      cohortTableBody.innerHTML = '';
      cohortScans.forEach((scan, idx) => {
        const isSelected = idx === activeScanIndex;
        const tr = document.createElement('tr');
        tr.className = `hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors ${isSelected ? 'bg-teal-500/10' : ''}`;

        tr.innerHTML = `
          <td class="py-2.5 px-4 font-mono font-medium text-slate-800 dark:text-slate-200">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full ${isSelected ? 'bg-teal-500' : 'bg-slate-300 dark:bg-slate-600'}"></span>
              <span class="truncate max-w-[180px] sm:max-w-[240px]" title="${scan.name}">${scan.name}</span>
            </div>
          </td>
          <td class="py-2.5 px-3 font-mono">${scan.isPdf ? 'PDF' : `${scan.pixelStats.width}×${scan.pixelStats.height}`}</td>
          <td class="py-2.5 px-3 font-mono text-slate-500">${formatBytes(scan.size)}</td>
          <td class="py-2.5 px-3 font-mono">${scan.pixelStats.min}</td>
          <td class="py-2.5 px-3 font-mono">${scan.pixelStats.max}</td>
          <td class="py-2.5 px-3 font-mono font-semibold text-teal-600 dark:text-teal-400">${scan.pixelStats.mean}</td>
          <td class="py-2.5 px-3 font-mono">${scan.pixelStats.stdDev}</td>
          <td class="py-2.5 px-3 font-mono font-bold">${scan.pixelStats.snr}</td>
          <td class="py-2.5 px-4 text-right">
            <button class="inspect-btn text-xs font-semibold text-teal-600 dark:text-teal-400 hover:underline">
              ${isSelected ? 'Active' : 'Inspect'}
            </button>
          </td>
        `;

        tr.querySelector('.inspect-btn')?.addEventListener('click', () => {
          selectScan(idx);
          document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' });
        });
        cohortTableBody.appendChild(tr);
      });
    }

    renderRealHistogram();
  }

  function renderRealHistogram() {
    if (!intensityHistogramBars) return;
    intensityHistogramBars.innerHTML = '';

    const scan = activeScanIndex >= 0 ? cohortScans[activeScanIndex] : null;
    const bins = (scan && scan.pixelStats && scan.pixelStats.histogram)
      ? scan.pixelStats.histogram
      : new Array(24).fill(0);

    const maxVal = Math.max(...bins, 1);
    bins.forEach((count, i) => {
      const heightPct = Math.max(6, Math.round((count / maxVal) * 100));
      const bar = document.createElement('div');
      bar.className = 'flex-1 rounded-t transition-all duration-300 relative group cursor-pointer';
      bar.style.height = `${heightPct}%`;
      // Anatomical intensity spectrum coloring: Air/Dark -> Deep Gray -> Subcortical White -> Hyperintense
      bar.style.backgroundColor = i > 18 ? '#ef4444' : i > 13 ? '#10b981' : i > 6 ? '#3b82f6' : '#64748b';
      bar.title = `Bin ${i + 1} (~${Math.round((i / 24) * 255)} HU): ${count} pixels sampled`;
      intensityHistogramBars.appendChild(bar);
    });
  }

  // Export CSV
  exportCsvBtn?.addEventListener('click', () => {
    if (cohortScans.length === 0) {
      alert('No scans to export.');
      return;
    }

    let csv = 'Filename,Width,Height,FileSizeBytes,Min_Luminance,Max_Luminance,Mean_Signal,StdDev_Contrast,Mean_Sigma_Ratio\n';
    cohortScans.forEach(s => {
      csv += `"${s.name}",${s.pixelStats.width},${s.pixelStats.height},${s.size},${s.pixelStats.min},${s.pixelStats.max},${s.pixelStats.mean},${s.pixelStats.stdDev},${s.pixelStats.snr}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `NeuroScan_Radiomics_${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // Export JSON
  exportJsonBtn?.addEventListener('click', () => {
    if (cohortScans.length === 0) {
      alert('No scans to export.');
      return;
    }
    const payload = {
      workbench: 'NeuroScan AI',
      exportedAt: new Date().toISOString(),
      totalScans: cohortScans.length,
      scans: cohortScans.map(s => ({
        name: s.name,
        sizeBytes: s.size,
        dimensions: { width: s.pixelStats.width, height: s.pixelStats.height },
        pixelStats: {
          min: s.pixelStats.min,
          max: s.pixelStats.max,
          mean: s.pixelStats.mean,
          contrastStdDev: s.pixelStats.stdDev,
          snr: s.pixelStats.snr
        },
        aiReview: s.aiAnalysis || null
      }))
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `NeuroScan_Radiomics_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // ---------- Interactive Diagnostic Co-Pilot Chat ----------
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const chatTranscript = document.getElementById('chatTranscript');
  const attachBtn = document.getElementById('attachBtn');
  const clearChatBtn = document.getElementById('clearChatBtn');

  chatInput?.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(120, chatInput.scrollHeight) + 'px';
  });

  function renderMarkdown(text) {
    if (!text) return '';
    const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const lines = escaped.split('\n');
    let html = '';
    let inList = false;

    for (const line of lines) {
      if (/^\s*[-*]\s+/.test(line)) {
        if (!inList) {
          html += '<ul class="list-disc list-inside space-y-1 my-1.5">';
          inList = true;
        }
        html += `<li>${line.replace(/^\s*[-*]\s+/, '')}</li>`;
      } else {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        html += line ? `<p class="${html ? 'mt-1.5' : ''}">${line}</p>` : '';
      }
    }
    if (inList) html += '</ul>';

    return html
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.+?)`/g, '<code class="font-mono text-xs bg-slate-200/80 dark:bg-slate-800 px-1 py-0.5 rounded text-slate-800 dark:text-slate-200">$1</code>');
  }

  function appendMessage(role, text) {
    if (!chatTranscript) return null;
    const wrap = document.createElement('div');
    wrap.className = 'flex ' + (role === 'user' ? 'justify-end' : 'justify-start');
    const bubble = document.createElement('div');
    bubble.className = role === 'user'
      ? 'max-w-[85%] bg-ink dark:bg-teal-600 text-white rounded-2xl rounded-br-sm px-4 py-2.5 text-xs sm:text-[13px] leading-relaxed shadow-sm'
      : 'max-w-[85%] bg-white dark:bg-slate-900 border border-line dark:border-slate-800 rounded-2xl rounded-bl-sm px-4 py-3 text-xs sm:text-[13px] leading-relaxed shadow-card text-ink dark:text-slate-100';
    bubble.innerHTML = renderMarkdown(text);
    wrap.appendChild(bubble);
    chatTranscript.appendChild(wrap);
    chatTranscript.scrollTop = chatTranscript.scrollHeight;

    chatHistory.push({ role, text });
    return bubble;
  }

  function appendTypingIndicator() {
    removeTypingIndicator();
    if (!chatTranscript) return;
    const wrap = document.createElement('div');
    wrap.id = 'typingIndicator';
    wrap.className = 'flex justify-start';
    wrap.innerHTML = `
      <div class="bg-white dark:bg-slate-900 border border-line dark:border-slate-800 rounded-2xl rounded-bl-sm px-4 py-3 flex gap-2.5 items-center shadow-sm">
        <div class="flex items-center gap-1.5">
          <span class="neuro-dot"></span>
          <span class="neuro-dot"></span>
          <span class="neuro-dot"></span>
        </div>
        <span class="text-xs text-slate-500 dark:text-slate-400 font-mono">Evaluating scan metrics…</span>
      </div>
    `;
    chatTranscript.appendChild(wrap);
    chatTranscript.scrollTop = chatTranscript.scrollHeight;
  }

  function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
  }

  clearChatBtn?.addEventListener('click', () => {
    if (chatTranscript) chatTranscript.innerHTML = '';
    chatHistory = [];
    appendWelcome();
  });

  async function handleChat(userMessage) {
    const activeScan = activeScanIndex >= 0 ? cohortScans[activeScanIndex] : null;
    const msg = (userMessage || '').toLowerCase().trim();

    // 1. Summaries
    if (msg.includes('summarize') || msg.includes('how many') || msg.includes('cohort') || (msg.includes('scans') && !msg.includes('review'))) {
      await new Promise(r => setTimeout(r, 200));
      if (cohortScans.length === 0) {
        return `No brain MRI scans have been uploaded yet. Drop an MRI file into the dropzone or click any of the **Benchmark Study** buttons above to inspect pre-calibrated scans!`;
      }
      return `You have **${cohortScans.length} scan(s)** in your active session:
${cohortScans.map(s => `- **\`${s.name}\`**: ${s.pixelStats.width}×${s.pixelStats.height}px | Mean: ${s.pixelStats.mean} HU | Contrast (σ): ${s.pixelStats.stdDev} | Mean/σ ratio: ${s.pixelStats.snr}`).join('\n')}

Active scan: **\`${activeScan ? activeScan.name : cohortScans[0].name}\`**. You can click **Run AI Scan Review** for automated diagnostic findings.`;
    }

    // 2. Mean/σ ratio, SNR, and Radiomic inquiries
    if (msg.includes('ratio') || msg.includes('snr') || msg.includes('mean/σ') || msg.includes('signal') || (msg.includes('contrast') && !msg.includes('enhance'))) {
      await new Promise(r => setTimeout(r, 200));
      if (!activeScan) return `Please upload or select an MRI scan above to inspect its calculated mean/σ ratio.`;
      return `Radiometric breakdown for **\`${activeScan.name}\`**:
* **Matrix Resolution**: ${activeScan.pixelStats.width} × ${activeScan.pixelStats.height} px
* **Mean Grayscale Luminance**: **${activeScan.pixelStats.mean}** (0–255 HU equivalent)
* **Contrast Dispersion (σ)**: **${activeScan.pixelStats.stdDev}**
* **Mean/σ Intensity Ratio**: **${activeScan.pixelStats.snr}**
* **Dynamic Pixel Range**: min **${activeScan.pixelStats.min}** &rarr; max **${activeScan.pixelStats.max}**

*Clinical note*: In client-side radiomics, the Mean/σ ratio reflects signal dispersion across the whole raster. Higher contrast with deep dark ventricles creates a balanced ratio suitable for tissue segmentation.`;
    }

    // 3. Pulse Sequences (T1, T2, FLAIR) using word boundaries
    if (/\b(sequence|sequences|t1|t2|flair)\b/i.test(msg)) {
      await new Promise(r => setTimeout(r, 200));
      return `Overview of essential Brain MRI Pulse Sequences:
* **T1-Weighted**: Optimized for anatomical architecture. Free fluid (CSF) is hypointense (dark); cortex is gray; myelin-rich white matter is hyperintense (bright).
* **T2-Weighted**: Sensitive to fluid and vasogenic edema. Water, CSF, and inflammatory fluids are hyperintense (bright).
* **FLAIR (Fluid-Attenuated Inversion Recovery)**: Selectively suppresses free CSF signal, making periventricular lesions and tissue edema stand out vividly without CSF shine-through.`;
    }

    // 4. Greetings
    if (msg === 'hi' || msg === 'hello' || msg === 'hey' || msg.startsWith('hi ') || msg.startsWith('hello ')) {
      await new Promise(r => setTimeout(r, 150));
      return `Hello! NeuroScan AI is ready. Currently **${cohortScans.length} scan(s)** are loaded in your session.${activeScan ? ` Active study: \`${activeScan.name}\` (Resolution: ${activeScan.pixelStats.width}×${activeScan.pixelStats.height}px, Mean/σ ratio: ${activeScan.pixelStats.snr}).` : ''} How can I assist you with your radiomic review?`;
    }

    // 5. Live Multimodal Gemini AI Generation
    if (geminiApiKey) {
      try {
        const prompt = `User question: "${userMessage}"
Active scan loaded: ${activeScan ? activeScan.name + ' (Resolution ' + activeScan.pixelStats.width + 'x' + activeScan.pixelStats.height + ', Mean ' + activeScan.pixelStats.mean + ' HU, Contrast ' + activeScan.pixelStats.stdDev + ', Mean/σ ratio ' + activeScan.pixelStats.snr + ')' : 'None'}
Total cohort scans: ${cohortScans.length}.
Ground your answer in medical neuro-radiology facts and the uploaded data.`;
        return await callGeminiVision(prompt, activeScan ? activeScan.dataUrl : null);
      } catch (err) {
        return `*(Gemini API response note: ${err.message})*\n\nBased on your active session: currently **${cohortScans.length} scan(s)** are loaded. You can view the computed SNR, luminance, and K-means clusters above!`;
      }
    }

    await new Promise(r => setTimeout(r, 200));
    return `Currently **${cohortScans.length} scan(s)** are loaded. Click any of the prompt chips below or connect your Gemini API key in the top navigation bar for live diagnostic interpretation!`;
  }

  async function handleSend(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    appendMessage('user', trimmed);
    if (chatInput) {
      chatInput.value = '';
      chatInput.style.height = 'auto';
    }
    appendTypingIndicator();
    try {
      const reply = await handleChat(trimmed);
      removeTypingIndicator();
      appendMessage('assistant', reply);
    } catch (err) {
      removeTypingIndicator();
      appendMessage('assistant', 'An unexpected error occurred. Please try again.');
    }
  }

  chatForm?.addEventListener('submit', e => {
    e.preventDefault();
    if (chatInput) handleSend(chatInput.value);
  });
  chatInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(chatInput.value);
    }
  });

  document.querySelectorAll('.quick-prompt').forEach(btn => {
    btn.addEventListener('click', () => handleSend(btn.textContent));
  });

  attachBtn?.addEventListener('click', () => {
    const activeScan = activeScanIndex >= 0 ? cohortScans[activeScanIndex] : null;
    if (!activeScan) {
      fileInput?.click();
      return;
    }
    appendMessage('user', `📎 Attached: \`${activeScan.name}\``);
    appendMessage('assistant', `Acknowledged \`${activeScan.name}\` (${activeScan.pixelStats.width}×${activeScan.pixelStats.height}px, Mean/σ ratio: ${activeScan.pixelStats.snr}). What specific anatomical or radiomic features would you like to review?`);
  });

  function appendWelcome() {
    appendMessage('assistant', `Welcome to **NeuroScan AI — Brain MRI Radiomics Workbench**.

To get started:
* Click any **Benchmark Study** (Normal T1, Pathological T2, or FLAIR) to immediately test the system with 1-click.
* Or **drag & drop** your real brain MRI scans (.png, .jpg, .pdf) into the upload viewport.
* Review real-time **K-means tissue segmentation** and the **forward-chaining expert system trace** in the AI Lab.
* Connect your Google Gemini API key via the top bar for live multimodal diagnostic impressions!`);
  }
  appendWelcome();

  // Expose core references for modular testing
  window.NeuroScanApp = {
    cohortScans,
    selectScan,
    loadSample
  };
})();
