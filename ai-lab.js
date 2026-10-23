/* =========================================================================
   NeuroScan AI Lab — Syllabus-Driven AI Components (Zero External API)
   Unit 5 (Learning): K-means clustering (unsupervised) for tissue segmentation
   Units 3 & 6: Rule-based expert system with forward chaining + explanation trace
   ========================================================================= */

(function () {
  'use strict';

  const N = 128; // Standardized working matrix resolution for responsive real-time clustering
  // Clinical pseudo-color palette for segmented tissue classes (dark -> bright)
  // Background/CSF -> Deep Parenchyma/Gray Matter -> White Matter -> Hyperintensity/Focal Lesions
  const PALETTES = {
    clinical: ['#09090b', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
    grayscale: ['#0a0a0a', '#4a4a4a', '#8a8a8a', '#c8c8c8', '#ffffff'],
    thermal: ['#000004', '#51127c', '#b73779', '#fc8961', '#fcfdbf']
  };

  const TISSUE_NAMES = [
    'Air / Background / Bone Cortex',
    'Cerebrospinal Fluid (CSF) / Deep Sulci',
    'Cortical Gray Matter',
    'Subcortical White Matter',
    'Hyperintense Region / Lesion'
  ];

  let currentSegmentation = null;
  let activeClusterFilter = -1; // -1 = show all clusters

  // ---------- Unit 5: K-means (1-D intensity clustering) ----------
  function kmeans(values, k, maxIter = 60) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }

    if (hi === lo) hi = lo + 1;

    // Deterministic spread initialization across dynamic range
    let cent = Array.from({ length: k }, (_, i) => lo + ((i + 0.5) / k) * (hi - lo));
    const labels = new Uint8Array(values.length);
    let iter = 0;
    let moved = true;

    while (moved && iter < maxIter) {
      moved = false;
      iter++;
      const sum = new Float64Array(k);
      const cnt = new Uint32Array(k);

      // Assignment step
      for (let i = 0; i < values.length; i++) {
        const val = values[i];
        let best = 0;
        let bd = Infinity;
        for (let c = 0; c < k; c++) {
          const d = Math.abs(val - cent[c]);
          if (d < bd) {
            bd = d;
            best = c;
          }
        }
        if (labels[i] !== best) {
          labels[i] = best;
          moved = true;
        }
        sum[best] += val;
        cnt[best]++;
      }

      // Update step
      for (let c = 0; c < k; c++) {
        if (cnt[c] > 0) {
          cent[c] = sum[c] / cnt[c];
        }
      }
    }

    // Compute Sum of Squared Errors (SSE)
    let sse = 0;
    for (let i = 0; i < values.length; i++) {
      sse += (values[i] - cent[labels[i]]) ** 2;
    }

    // Sort clusters monotonically (darkest -> brightest) so classes maintain stable semantic meaning
    const order = cent.map((c, i) => [c, i]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
    const remap = new Uint8Array(k);
    order.forEach((oldIdx, newIdx) => (remap[oldIdx] = newIdx));

    for (let i = 0; i < labels.length; i++) {
      labels[i] = remap[labels[i]];
    }

    const sortedCentroids = order.map(i => cent[i]);
    const counts = new Array(k).fill(0);
    for (let i = 0; i < labels.length; i++) {
      counts[labels[i]]++;
    }

    return {
      labels,
      centroids: sortedCentroids,
      counts,
      iterations: iter,
      sse,
      k
    };
  }

  // ---------- Units 3 & 6: Rule-Based Expert System (Forward Chaining) ----------
  const RULES = [
    {
      id: 'R1',
      if: ['low_contrast'],
      then: 'poor_quality',
      why: 'Low contrast dispersion (σ < 25) obscures fine anatomical tissue boundaries'
    },
    {
      id: 'R2',
      if: ['under_exposed'],
      then: 'poor_quality',
      why: 'Severe hypo-illumination (mean luminance < 28) results in signal starvation'
    },
    {
      id: 'R3',
      if: ['poor_quality'],
      then: 'recommend_rescan',
      why: 'Uncertain signal-to-noise ratio compromises quantitative evaluation; re-acquisition recommended'
    },
    {
      id: 'R4',
      if: ['large_bright_cluster', 'good_quality'],
      then: 'focal_hyperintensity',
      why: 'Brightest intensity cluster occupies an unusually elevated fraction of brain parenchyma'
    },
    {
      id: 'R5',
      if: ['left_right_asymmetry', 'good_quality'],
      then: 'hemispheric_asymmetry',
      why: 'Bilateral intensity divergence across the mid-sagittal plane exceeds clinical tolerance (> 12%)'
    },
    {
      id: 'R6',
      if: ['focal_hyperintensity', 'hemispheric_asymmetry'],
      then: 'flag_for_radiologist',
      why: 'Correlated focal hyperintensity and hemispheric asymmetry indicate localized structural variance'
    },
    {
      id: 'R7',
      if: ['focal_hyperintensity'],
      then: 'suggest_flair_correlation',
      why: 'Hyperintense signal should be correlated with FLAIR / T2-weighted pulse sequences to rule out edema'
    }
  ];

  const FACT_DEFINITIONS = {
    low_contrast: { label: 'Low Contrast Dispersion', desc: 'Standard deviation σ < 25 HU', icon: 'activity' },
    under_exposed: { label: 'Severe Hypo-intensity', desc: 'Mean signal luminance < 28 HU', icon: 'moon' },
    good_quality: { label: 'Adequate Image Quality', desc: 'Sufficient contrast & luminance for segmentation', icon: 'check-circle' },
    large_bright_cluster: { label: 'Elevated High-Intensity Tissue Volume', desc: 'Brightest K-means cluster > 12% total volume', icon: 'zap' },
    left_right_asymmetry: { label: 'Bilateral Hemispheric Asymmetry', desc: 'Inter-hemispheric intensity divergence > 12%', icon: 'split' },
    poor_quality: { label: 'Sub-Optimal Acquisition', desc: 'Inadequate SNR or contrast', icon: 'alert-triangle' },
    recommend_rescan: { label: 'ACTION: Recommend Scan Re-acquisition', desc: 'Repeat slice acquisition with calibrated windowing', icon: 'repeat' },
    focal_hyperintensity: { label: 'FINDING: Focal Hyperintensity', desc: 'Localized high-signal tissue region detected', icon: 'crosshair' },
    hemispheric_asymmetry: { label: 'FINDING: Hemispheric Asymmetry', desc: 'Significant bilateral variance across falx cerebri', icon: 'compass' },
    flag_for_radiologist: { label: 'FLAG: Priority Review by Neuro-Radiologist', desc: 'Multiple independent radiomic criteria triggered', icon: 'shield-alert' },
    suggest_flair_correlation: { label: 'PROTOCOL: Correlate with FLAIR / T2 Sequence', desc: 'Differentiate CSF nulling from cellular edema', icon: 'layers' }
  };

  function forwardChain(initialFacts) {
    const facts = new Set(initialFacts);
    const trace = [];
    let changed = true;
    let iteration = 0;

    while (changed && iteration < 20) {
      changed = false;
      iteration++;
      for (const r of RULES) {
        if (!facts.has(r.then)) {
          const prerequisitesMet = r.if.every(f => facts.has(f));
          if (prerequisitesMet) {
            facts.add(r.then);
            trace.push({
              rule: r,
              cycle: iteration,
              inputs: [...r.if],
              output: r.then
            });
            changed = true;
          }
        }
      }
    }
    return { facts, trace };
  }

  // ---------- Image Feature Extraction ----------
  function extractGrayscaleMatrix(img) {
    const canvas = document.createElement('canvas');
    canvas.width = N;
    canvas.height = N;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, N, N);
    const imgData = ctx.getImageData(0, 0, N, N).data;
    const gray = new Float32Array(N * N);

    for (let i = 0; i < gray.length; i++) {
      // Standard Rec.601 perceptual luminance
      gray[i] = 0.299 * imgData[i * 4] + 0.587 * imgData[i * 4 + 1] + 0.114 * imgData[i * 4 + 2];
    }
    return gray;
  }

  function calculateBilateralAsymmetry(gray) {
    let diff = 0;
    let total = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N / 2; x++) {
        const left = gray[y * N + x];
        const right = gray[y * N + (N - 1 - x)];
        diff += Math.abs(left - right);
        total += left + right;
      }
    }
    return total > 0 ? (2 * diff) / total : 0;
  }

  function deriveInitialFacts(stats, km, asym) {
    const facts = [];
    const sd = parseFloat(stats.stdDev || 0);
    const mean = parseFloat(stats.mean || 0);
    const k = km.centroids.length;
    const brightClusterCount = km.counts[k - 1] || 0;
    const brightFrac = brightClusterCount / km.labels.length;

    if (sd < 25) facts.push('low_contrast');
    if (mean < 28) facts.push('under_exposed');
    if (sd >= 25 && mean >= 28) facts.push('good_quality');
    if (brightFrac > 0.12) facts.push('large_bright_cluster');
    if (asym > 0.12) facts.push('left_right_asymmetry');

    return { facts, brightFrac, sd, mean, asym };
  }

  // ---------- UI Rendering ----------
  const $ = id => document.getElementById(id);
  const esc = s => String(s || '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function renderSegmentationCanvas(km, palette, filterCluster = -1) {
    const cv = $('labCanvas');
    if (!cv) return;
    cv.width = N;
    cv.height = N;
    const ctx = cv.getContext('2d');
    const out = ctx.createImageData(N, N);

    for (let i = 0; i < km.labels.length; i++) {
      const cIdx = km.labels[i];
      if (filterCluster !== -1 && cIdx !== filterCluster) {
        // Mute other clusters
        out.data[i * 4] = 15;
        out.data[i * 4 + 1] = 18;
        out.data[i * 4 + 2] = 23;
        out.data[i * 4 + 3] = 255;
        continue;
      }

      const hex = palette[Math.min(cIdx, palette.length - 1)];
      out.data[i * 4] = parseInt(hex.slice(1, 3), 16);
      out.data[i * 4 + 1] = parseInt(hex.slice(3, 5), 16);
      out.data[i * 4 + 2] = parseInt(hex.slice(5, 7), 16);
      out.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(out, 0, 0);
  }

  function runLab(providedScan) {
    const empty = $('labEmpty');
    const body = $('labBody');
    if (!empty || !body) return;

    let scan = providedScan;
    if (!scan && typeof cohortScans !== 'undefined' && typeof activeScanIndex !== 'undefined') {
      scan = activeScanIndex >= 0 ? cohortScans[activeScanIndex] : null;
    }

    if (!scan || scan.isPdf || !scan.dataUrl) {
      empty.classList.remove('hidden');
      body.classList.add('hidden');
      empty.innerHTML = `
        <div class="flex flex-col items-center justify-center text-center py-10 text-slate-400">
          <div class="w-10 h-10 rounded-full bg-slate-800/80 flex items-center justify-center mb-2.5 text-slate-500">
            <i data-lucide="brain" class="w-5 h-5"></i>
          </div>
          <p class="font-medium text-slate-300 text-sm">${scan && scan.isPdf ? 'AI Lab requires an image scan (PNG/JPG), not a PDF document.' : 'Awaiting MRI Scan Upload'}</p>
          <p class="text-xs text-slate-500 mt-1 max-w-sm">Upload or select an MRI scan above to execute K-means tissue segmentation and the rule-based expert system.</p>
        </div>
      `;
      if (typeof safeCreateIcons === 'function') safeCreateIcons();
      return;
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const kSelect = $('labK');
      const k = parseInt(kSelect ? kSelect.value : 4, 10) || 4;
      const palette = PALETTES.clinical.slice(0, k);

      const gray = extractGrayscaleMatrix(img);
      const km = kmeans(gray, k);
      const asym = calculateBilateralAsymmetry(gray);
      const { facts, brightFrac } = deriveInitialFacts(scan.pixelStats, km, asym);
      const inference = forwardChain(facts);

      currentSegmentation = { km, palette, scan, asym, brightFrac, facts, inference };
      activeClusterFilter = -1;

      empty.classList.add('hidden');
      body.classList.remove('hidden');

      // Update original scan figure
      const labOrig = $('labOrig');
      if (labOrig) labOrig.src = scan.dataUrl;

      // Render segmentation canvas
      renderSegmentationCanvas(km, palette, activeClusterFilter);

      // Render cluster stats breakdown with interactive filter toggles
      const kmStats = $('labKmStats');
      if (kmStats) {
        kmStats.innerHTML = `
          <div class="space-y-1.5">
            ${km.centroids.map((centroid, i) => {
              const pct = ((100 * km.counts[i]) / km.labels.length).toFixed(1);
              const color = palette[Math.min(i, palette.length - 1)];
              const tissueName = TISSUE_NAMES[i] || `Tissue Class ${i + 1}`;
              return `
                <div class="cluster-row flex items-center justify-between p-1.5 rounded-lg border border-line/60 dark:border-slate-800 bg-white/50 dark:bg-slate-900/40 hover:bg-slate-100 dark:hover:bg-slate-800/60 cursor-pointer transition-colors text-xs"
                     data-cluster="${i}" title="Click to isolate this tissue cluster">
                  <div class="flex items-center gap-2 min-w-0">
                    <span class="w-3.5 h-3.5 rounded-sm shrink-0 border border-black/20" style="background:${color}"></span>
                    <div class="truncate">
                      <span class="font-medium text-slate-800 dark:text-slate-200">${tissueName}</span>
                      <span class="text-[10px] text-slate-400 font-mono block">Centroid: ${centroid.toFixed(1)} HU</span>
                    </div>
                  </div>
                  <div class="text-right shrink-0 font-mono">
                    <span class="font-bold text-slate-700 dark:text-slate-300">${pct}%</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
          <div class="mt-2.5 pt-2 border-t border-line/60 dark:border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-500">
            <span>Converged: ${km.iterations} iterations</span>
            <span>SSE: ${km.sse.toExponential(2)}</span>
          </div>
        `;

        // Attach isolation click listeners
        kmStats.querySelectorAll('.cluster-row').forEach(row => {
          row.addEventListener('click', () => {
            const cIdx = parseInt(row.getAttribute('data-cluster'), 10);
            activeClusterFilter = activeClusterFilter === cIdx ? -1 : cIdx;
            renderSegmentationCanvas(km, palette, activeClusterFilter);
            kmStats.querySelectorAll('.cluster-row').forEach(r => {
              const match = parseInt(r.getAttribute('data-cluster'), 10) === activeClusterFilter;
              r.classList.toggle('ring-2', match);
              r.classList.toggle('ring-teal-500', match);
            });
          });
        });
      }

      // Render Extracted Facts
      const factsEl = $('labFacts');
      if (factsEl) {
        factsEl.innerHTML = facts.length ? facts.map(f => {
          const def = FACT_DEFINITIONS[f] || { label: f, desc: '' };
          return `
            <li class="p-2 rounded-lg border border-line/60 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-900/40 text-xs">
              <div class="flex items-center gap-1.5 font-semibold text-slate-800 dark:text-slate-200">
                <span class="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                <span>${esc(def.label)}</span>
              </div>
              ${def.desc ? `<p class="text-[11px] text-slate-500 mt-0.5 ml-3">${esc(def.desc)}</p>` : ''}
            </li>
          `;
        }).join('') : '<li class="text-xs text-slate-500 italic p-2">No abnormal baseline facts detected.</li>';
      }

      // Render Inference Trace (Rules Fired)
      const traceEl = $('labTrace');
      if (traceEl) {
        traceEl.innerHTML = inference.trace.length ? inference.trace.map(t => {
          const rule = t.rule;
          const outDef = FACT_DEFINITIONS[t.output] || { label: t.output };
          return `
            <li class="p-2.5 rounded-lg border border-line/60 dark:border-slate-800 bg-white dark:bg-slate-900/60 text-xs">
              <div class="flex items-center justify-between mb-1">
                <span class="font-mono font-bold text-[11px] px-1.5 py-0.5 rounded bg-teal-50 dark:bg-teal-950/60 text-teal-700 dark:text-teal-400 border border-teal-200 dark:border-teal-800/60">
                  RULE ${rule.id}
                </span>
                <span class="text-[10px] text-slate-400 font-mono">Cycle ${t.cycle}</span>
              </div>
              <p class="font-mono text-[11px] text-slate-600 dark:text-slate-300 mb-1">
                <strong class="text-slate-700 dark:text-slate-200">IF</strong> ${rule.if.map(k => esc(FACT_DEFINITIONS[k]?.label || k)).join(' <strong class="text-slate-700 dark:text-slate-200">&amp;</strong> ')}
                <br>&rarr; <strong class="text-teal-600 dark:text-teal-400">THEN</strong> ${esc(outDef.label)}
              </p>
              <p class="text-[11px] text-slate-500 italic">"${esc(rule.why)}"</p>
            </li>
          `;
        }).join('') : '<li class="text-xs text-slate-500 italic p-2">No heuristic rules fired based on current facts.</li>';
      }

      // Render Conclusions & Action Directives
      const conclusionsEl = $('labConclusions');
      if (conclusionsEl) {
        const conclusions = [...inference.facts].filter(f => /^(recommend|flag|suggest|focal|hemispheric)/.test(f));
        conclusionsEl.innerHTML = conclusions.length ? conclusions.map(c => {
          const def = FACT_DEFINITIONS[c] || { label: c, desc: '' };
          const isUrgent = c.startsWith('flag') || c.startsWith('recommend');
          return `
            <div class="p-2.5 rounded-lg border text-xs leading-relaxed ${
              isUrgent 
                ? 'border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200' 
                : 'border-teal-500/40 bg-teal-500/10 text-teal-900 dark:text-teal-200'
            }">
              <div class="flex items-center gap-2 font-semibold">
                <span class="w-2 h-2 rounded-full ${isUrgent ? 'bg-amber-500' : 'bg-teal-500'}"></span>
                <span>${esc(def.label)}</span>
              </div>
              ${def.desc ? `<p class="mt-1 text-[11.5px] opacity-90 pl-4">${esc(def.desc)}</p>` : ''}
            </div>
          `;
        }).join('') : `
          <div class="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300 text-xs">
            <span class="font-semibold block mb-0.5">✅ No Immediate Action Directives</span>
            Extracted radiomic parameters lie within normal calibrated tolerances.
          </div>
        `;
      }

      // Metadata summary
      const metaEl = $('labMeta');
      if (metaEl) {
        metaEl.textContent = `Radiomic Vector: Contrast σ=${scan.pixelStats.stdDev} | Luminance Mean=${scan.pixelStats.mean} HU | Hemispheric Asymmetry=${(asym * 100).toFixed(1)}% | Bright Cluster Vol=${(brightFrac * 100).toFixed(1)}%`;
      }

      if (typeof safeCreateIcons === 'function') safeCreateIcons();
    };

    img.src = scan.dataUrl;
  }

  function resetLab() {
    currentSegmentation = null;
    activeClusterFilter = -1;
    const empty = $('labEmpty');
    const body = $('labBody');
    if (empty) {
      empty.classList.remove('hidden');
      empty.innerHTML = `
        <div class="flex flex-col items-center justify-center text-center py-10 text-slate-400">
          <div class="w-10 h-10 rounded-full bg-slate-800/80 flex items-center justify-center mb-2.5 text-slate-500">
            <i data-lucide="brain" class="w-5 h-5"></i>
          </div>
          <p class="font-medium text-slate-300 text-sm">Awaiting MRI Scan Upload</p>
          <p class="text-xs text-slate-500 mt-1 max-w-sm">Upload or select an MRI scan above to execute K-means tissue segmentation and the rule-based expert system.</p>
        </div>
      `;
    }
    if (body) body.classList.add('hidden');
    if (typeof safeCreateIcons === 'function') safeCreateIcons();
  }

  function init() {
    const labK = $('labK');
    const labRun = $('labRun');

    labK?.addEventListener('change', () => runLab());
    labRun?.addEventListener('click', () => runLab());

    // Listen to decoupled custom events dispatched by app.js
    window.addEventListener('neuroscan:scanSelected', e => {
      runLab(e.detail?.scan);
    });

    window.addEventListener('neuroscan:scansCleared', () => {
      resetLab();
    });

    // Also support global method invocation
    window.NeuroAILab = {
      runLab,
      resetLab,
      kmeans,
      forwardChain,
      RULES,
      getCurrentSegmentation: () => currentSegmentation
    };

    runLab();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
