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
  window.NeuroScanAILab = {
    kmeans,
    PALETTES,
    TISSUE_NAMES
  };
  console.log('AI Lab Unit 5 K-means clustering module initialized.');
})();
