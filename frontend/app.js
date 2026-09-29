/**
 * News Bias Detector — Frontend Application Logic
 * Minimalist, Vanilla JS interface connecting to FastAPI backend
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const articleText = document.getElementById("articleText");
  const charCount = document.getElementById("charCount");
  const wordCount = document.getElementById("wordCount");
  const clearTextBtn = document.getElementById("clearTextBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const sampleSelect = document.getElementById("sampleSelect");
  const sampleChips = document.getElementById("sampleChips");

  const fileDropzone = document.getElementById("fileDropzone");
  const fileInput = document.getElementById("fileInput");
  const browseTrigger = document.getElementById("browseTrigger");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultsContent = document.getElementById("resultsContent");

  const predictedClass = document.getElementById("predictedClass");
  const confidenceValue = document.getElementById("confidenceValue");
  const distributionList = document.getElementById("distributionList");

  const tokenCount = document.getElementById("tokenCount");
  const inVocabCount = document.getElementById("inVocabCount");
  const oovCount = document.getElementById("oovCount");
  const coverageRate = document.getElementById("coverageRate");
  const oovDetails = document.getElementById("oovDetails");
  const oovTags = document.getElementById("oovTags");

  const copyJsonBtn = document.getElementById("copyJsonBtn");
  const copySummaryBtn = document.getElementById("copySummaryBtn");

  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIconSun = document.getElementById("themeIconSun");
  const themeIconMoon = document.getElementById("themeIconMoon");

  const openMetricsBtn = document.getElementById("openMetricsBtn");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const metricsModal = document.getElementById("metricsModal");
  const toast = document.getElementById("toast");

  // State
  let samplesList = [];
  let currentResult = null;

  // Initialize theme
  initTheme();

  // Load sample articles from API
  fetchSamples();

  // Load model metrics
  fetchMetrics();

  // Event Listeners
  articleText.addEventListener("input", updateCounts);
  clearTextBtn.addEventListener("click", handleClear);
  analyzeBtn.addEventListener("click", () => handleAnalyze());

  articleText.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleAnalyze();
    }
  });

  sampleSelect.addEventListener("change", (e) => {
    const selectedId = e.target.value;
    if (!selectedId) return;
    const sample = samplesList.find((s) => s.id === selectedId);
    if (sample) {
      loadSample(sample);
    }
  });

  // File Upload Handlers
  browseTrigger.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelect);

  fileDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileDropzone.classList.add("dragover");
  });

  fileDropzone.addEventListener("dragleave", () => {
    fileDropzone.classList.remove("dragover");
  });

  fileDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    fileDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  });

  // Modal handlers
  openMetricsBtn.addEventListener("click", () => {
    metricsModal.classList.remove("hidden");
  });

  closeModalBtn.addEventListener("click", () => {
    metricsModal.classList.add("hidden");
  });

  metricsModal.addEventListener("click", (e) => {
    if (e.target === metricsModal) {
      metricsModal.classList.add("hidden");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !metricsModal.classList.contains("hidden")) {
      metricsModal.classList.add("hidden");
    }
  });

  // Copy handlers
  copyJsonBtn.addEventListener("click", () => {
    if (!currentResult) return;
    navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2))
      .then(() => showToast("Analysis JSON copied to clipboard"))
      .catch(() => showToast("Failed to copy to clipboard"));
  });

  copySummaryBtn.addEventListener("click", () => {
    if (!currentResult) return;
    const summary = `News Bias Analysis:
- Predicted Stance: ${currentResult.predicted_class.toUpperCase()}
- Confidence: ${(currentResult.confidence * 100).toFixed(1)}%
- Probabilities:
  * Conservative: ${(currentResult.probabilities.conservative * 100).toFixed(1)}%
  * Liberal: ${(currentResult.probabilities.liberal * 100).toFixed(1)}%
  * Neutral: ${(currentResult.probabilities.neutral * 100).toFixed(1)}%
- Tokens Analyzed: ${currentResult.token_count} (GloVe coverage: ${currentResult.coverage_rate}%)`;
    navigator.clipboard.writeText(summary)
      .then(() => showToast("Summary copied to clipboard"))
      .catch(() => showToast("Failed to copy to clipboard"));
  });

  // Functions
  function updateCounts() {
    const text = articleText.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCount.textContent = `${chars.toLocaleString()} character${chars === 1 ? "" : "s"}`;
    wordCount.textContent = `${words.toLocaleString()} word${words === 1 ? "" : "s"}`;
  }

  function handleClear() {
    articleText.value = "";
    sampleSelect.value = "";
    document.querySelectorAll(".chip-btn").forEach((btn) => btn.classList.remove("active"));
    updateCounts();
    emptyState.classList.remove("hidden");
    resultsContent.classList.add("hidden");
    loadingState.classList.add("hidden");
    currentResult = null;
    articleText.focus();
  }

  async function fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      samplesList = await res.json();

      // Populate Select Dropdown
      samplesList.forEach((sample) => {
        const opt = document.createElement("option");
        opt.value = sample.id;
        opt.textContent = `${sample.title} (${capitalize(sample.expected_stance)})`;
        sampleSelect.appendChild(opt);
      });

      // Render preset chips (first 4)
      sampleChips.innerHTML = '<span class="chips-label">Presets:</span>';
      samplesList.slice(0, 4).forEach((sample) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "chip-btn";
        btn.textContent = `${sample.title.split(" ")[0]}… (${capitalize(sample.expected_stance)})`;
        btn.title = sample.title;
        btn.addEventListener("click", () => {
          document.querySelectorAll(".chip-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          sampleSelect.value = sample.id;
          loadSample(sample);
        });
        sampleChips.appendChild(btn);
      });
    } catch (err) {
      console.warn("Could not load sample articles:", err);
    }
  }

  function loadSample(sample) {
    articleText.value = sample.text;
    updateCounts();
    handleAnalyze();
  }

  async function handleAnalyze() {
    const text = articleText.value.trim();
    if (!text) {
      showToast("Please enter or paste an article excerpt to analyze.");
      articleText.focus();
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      currentResult = data;
      renderResults(data);
    } catch (err) {
      console.error("Prediction error:", err);
      showToast(err.message || "Analysis failed. Please check backend connection.");
      setLoading(false);
      emptyState.classList.remove("hidden");
      resultsContent.classList.add("hidden");
    }
  }

  function handleFileSelect(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }

  function processFile(file) {
    if (!file.name.match(/\.(txt|text|md|csv)$/i)) {
      showToast("Please provide a text (.txt or .md) file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      articleText.value = content;
      updateCounts();
      showToast(`Loaded file: ${file.name}`);
      handleAnalyze();
    };
    reader.readAsText(file);
  }

  function renderResults(data) {
    setLoading(false);
    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    // Primary Stance Banner
    predictedClass.textContent = data.predicted_class;
    confidenceValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // Probability Bars
    distributionList.innerHTML = "";
    const sortedProbabilities = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);

    sortedProbabilities.forEach(([cls, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const isWinner = cls.toLowerCase() === data.predicted_class.toLowerCase();

      const row = document.createElement("div");
      row.className = `dist-row ${isWinner ? "is-winner" : ""}`;
      row.innerHTML = `
        <div class="dist-meta">
          <span class="dist-label">
            <span class="dist-dot"></span>
            ${capitalize(cls)}
          </span>
          <span class="dist-val">${pct}%</span>
        </div>
        <div class="dist-track">
          <div class="dist-fill" style="width: 0%"></div>
        </div>
      `;
      distributionList.appendChild(row);

      // Trigger width animation on next frame
      requestAnimationFrame(() => {
        row.querySelector(".dist-fill").style.width = `${pct}%`;
      });
    });

    // Coverage Metrics
    tokenCount.textContent = data.token_count.toLocaleString();
    inVocabCount.textContent = data.in_vocab_count.toLocaleString();
    oovCount.textContent = data.oov_count.toLocaleString();
    coverageRate.textContent = `${data.coverage_rate}%`;

    // Out-of-vocabulary details
    if (data.oov_tokens && data.oov_tokens.length > 0) {
      oovDetails.classList.remove("hidden");
      oovTags.innerHTML = "";
      data.oov_tokens.forEach((tok) => {
        const tag = document.createElement("span");
        tag.className = "oov-tag";
        tag.textContent = tok;
        oovTags.appendChild(tag);
      });
    } else {
      oovDetails.classList.add("hidden");
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      emptyState.classList.add("hidden");
      resultsContent.classList.add("hidden");
      loadingState.classList.remove("hidden");
      analyzeBtn.disabled = true;
    } else {
      loadingState.classList.add("hidden");
      analyzeBtn.disabled = false;
    }
  }

  async function fetchMetrics() {
    try {
      const res = await fetch("/api/metrics");
      if (!res.ok) return;
      const data = await res.json();
      const meta = data.metadata;

      if (meta && meta.test_metrics) {
        const acc = meta.test_metrics.accuracy;
        const f1 = meta.test_metrics.macro_f1;
        const prec = meta.test_metrics.macro_precision;
        const rec = meta.test_metrics.macro_recall;

        document.getElementById("metricAccuracy").textContent = acc ? `${(acc * 100).toFixed(1)}%` : "—";
        document.getElementById("metricF1").textContent = f1 ? (f1 * 100).toFixed(1) : "—";
        document.getElementById("metricPrecision").textContent = prec ? (prec * 100).toFixed(1) : "—";
        document.getElementById("metricRecall").textContent = rec ? (rec * 100).toFixed(1) : "—";
      }
    } catch (err) {
      console.warn("Could not fetch model metrics:", err);
    }
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.remove("hidden");
    toast.style.opacity = "1";
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.classList.add("hidden"), 200);
    }, 2800);
  }

  function initTheme() {
    const savedTheme = localStorage.getItem("veritas_theme") || "light";
    setTheme(savedTheme);

    themeToggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      setTheme(next);
    });
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("veritas_theme", theme);
    if (theme === "dark") {
      themeIconSun.classList.add("hidden");
      themeIconMoon.classList.remove("hidden");
    } else {
      themeIconSun.classList.remove("hidden");
      themeIconMoon.classList.add("hidden");
    }
  }

  function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
});
