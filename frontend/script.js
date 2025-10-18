//3yit
//api base first thing here
const API_BASE = (localStorage.getItem('API_BASE'))
  || `${location.protocol}//${location.hostname}:8000`;

//all the stuff i need to take first
const btnLong = document.getElementById("btn-long");
const btnShort = document.getElementById("btn-short");
const header = document.getElementById("header");
const mainSelection = document.getElementById("main-selection");
const formSection = document.getElementById("form-section");
const formTitle = document.getElementById("form-title");
const modelSelect = document.getElementById("model-select");
const keywordsGroup = document.getElementById("keywords-group");
const form = document.getElementById("video-form");
const resultSection = document.getElementById("result-section");
const resultMain = document.getElementById("result-main-message");
const resultViews = document.getElementById("predicted-views");
const feedbackContainer = document.getElementById("feedback-cards-container");
const showPlotsBtn = document.getElementById("show-plots-btn");
const plotsSection = document.getElementById("plots-section");

let predictionReady = false;
let lastPredictionMetrics = null;

//quantiles
const LONG_QUANTILES = { q25: 9.00, q50: 11.00, q75: 12.92, q95: 15.13 };
//now this for short should be different with actual quantiles saraha but im lazy ill do it later maybe 
const SHORT_QUANTILES = { q25: 9.40, q50: 10.60, q75: 11.60, q95: 12.60 };


const LONG_SCORE_MEDIAN = 11.00;
//3awed hna f short median idk maybe ill change it later to actual value
const SHORT_SCORE_MEDIAN = 10.60;

function quantilesFor(type) {
  return type === "long" ? LONG_QUANTILES : SHORT_QUANTILES;
}
function medianFor(type) {
  return type === "long" ? LONG_SCORE_MEDIAN : SHORT_SCORE_MEDIAN;
}
function labelFromScore(score, videoType) {
  const Q = quantilesFor(videoType);
  if (score >= Q.q95) return { key: "q95plus", title: "Potentiel viral très élevé" };
  if (score >= Q.q75) return { key: "q75to95", title: "Très bon potentiel" };
  if (score >= Q.q50) return { key: "q50to75", title: "Bon potentiel" };
  if (score >= Q.q25) return { key: "q25to50", title: "Potentiel modéré" };
  return { key: "belowq25", title: "Faible potentiel" };
}
function formatViews(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M vues estimées`;
  if (n >= 100_000) return `${(n / 1_000).toFixed(0)}K vues estimées`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}K vues estimées`;
  return `${n} vues estimées`;
}

//back button
const backBtn = document.createElement("button");
backBtn.textContent = "Retour au menu principal";
backBtn.id = "back-to-main";
backBtn.classList.add("back-btn");

let currentModel = null;

//find model to activate
function activateModel(modelType) {
  currentModel = modelType;
  const label = modelType === "long" ? "Format Long" : "Format Court";
  formTitle.textContent = modelType === "long"
    ? "Remplissez ce formulaire et déposez votre vidéo longue. L’IA prédira ses performances et sa portée."
    : "Remplissez ce formulaire et déposez votre vidéo courte. L’IA évaluera son potentiel de viralité et d’engagement.";
  keywordsGroup.style.display = modelType === "long" ? "block" : "none";
  modelSelect.value = modelType;

  mainSelection.classList.add("hidden");
  header.classList.remove("hidden");
  formSection.classList.remove("hidden");
}

//button to return to main menu
function returnToMain() {
  currentModel = null;
  predictionReady = false;

  form.reset();
  formTitle.textContent = "Sélectionnez un format pour commencer";

  resultSection.classList.add("hidden");
  plotsSection.classList.add("hidden");
  formSection.classList.add("hidden");
  header.classList.add("hidden");
  mainSelection.classList.remove("hidden");

  //res result block
  resultMain.innerHTML = `
    <div id="main-text"></div>
    <div id="predicted-views"></div>
  `;
  feedbackContainer.innerHTML = "";
  const loader = document.getElementById("loading-animation");
  if (loader) loader.remove();
}

//now to show the predictions
function showPredictionResult(data) {
  header.classList.add("hidden");
  formSection.classList.add("hidden");

  lastPredictionMetrics = {
    predicted_score: data.predicted_score ?? null,
    duration_min: data.duration_seconds / 60,
    desc_len: document.getElementById("description").value.trim().length,
    title_len: document.getElementById("title").value.trim().length,
    title_desc_len_ratio:
      document.getElementById("title").value.trim().length /
      Math.max(1, document.getElementById("description").value.trim().length),
    kw_count:
      currentModel === "long"
        ? document
            .getElementById("keywords")
            .value.split(",")
            .map((s) => s.trim())
            .filter(Boolean).length
        : 0
  };

  setTimeout(() => {
    //remove loader
    const loader = document.getElementById("loading-animation");
    if (loader) {
      loader.style.opacity = 0;
      setTimeout(() => loader.remove(), 400);
    }

    // --- Calibrated headline + views (with low-views warning) ---
    const score = data.predicted_score;
    const vt = data.video_type;
    const lbl = labelFromScore(score, vt);

    //gotta amek sure the chart uses the model score later
    lastPredictionMetrics = { ...lastPredictionMetrics, predicted_score: score };

    const headline = `Votre vidéo a ${lbl.title.toLowerCase()} !`;

    let viewsText;
    if (data.predicted_views <= 20000) {
      viewsText = "⚠️ Attention : moins de 10K vues possibles";
    } else {
      viewsText = `~${formatViews(data.predicted_views)}~`;
    }

    resultMain.innerHTML = `
      <div id="main-text" class="fade-in-block">${headline}</div>
      <div id="predicted-views" class="fade-in-block">${viewsText}</div>
    `;

    //animations
    const fadeIns = resultMain.querySelectorAll(".fade-in-block");
    fadeIns.forEach((el) => {
      el.classList.remove("fade-in-block");
      void el.offsetWidth;
      el.classList.add("fade-in-block");
    });

    //feedback
    feedbackContainer.innerHTML = "";

    const combinedFeedback = [
      ...data.feedback.numeric.slice(0, 2), // show top 2 numeric
      ...data.feedback.text.slice(0, 3),    // and top 3 text
    ];

    combinedFeedback.forEach((item, i) => {
      const card = document.createElement("div");
      card.classList.add("feedback-card");
      card.style.animationDelay = `${0.6 + i * 0.5}s`;

      const title = document.createElement("h3");
      title.classList.add("feedback-title");
      title.textContent = item.feature || item.field || "Analyse";

      const body = document.createElement("p");
      body.textContent = item.message;

      card.appendChild(title);
      card.appendChild(body);
      feedbackContainer.appendChild(card);
    });

    //add the back button
    if (!resultSection.contains(backBtn)) {
      resultSection.appendChild(backBtn);
    }

    resultSection.classList.remove("hidden");
    predictionReady = true;
  }, 1200);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const title = document.getElementById("title").value.trim();
  const description = document.getElementById("description").value.trim();
  const keywords = document.getElementById("keywords").value.trim();
  const videoFile = document.getElementById("video").files[0];

  if (!title || !description || !videoFile || (currentModel === "long" && !keywords)) {
    alert("⚠️ Veuillez remplir tous les champs requis.");
    return;
  }

  //hide other stuff
  header.classList.add("hidden");
  formSection.classList.add("hidden");

  //reset and show loader
  resultMain.innerHTML = "";
  feedbackContainer.innerHTML = "";
  resultMain.style.opacity = "1";
  resultMain.style.transform = "translateY(0)";
  resultMain.style.animation = "none";
  void resultMain.offsetWidth;

  resultMain.innerHTML = `
    <div id="loading-animation" class="loading-spinner">
      <span class="dot dot1"></span>
      <span class="dot dot2"></span>
      <span class="dot dot3"></span>
      <p>Le modèle IA analyse votre vidéo...</p>
    </div>
  `;
  resultSection.classList.remove("hidden");

  //prep the FormData 3yit
  const formData = new FormData();
  formData.append("video_type", currentModel);
  formData.append("title", title);
  formData.append("description", description);
  formData.append("file", videoFile);
  if (currentModel === "long") {
    formData.append("keywords", keywords);
  }

  try {
    //send POST request to FastAPI
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Erreur serveur");

    const data = await response.json();

    console.log("Backend response:", data);

    //show results
    showPredictionResult(data);

  } catch (error) {
    console.error("Prediction failed:", error);
    alert("Erreur lors de la prédiction. Veuillez réessayer.");
  }
});

btnLong.addEventListener("click", () => activateModel("long"));
btnShort.addEventListener("click", () => activateModel("short"));

modelSelect.addEventListener("change", (e) => {
  const selected = e.target.value;
  if (selected !== currentModel) {
    activateModel(selected);
  }
});

backBtn.addEventListener("click", returnToMain);

let chartInstance = null;

showPlotsBtn.addEventListener("click", () => {
  if (!predictionReady) {
    alert("Veuillez effectuer une prédiction avant d'accéder à l’analyse avancée.");
    return;
  }

  resultSection.classList.add("hidden");
  plotsSection.classList.remove("hidden");

  if (!plotsSection.contains(backBtn)) {
    plotsSection.appendChild(backBtn);
  }

  renderChart();
  //update model info sentence
  const modelInfo = document.getElementById("model-info");
  if (modelInfo) {
    modelInfo.textContent =
      currentModel === "long"
        ? "Ce modèle IA a été entraîné sur près de 100 000 vidéos longues!"
        : "Ce modèle IA a été entraîné sur plus de 55 000 vidéos courtes!";
  }
});

async function renderChart() {
  const ctx = document.getElementById("plot-canvas").getContext("2d");
  if (chartInstance) chartInstance.destroy();

  try {
    const jsonPath =
      currentModel === "long"
        ? "./feature_impact_long.json"
        : "./feature_impact_short.json";

    const res = await fetch(jsonPath);
    if (!res.ok) throw new Error("Impossible de charger les données");
    const averages = await res.json();

    const labels = [];
    const avgVals = [];  
    const userVals = []; 
    const rawForTooltip = []; //keep raw numbers for tooltip display

    const featureDisplay = {
      duration_min: "Durée (min)",
      desc_len: "Description (car.)",
      title_desc_len_ratio: "Ratio Titre/Desc",
      Score: "Score"
    };

    //Map JSON feature name -> user's raw value
    const userValFor = (feature) => {
      if (!lastPredictionMetrics) return 0;
      if (feature === "duration_min") return lastPredictionMetrics.duration_min;                   // minutes
      if (feature === "desc_len") return lastPredictionMetrics.desc_len;                          // chars
      if (feature === "title_desc_len_ratio") return lastPredictionMetrics.title_desc_len_ratio;  // ratio
      if (feature === "duration") return lastPredictionMetrics.duration_min * 60;                 // seconds
      if (feature === "title_len") return lastPredictionMetrics.title_len;                        // chars
      if (feature === "kw_count") return lastPredictionMetrics.kw_count;                          // #
      return lastPredictionMetrics[feature] ?? 0;
    };

    //choose which features to plot
    let keysToKeep;
    if (currentModel === "long") {
      keysToKeep = ["duration_min", "desc_len"];
    } else {
      keysToKeep = ["title_desc_len_ratio", "desc_len"];
    }

    //do the normalized bars
    for (const key of keysToKeep) {
      if (!averages[key]) continue;

      const avgRaw = averages[key].avg_top10 ?? 0;
      const userRaw = userValFor(key) ?? 0;
      const scale = Math.max(avgRaw, userRaw, 1e-9); //avoid divide by 0

      labels.push(featureDisplay[key] || key);
      avgVals.push((avgRaw / scale) * 100);
      userVals.push((userRaw / scale) * 100);
      rawForTooltip.push({ key, avgRaw, userRaw });
    }

    //hmm idk maybe ill do score (normalized, median per model) not sure
    {
      const avgRaw = medianFor(currentModel);
      const userRaw = lastPredictionMetrics?.predicted_score ?? 0;
      const scale = Math.max(avgRaw, userRaw, 1e-9);

      labels.push(featureDisplay.Score);
      avgVals.push((avgRaw / scale) * 100);
      userVals.push((userRaw / scale) * 100);
      rawForTooltip.push({ key: "Score", avgRaw, userRaw });
    }

    chartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Votre vidéo",
            data: userVals,
            backgroundColor: "rgba(102, 153, 255, 0.8)",
          },
          {
            label: "Moyenne des meilleures vidéos",
            data: avgVals,
            backgroundColor: "rgba(204, 204, 255, 0.5)",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: "#fff" } },
          title: {
            display: true,
            text: "Comparaison des caractéristiques (normalisées)",
            color: "#fff",
            font: { size: 18 },
          },
          tooltip: {
            callbacks: {
              //show raw values in tooltip
              label: function(ctx) {
                const i = ctx.dataIndex;
                const raw = rawForTooltip[i];
                const isUser = ctx.dataset.label === "Votre vidéo";
                const rawVal = isUser ? raw.userRaw : raw.avgRaw;

                //pretty formatting per metric
                const key = raw.key;
                let formatted;
                if (key === "duration_min") {
                  formatted = `${rawVal.toFixed(2)} min`;
                } else if (key === "desc_len") {
                  formatted = `${Math.round(rawVal)} car.`;
                } else if (key === "title_desc_len_ratio") {
                  formatted = rawVal.toFixed(2);
                } else if (key === "Score") {
                  formatted = rawVal.toFixed(2);
                } else {
                  formatted = `${rawVal}`;
                }

                const pct = ctx.parsed.y != null ? `${Math.round(ctx.parsed.y)}%` : "";
                return `${ctx.dataset.label}: ${pct} (valeur: ${formatted})`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: 100,
            ticks: {
              color: "#fff",
              callback: (v) => `${v}%`
            },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
          x: {
            ticks: { color: "#fff" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
    });
  } catch (err) {
    console.error(err);
    alert("Erreur lors du chargement de l'analyse graphique.");
  }
}