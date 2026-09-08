// popup/popup.js
// Handles the popup UI: asks background.js for a prediction on the
// current tab and renders the result (or an error).

const explanationValueEl = document.getElementById("explanationValue");
const urlValueEl = document.getElementById("urlValue");
const predictionValueEl = document.getElementById("predictionValue");
const confidenceValueEl = document.getElementById("confidenceValue");
const confidenceBarEl = document.getElementById("confidenceBar");
const errorMessageEl = document.getElementById("errorMessage");
const refreshBtn = document.getElementById("refreshBtn");
const refreshIcon = document.getElementById("refreshIcon");

function setLoadingState() {
  urlValueEl.textContent = "Loading...";
  urlValueEl.removeAttribute("title");
  predictionValueEl.textContent = "—";
  predictionValueEl.className = "value badge pending";
  confidenceValueEl.textContent = "—";
  if (confidenceBarEl) {
    confidenceBarEl.style.width = "0%";
  }
  explanationValueEl.textContent = "Loading explanation...";
  hideError();
  refreshBtn.disabled = true;
  refreshIcon.classList.add("spinning");
}

function clearLoadingState() {
  refreshBtn.disabled = false;
  refreshIcon.classList.remove("spinning");
}

function showError(message) {
  errorMessageEl.textContent = message;
  errorMessageEl.hidden = false;
}

function hideError() {
  errorMessageEl.hidden = true;
  errorMessageEl.textContent = "";
}

function renderResult(result) {
  const currentUrl = result.url || "Unknown";
  urlValueEl.textContent = currentUrl;
  urlValueEl.title = currentUrl;

  if (!result.ok) {
    predictionValueEl.textContent = "—";
    predictionValueEl.className = "value badge pending";
    confidenceValueEl.textContent = "—";
    if (confidenceBarEl) {
      confidenceBarEl.style.width = "0%";
    }
    explanationValueEl.textContent = "No explanation available.";
    showError(result.error || "Something went wrong.");
    return;
  }

  hideError();

  const prediction = result.prediction; // "legitimate" | "phishing"
  const confidenceVal = typeof result.confidence === "number" ? result.confidence : 0;
  const confidencePercent = (confidenceVal * 100).toFixed(1) + "%";

  predictionValueEl.textContent = prediction;
  confidenceValueEl.textContent = confidencePercent;

  if (confidenceBarEl) {
    const barWidth = Math.min(100, Math.max(0, confidenceVal * 100));
    confidenceBarEl.style.width = `${barWidth}%`;
  }

  explanationValueEl.textContent =
    result.explanation || "No explanation available.";

  predictionValueEl.className = "value badge";
  if (prediction === "legitimate") {
    predictionValueEl.classList.add("legitimate");
  } else if (prediction === "phishing") {
    predictionValueEl.classList.add("phishing");
  } else {
    predictionValueEl.classList.add("pending");
  }
}

function requestPrediction() {
  setLoadingState();

  chrome.runtime.sendMessage({ type: "GET_PREDICTION" }, (response) => {
    clearLoadingState();

    // chrome.runtime.lastError happens if the background worker
    // couldn't be reached at all.
    if (chrome.runtime.lastError) {
      urlValueEl.textContent = "Unknown";
      urlValueEl.removeAttribute("title");
      predictionValueEl.textContent = "—";
      predictionValueEl.className = "value badge pending";
      confidenceValueEl.textContent = "—";
      if (confidenceBarEl) {
        confidenceBarEl.style.width = "0%";
      }
      explanationValueEl.textContent = "No explanation available.";
      showError(chrome.runtime.lastError.message);
      return;
    }

    renderResult(response);
  });
}

refreshBtn.addEventListener("click", requestPrediction);

// Fetch a prediction as soon as the popup opens, handling both
// document-ready states reliably.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", requestPrediction);
} else {
  requestPrediction();
}