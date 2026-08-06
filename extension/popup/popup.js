// popup/popup.js
// Handles the popup UI: asks background.js for a prediction on the
// current tab and renders the result (or an error).

const urlValueEl = document.getElementById("urlValue");
const predictionValueEl = document.getElementById("predictionValue");
const confidenceValueEl = document.getElementById("confidenceValue");
const errorMessageEl = document.getElementById("errorMessage");
const refreshBtn = document.getElementById("refreshBtn");
const refreshIcon = document.getElementById("refreshIcon");

function setLoadingState() {
  urlValueEl.textContent = "Loading...";
  predictionValueEl.textContent = "—";
  predictionValueEl.className = "value badge pending";
  confidenceValueEl.textContent = "—";
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
  urlValueEl.textContent = result.url || "Unknown";

  if (!result.ok) {
    predictionValueEl.textContent = "—";
    predictionValueEl.className = "value badge pending";
    confidenceValueEl.textContent = "—";
    showError(result.error || "Something went wrong.");
    return;
  }

  hideError();

  const prediction = result.prediction; // "legitimate" | "phishing"
  const confidencePercent = (result.confidence * 100).toFixed(1) + "%";

  predictionValueEl.textContent = prediction;
  confidenceValueEl.textContent = confidencePercent;

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
      predictionValueEl.textContent = "—";
      predictionValueEl.className = "value badge pending";
      confidenceValueEl.textContent = "—";
      showError(chrome.runtime.lastError.message);
      return;
    }

    renderResult(response);
  });
}

refreshBtn.addEventListener("click", requestPrediction);

// Fetch a prediction as soon as the popup opens.
document.addEventListener("DOMContentLoaded", requestPrediction);