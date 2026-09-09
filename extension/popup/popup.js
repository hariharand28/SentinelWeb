// SentinelWeb Popup
// ML verdict is displayed immediately.
// Gemini explanation is displayed asynchronously when ready.

const explanationValueEl = document.getElementById("explanationValue");
const urlValueEl = document.getElementById("urlValue");
const predictionValueEl = document.getElementById("predictionValue");
const confidenceValueEl = document.getElementById("confidenceValue");
const confidenceBarEl = document.getElementById("confidenceBar");
const errorMessageEl = document.getElementById("errorMessage");
const refreshBtn = document.getElementById("refreshBtn");
const refreshIcon = document.getElementById("refreshIcon");


// ------------------------------------------------------------
// Loading state
// ------------------------------------------------------------

function setLoadingState() {
  urlValueEl.textContent = "Loading...";
  urlValueEl.removeAttribute("title");

  predictionValueEl.textContent = "—";
  predictionValueEl.className = "value badge pending";

  confidenceValueEl.textContent = "—";

  // Reset confidence bar
  if (confidenceBarEl) {
    confidenceBarEl.style.width = "0%";
  }

  // Gemini is processed separately
  explanationValueEl.textContent = "Generating AI explanation...";

  hideError();

  refreshBtn.disabled = true;
  refreshIcon.classList.add("spinning");
}


// ------------------------------------------------------------
// Clear loading state
// ------------------------------------------------------------

function clearLoadingState() {
  refreshBtn.disabled = false;
  refreshIcon.classList.remove("spinning");
}


// ------------------------------------------------------------
// Error handling
// ------------------------------------------------------------

function showError(message) {
  errorMessageEl.textContent = message;
  errorMessageEl.hidden = false;
}


function hideError() {
  errorMessageEl.hidden = true;
  errorMessageEl.textContent = "";
}


// ------------------------------------------------------------
// Reset UI
// ------------------------------------------------------------

function resetResult() {
  predictionValueEl.textContent = "—";
  predictionValueEl.className = "value badge pending";

  confidenceValueEl.textContent = "—";

  if (confidenceBarEl) {
    confidenceBarEl.style.width = "0%";
  }

  explanationValueEl.textContent = "Generating AI explanation...";
}


// ------------------------------------------------------------
// Render ML prediction
// ------------------------------------------------------------

function renderPrediction(result) {
  const currentUrl = result.url || "Unknown";

  urlValueEl.textContent = currentUrl;
  urlValueEl.title = currentUrl;

  // Backend / extension error
  if (!result.ok) {
    resetResult();

    explanationValueEl.textContent = "No explanation available.";

    showError(
      result.error || "Something went wrong while scanning the page."
    );

    return;
  }

  hideError();

  const prediction = result.prediction;

  // Safely convert confidence to number
  const confidenceVal =
    typeof result.confidence === "number"
      ? result.confidence
      : Number(result.confidence);

  const safeConfidence = Number.isFinite(confidenceVal)
    ? Math.min(1, Math.max(0, confidenceVal))
    : 0;

  const confidencePercent =
    (safeConfidence * 100).toFixed(1) + "%";


  // ----------------------------------------------------------
  // SHOW ML RESULT IMMEDIATELY
  // ----------------------------------------------------------

  predictionValueEl.textContent = prediction;
  confidenceValueEl.textContent = confidencePercent;


  // ----------------------------------------------------------
  // Confidence progress bar
  // ----------------------------------------------------------

  if (confidenceBarEl) {
    const percentage = safeConfidence * 100;

    // Force the width directly.
    // This overrides the CSS default width: 0%.
    confidenceBarEl.style.width = `${percentage}%`;
  }


  // ----------------------------------------------------------
  // Verdict color
  // ----------------------------------------------------------

  predictionValueEl.className = "value badge";

  if (prediction === "legitimate") {
    predictionValueEl.classList.add("legitimate");
  } else if (prediction === "phishing") {
    predictionValueEl.classList.add("phishing");
  } else {
    predictionValueEl.classList.add("pending");
  }


  // ----------------------------------------------------------
  // Gemini is still running
  // ----------------------------------------------------------

  explanationValueEl.textContent =
    "Generating AI explanation...";
}


// ------------------------------------------------------------
// Request prediction from background service worker
// ------------------------------------------------------------

function requestPrediction() {
  setLoadingState();

  chrome.runtime.sendMessage(
    {
      type: "GET_PREDICTION"
    },
    (response) => {

      // Chrome runtime error
      if (chrome.runtime.lastError) {
        clearLoadingState();

        urlValueEl.textContent = "Unknown";
        urlValueEl.removeAttribute("title");

        resetResult();

        explanationValueEl.textContent =
          "No explanation available.";

        showError(
          chrome.runtime.lastError.message ||
          "Unable to communicate with SentinelWeb."
        );

        return;
      }


      // No response from background
      if (!response) {
        clearLoadingState();

        resetResult();

        explanationValueEl.textContent =
          "No explanation available.";

        showError(
          "No response received from the SentinelWeb backend."
        );

        return;
      }


      // ------------------------------------------------------
      // ML RESULT ARRIVES FIRST
      // ------------------------------------------------------

      renderPrediction(response);

      // IMPORTANT:
      // Stop spinner immediately after ML result.
      // Do NOT wait for Gemini.
      clearLoadingState();
    }
  );
}


// ------------------------------------------------------------
// Gemini explanation arrives later
// ------------------------------------------------------------

chrome.runtime.onMessage.addListener((message) => {

  if (!message) {
    return;
  }

  if (message.type === "EXPLANATION_READY") {

    explanationValueEl.textContent =
      message.explanation ||
      "No explanation available.";
  }


  // Optional Gemini failure message
  if (message.type === "EXPLANATION_ERROR") {

    explanationValueEl.textContent =
      "AI explanation is temporarily unavailable.";
  }
});


// ------------------------------------------------------------
// Rescan button
// ------------------------------------------------------------

refreshBtn.addEventListener(
  "click",
  requestPrediction
);


// ------------------------------------------------------------
// Popup initialization
// ------------------------------------------------------------

if (document.readyState === "loading") {

  document.addEventListener(
    "DOMContentLoaded",
    requestPrediction
  );

} else {

  requestPrediction();
}