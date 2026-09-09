// SentinelWeb background service worker

const API_BASE = "http://20.219.53.10:8000";
const PREDICT_ENDPOINT = `${API_BASE}/predict`;
const EXPLAIN_ENDPOINT = `${API_BASE}/explain`;

async function getActiveTabUrl() {
  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  if (!tab || !tab.url) {
    throw new Error("Could not determine the active tab's URL.");
  }

  return tab.url;
}


// Safely send explanation to popup.
// Popup may already be closed when Gemini finishes.
function sendExplanationMessage(explanation) {
  chrome.runtime.sendMessage(
    {
      type: "EXPLANATION_READY",
      explanation
    },
    () => {
      // Ignore the harmless error when no popup is listening.
      void chrome.runtime.lastError;
    }
  );
}


async function fetchPrediction(url) {
  const response = await fetch(PREDICT_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ url })
  });

  if (!response.ok) {
    throw new Error(`Backend returned status ${response.status}`);
  }

  const data = await response.json();

  if (
    !data ||
    typeof data.prediction !== "string" ||
    typeof data.confidence !== "number"
  ) {
    throw new Error("Backend returned an unexpected prediction response.");
  }

  return data;
}


async function fetchExplanation(url) {
  try {
    const response = await fetch(EXPLAIN_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url })
    });

    if (!response.ok) {
      throw new Error(`Explanation endpoint returned ${response.status}`);
    }

    const data = await response.json();

    const explanation =
      typeof data.explanation === "string"
        ? data.explanation
        : "No explanation available.";

    sendExplanationMessage(explanation);

  } catch (error) {
    sendExplanationMessage(
      "AI explanation is temporarily unavailable. " +
      "The phishing prediction was generated successfully by the Random Forest model."
    );
  }
}


async function getPredictionForActiveTab() {
  const url = await getActiveTabUrl();

  if (!/^https?:\/\//i.test(url)) {
    return {
      ok: false,
      error: "This page can't be scanned (not an http/https URL).",
      url
    };
  }

  // Wait ONLY for Random Forest prediction.
  const result = await fetchPrediction(url);

  const predictionResult = {
    ok: true,
    url,
    prediction: result.prediction,
    confidence: result.confidence,
    explanation: ""
  };

  // Start Gemini separately.
  // DO NOT await this.
  fetchExplanation(url);

  // Return ML verdict immediately.
  return predictionResult;
}


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  if (message && message.type === "GET_PREDICTION") {

    getPredictionForActiveTab()
      .then((result) => sendResponse(result))
      .catch((err) => {
        sendResponse({
          ok: false,
          error: err.message || "Unknown error contacting the backend."
        });
      });

    return true;
  }
});