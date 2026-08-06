// background/background.js
// MV3 service worker
// Responsible for:
//   1. Getting the URL of the currently active tab.
//   2. Sending that URL to the SentinelWeb backend for a phishing prediction.
//   3. Returning the result back to whoever asked (popup.js).

const API_ENDPOINT = "http://127.0.0.1:8000/predict";

/**
 * Gets the URL of the currently active tab in the current window.
 * @returns {Promise<string>} the active tab's URL
 */
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

/**
 * Calls the SentinelWeb backend with the given URL and returns the
 * prediction result.
 * @param {string} url
 * @returns {Promise<{prediction: string, confidence: number}>}
 */
async function fetchPrediction(url) {
  const response = await fetch(API_ENDPOINT, {
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

  if (!data || typeof data.prediction !== "string" || typeof data.confidence !== "number") {
    throw new Error("Backend returned an unexpected response shape.");
  }

  return data;
}

/**
 * Full flow: get active tab URL -> ask backend -> return combined result.
 */
async function getPredictionForActiveTab() {
  const url = await getActiveTabUrl();

  // Guard against internal browser pages (chrome://, edge://, about:, etc.)
  // since the backend can't meaningfully classify these and requests to
  // them are not useful.
  if (!/^https?:\/\//i.test(url)) {
    return {
      ok: false,
      error: "This page can't be scanned (not an http/https URL).",
      url
    };
  }

  const result = await fetchPrediction(url);

  return {
    ok: true,
    url,
    prediction: result.prediction,
    confidence: result.confidence
  };
}

// Listen for messages from the popup (or content script) asking for a
// prediction on the currently active tab.
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

    // Return true to indicate we will respond asynchronously.
    return true;
  }
});