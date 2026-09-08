# SentinelWeb

> Intelligent Phishing URL Detection System powered by Machine Learning & Generative AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![Chrome Extension](https://img.shields.io/badge/Chrome%20Extension-Manifest%20V3-yellow.svg)](https://developer.chrome.com/docs/extensions/mv3/intro/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-Explanation%20Layer-8E75B2.svg)](https://ai.google.dev/)

---

## 1. Project Identity & Purpose

**SentinelWeb** is an engineering cybersecurity solution designed to protect users in real-time from phishing attacks while browsing the web. Phishing websites impersonate legitimate institutions (banks, social media, payment gateways) using deceptive URLs, homograph attacks, typo-squatting, and social engineering tricks.

SentinelWeb operates directly in the browser via a lightweight Chrome Extension (Manifest V3). When a user navigates to any website:
1. The extension inspects the active tab's URL.
2. The URL is processed through a high-performance **FastAPI** backend.
3. Over **70 lexical, domain-based, and security features** are extracted from the URL.
4. An optimized **Random Forest Classifier** evaluates the feature vector and produces the final classification:
   - **`legitimate`**
   - **`phishing`**
   along with a calibrated **confidence score** (0.0% - 100.0%).
5. **Google Gemini** acts strictly as an **Explanation Layer**, translating technical security indicators (e.g., entropy, subdomain depth, suspicious tokens, IP literals) into clear, actionable advice for the end user.

> **CRITICAL ARCHITECTURAL PRINCIPLE:**
> **Random Forest = FINAL CLASSIFIER**  
> **Google Gemini = EXPLANATION ONLY**  
> Gemini never decides, overrides, or alters the machine-learning verdict. If the Gemini API is unreachable, times out, or encounters errors, the security verdict is immediately delivered with a safe fallback notice.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                         Chrome Browser Extension                        |
|                                                                         |
|   +-----------------------+              +--------------------------+   |
|   | popup.html / popup.js | <----------  | background.js (Worker)   |   |
|   | (Dark Security UI)    |              | - Active tab querying    |   |
|   +-----------------------+              | - URL scheme validation  |   |
+------------------------------------------+------------------------------+
                                                 |
                                         HTTP POST /predict
                                                 |
                                                 v
+-------------------------------------------------------------------------+
|                             FastAPI Backend                             |
|                                                                         |
|   [CORS Middleware] -> [Input Validation (Pydantic)]                    |
|                                                                         |
|   1. Feature Extraction (backend/app/ml/)                               |
|      - LexicalFeatureExtractor   (length, digits, special characters)   |
|      - DomainFeatureExtractor    (subdomain depth, TLD, entropy)        |
|      - SecurityFeatureExtractor  (IP literals, suspicious keywords)    |
|      - Unified FeatureExtractor  (ordered feature dictionary)           |
|                                                                         |
|   2. Preprocessing & Alignment (StandardScaler, LabelEncoder)           |
|                                                                         |
|   3. Inference Engine (Random Forest Classifier)                        |
|      - Label: "legitimate" | "phishing"                                 |
|      - Confidence: predict_proba() percentage                           |
|                                                                         |
|   4. Generative AI Explanation Layer (backend/app/services/)            |
|      - Google Gemini API (gemini-2.5-flash / gemini-3.5-flash)          |
|      - Generates human-readable context from extracted indicators       |
|      - Safe fallback if Gemini is offline / rate-limited                |
+-------------------------------------------------------------------------+
```

---

## 3. Technology Stack

| Layer | Technologies | Role / Rationale |
|---|---|---|
| **Extension** | HTML5, CSS3, Vanilla JS (ES6+), Manifest V3 | Zero build overhead, native Chrome integration, secure permissions |
| **Backend** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 | High-throughput asynchronous REST API, automatic request validation |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib | Deterministic classification, standard feature serialization |
| **URL Analysis** | `urllib.parse`, `tldextract`, `math.log2` | Comprehensive parsing, public suffix extraction, Shannon entropy |
| **Generative AI** | Google GenAI SDK (`google-genai`), Gemini Flash | Zero-hallucination explanation layer conditioned on RF prediction |

---

## 4. Machine Learning & Feature Engineering

SentinelWeb extracts extensive static URL features across three core modules:

### A. Lexical Features (`lexical_features.py`)
- **Length metrics:** Full URL length, domain length, path length, query length, fragment length.
- **Character counts & ratios:** Vowel/consonant count, digit count, letter count, uppercase count.
- **Punctuation counters:** Count of `.`, `-`, `_`, `/`, `?`, `=`, `&`, `%`, `@`, `~`, `!`, `*`, `+`, `$`, `,`, `;`.
- **Lexical signals:** Consecutive hyphens, consecutive dots, directory depth, parameter count, Shannon entropy of URL string.

### B. Domain Features (`domain_features.py`)
- **Domain structure:** FQDN length, subdomain length, subdomain count, suffix/TLD length.
- **Special flags:** Is domain an IP address, has punycode (`xn--`), has port in URL, starts with digit, uses hyphen in domain.
- **Statistical complexity:** Shannon entropy of the domain name.

### C. Security & Heuristic Features (`security_features.py`)
- **Deceptive indicators:** Presence of `@` symbol (basic auth masking), double slash in path (`//`), embedded IP address.
- **Phishing keywords:** Detection of sensitive tokens (`login`, `secure`, `verify`, `account`, `banking`, `update`, `paypal`, `signin`).
- **TLD risk:** Recognition of suspicious or commonly abused top-level domains.

### D. Model Training & Serialization
- The features are scaled via `StandardScaler` fitted exclusively during training.
- Predictions are generated by a balanced `RandomForestClassifier` trained on verified benign and phishing datasets.
- Artifacts preserved under `backend/models/`:
  - `random_forest.pkl`: Trained ensemble model.
  - `scaler.pkl`: Fitted feature standardizer.
  - `label_encoder.pkl`: Class index to string label mapping.
  - `feature_columns.pkl`: Exact ordered list of features expected by the model.

---

## 5. Repository Structure

```
SentinelWeb/
├── .gitignore                      # Clean ignore rules (secrets, venv, pycache)
├── README.md                       # Comprehensive project documentation
├── backend/
│   ├── .env.example                # Template for environment configuration
│   ├── demo.py                     # Interactive CLI testing tool
│   ├── requirements.txt            # Locked backend dependencies
│   ├── models/                     # Trained ML model and preprocessing artifacts
│   │   ├── random_forest.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   └── feature_columns.pkl
│   ├── datasets/                   # Training/testing datasets (CSV)
│   ├── tests/                      # Automated test suite (pytest)
│   │   ├── __init__.py
│   │   ├── test_api.py             # FastAPI endpoint and error tests
│   │   ├── test_feature_extraction.py # Extractor consistency & coverage tests
│   │   └── test_predictor.py       # Predictor inference and boundary tests
│   └── app/
│       ├── main.py                 # FastAPI application with lifespan & CORS
│       ├── api/
│       │   ├── routes.py           # /predict & /health endpoint definitions
│       │   └── schemas.py          # Pydantic request/response models
│       ├── config/
│       │   └── constants.py        # Path anchors and feature definitions
│       ├── core/
│       │   └── logger.py           # Rotating file & console logging system
│       ├── exceptions/
│       │   └── ml_exceptions.py    # Custom domain exception hierarchy
│       ├── ml/
│       │   ├── lexical_features.py
│       │   ├── domain_features.py
│       │   ├── security_features.py
│       │   ├── feature_extractor.py
│       │   ├── preprocessor.py
│       │   └── predictor.py
│       └── services/
│           └── gemini_service.py   # Gemini Generative AI explanation service
└── extension/
    ├── manifest.json               # Manifest V3 configuration
    ├── assets/
    │   └── icon.png                # Extension branding icon
    ├── background/
    │   └── background.js           # MV3 Service Worker (tab tracking & API bridge)
    └── popup/
        ├── popup.html              # Dark-mode security console popup
        ├── popup.css               # Dynamic status styling with CSS :has()
        └── popup.js                # UI state management & rescan logic
```

---

## 6. Setup & Installation Guide

### Prerequisites
- Python 3.11 or higher
- Google Chrome or any Chromium-based browser (Brave, Edge)
- Google Gemini API Key (obtainable at [Google AI Studio](https://aistudio.google.com/))

---

### Step 1: Backend Setup

1. **Navigate to the backend directory:**
   ```powershell
   cd d:\Projects\SentinalWeb-ML\main\SentinelWeb\backend
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
   Open `backend/.env` in an editor and insert your Gemini API Key:
   ```ini
   GEMINI_API_KEY=AIzaSyYourActualApiKeyHere
   GEMINI_MODEL=models/gemini-2.5-flash
   CORS_ORIGINS=*
   PORT=8000
   ```

5. **Start the FastAPI Backend:**
   ```powershell
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
   Verify the backend is live:
   - Health check: `http://127.0.0.1:8000/health`
   - Interactive Docs (Swagger UI): `http://127.0.0.1:8000/docs`

---

### Step 2: Chrome Extension Setup

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** via the toggle in the top-right corner.
3. Click **Load unpacked** in the top-left toolbar.
4. Select the `extension/` directory:
   `d:\Projects\SentinalWeb-ML\main\SentinelWeb\extension`
5. Pin **SentinelWeb** to your Chrome toolbar.
6. Open any webpage and click the SentinelWeb icon. The popup will automatically analyze the active URL and display the verdict!

---

## 7. Interactive CLI Demo

To quickly test the ML model directly from the terminal without launching Chrome or the HTTP server:

```powershell
python demo.py
```
Enter any URL when prompted:
```
==================================================
SentinelWeb - Phishing URL Detector
==================================================

Enter Website URL (or 'exit'): https://paypal-security-update.account-verify.tk/login

========== RESULT ==========
Prediction : phishing
Confidence : 98.4%
Explanation: The URL mimics PayPal using deceptive keywords, excessive subdomains, and an untrusted top-level domain (.tk).
============================
```

---

## 8. Running Automated Tests

A comprehensive automated test suite covers the feature extractors, predictor inference, API validation, and Gemini error handling:

```powershell
cd backend
pytest tests/ -v
```

Expected output:
```
tests/test_feature_extraction.py::TestLexicalFeatureExtractor::test_extracts_dict_for_legitimate_url PASSED
tests/test_feature_extraction.py::TestFeatureExtractorUnified::test_no_key_collisions PASSED
tests/test_predictor.py::TestPredictor::test_predict_structure PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_predict_gemini_fallback PASSED
...
```

---

## 9. API Specification

### `GET /health`
Returns the operational health of the FastAPI service and model status.
- **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### `POST /predict`
Submits a URL for phishing analysis.
- **Request Body:**
  ```json
  {
    "url": "https://secure-login.bankofamerica.com"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "prediction": "legitimate",
    "confidence": 0.965,
    "explanation": "The domain matches the official financial institution and does not exhibit abnormal entropy, deceptive subdomain depth, or known homoglyphs."
  }
  ```

---

## 10. Security & Ethical Considerations

- **Privacy:** SentinelWeb transmits only the active tab's URL to the backend for feature evaluation. Page contents, DOM data, form inputs, and cookies are never accessed or logged.
- **Secrets Protection:** All sensitive credentials (`GEMINI_API_KEY`) are managed through local environment files (`.env`) excluded from version control.
- **CORS Protection:** In production, backend CORS can be restricted to the specific Chrome Extension ID (`chrome-extension://<extension_id>`).
- **Defensive Design:** Random Forest operates deterministically on local infrastructure, guaranteeing reliable classification even in air-gapped environments or during external API outages.

---

## 11. Final-Year Project Defense (Viva FAQ)

1. **Why not let Gemini classify the URL directly?**
   - *Determinism & Latency:* An LLM is non-deterministic, prone to hallucinations, slower (1-3s vs 10ms), and incurs per-request API costs.
   - *Separation of Concerns:* Random Forest provides mathematically rigorous, calibrated classification on statistical features; Gemini translates those statistical signals into user-friendly explanations.
2. **How does SentinelWeb prevent training-serving skew?**
   - Training and inference share identical feature extraction routines (`FeatureExtractor`) and use saved feature column sequences (`feature_columns.pkl`) alongside a persisted `StandardScaler`.
3. **What happens if Gemini is rate-limited or fails?**
   - The backend catches any exception from `gemini_service.py` and returns a standardized fallback explanation. The Random Forest classification and confidence score are never blocked or altered.

---

## 12. Authors & Acknowledgments

- **Project:** SentinelWeb Phishing Detection System
- **Domain:** Cybersecurity, Machine Learning, Applied Generative AI
- **Year:** Final-Year Engineering Project