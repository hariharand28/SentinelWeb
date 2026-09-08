# SentinelWeb

## AI-Powered Phishing URL Detection Chrome Extension

SentinelWeb is a machine-learning-based cybersecurity system that detects potentially malicious and phishing URLs through a Chrome browser extension.

The system analyzes the URL of the currently active website, extracts security-related URL and domain features, processes those features using the same preprocessing pipeline used during model training, and uses a **Random Forest Classifier** to determine whether the URL is:

- **Legitimate**
- **Phishing**

After the machine-learning model produces the final classification, **Google Gemini is used only to generate a human-readable explanation** of the result.

> **Important Architecture Principle**
>
> Random Forest makes the final security classification.
>
> Gemini does not make or change the classification. It is used only to explain the machine-learning result.

---

# 1. Project Overview

Phishing attacks commonly use deceptive URLs to trick users into visiting malicious websites and revealing sensitive information.

Traditional users may find it difficult to identify suspicious URLs because phishing websites can use:

- deceptive subdomains
- long and complex URLs
- suspicious paths
- IP addresses instead of domains
- URL encoding
- misleading login or verification paths
- suspicious domain structures
- unusual characters and patterns

SentinelWeb attempts to automate this first-level URL analysis by examining multiple characteristics of a website address and using a trained machine-learning model to classify it.

The system is designed as a practical cybersecurity tool that combines:

1. Browser extension technology
2. Machine learning
3. Backend API architecture
4. URL feature engineering
5. Generative AI explanations

---

# 2. Objectives

The main objectives of SentinelWeb are:

- Detect potentially phishing URLs using machine learning.
- Analyze URLs directly from a Chrome browser.
- Extract meaningful lexical, domain, and security-related URL features.
- Maintain consistency between training-time and inference-time preprocessing.
- Provide a confidence score for the prediction.
- Provide a human-readable explanation of the prediction.
- Separate machine-learning classification from generative AI explanation.
- Provide a simple and practical user interface through a browser extension.

---

# 3. System Architecture

The overall SentinelWeb architecture is:

```text
                    ┌──────────────────────────┐
                    │      Chrome Browser      │
                    │                          │
                    │   SentinelWeb Extension  │
                    └────────────┬─────────────┘
                                 │
                                 │ Active Tab URL
                                 ▼
                    ┌──────────────────────────┐
                    │      FastAPI Backend     │
                    │                          │
                    │      POST /predict       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Feature Extraction   │
                    │                          │
                    │ Lexical Features         │
                    │ Domain Features          │
                    │ Security Features        │
                    │ URL Features             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Preprocessing       │
                    │                          │
                    │ Feature Alignment        │
                    │ Numeric Conversion       │
                    │ Standard Scaling         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Random Forest Model    │
                    │                          │
                    │ Final Prediction         │
                    │ Confidence Score         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Gemini API          │
                    │                          │
                    │ Explanation Generation   │
                    │       ONLY               │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Chrome Extension   │
                    │                          │
                    │ URL                      │
                    │ Prediction               │
                    │ Confidence               │
                    │ Explanation               │
                    └──────────────────────────┘
