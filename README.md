# 🛡️ SentinelWeb

## AI-Powered Phishing URL Detection Chrome Extension

SentinelWeb is a machine-learning-based cybersecurity system that detects potentially phishing and malicious URLs through a Chrome browser extension.

The system analyzes the URL of the currently active website, extracts multiple lexical, domain, URL, and security-related features, preprocesses them using the same schema used during model training, and uses a **Random Forest Classifier** to classify the URL as:

- ✅ Legitimate
- 🚨 Phishing

After the machine-learning model produces the final prediction, **Google Gemini** is used only to generate a human-readable explanation of the result.

> **Core Architecture Principle**
>
> **Random Forest = Final Prediction**
>
> **Gemini = Explanation Only**
>
> Gemini does not replace, override, or modify the machine-learning classification.

---

# 📌 1. Project Overview

Phishing attacks are one of the most common forms of cyberattack. Attackers create deceptive URLs and websites that imitate legitimate services in order to trick users into revealing sensitive information such as passwords, payment information, and account credentials.

Many phishing URLs contain suspicious structural characteristics such as:

- unusually long URLs
- excessive subdomains
- IP addresses instead of domain names
- suspicious login or verification paths
- encoded characters
- excessive special characters
- suspicious query parameters
- punycode domains
- URL-shortening services
- unusual domain patterns

SentinelWeb uses these characteristics as machine-learning features and applies a trained Random Forest model to identify suspicious URLs.

The result is exposed through a FastAPI backend and displayed through a Chrome browser extension.

---

# 🎯 2. Project Objectives

The primary objectives of SentinelWeb are:

1. Detect potentially phishing URLs using machine learning.
2. Analyze URLs directly from the user's Chrome browser.
3. Extract meaningful URL and domain security features.
4. Maintain consistent preprocessing between training and inference.
5. Provide a prediction confidence score.
6. Generate a human-readable explanation using Gemini.
7. Separate deterministic machine-learning prediction from generative AI explanation.
8. Provide a simple browser-based cybersecurity interface.
9. Demonstrate the integration of machine learning, backend APIs, browser extensions, and generative AI in a practical cybersecurity application.

---

# 🧠 3. Core Concept

SentinelWeb follows this architecture:

```text
              ┌───────────────────────────┐
              │      Google Chrome        │
              │                           │
              │  SentinelWeb Extension   │
              └─────────────┬─────────────┘
                            │
                            │ Current URL
                            ▼
              ┌───────────────────────────┐
              │       FastAPI Backend     │
              │                           │
              │       POST /predict       │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │     Feature Extraction    │
              │                           │
              │  Lexical / Domain / URL  │
              │       / Security          │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │       Preprocessing       │
              │                           │
              │ Feature Alignment         │
              │ Numeric Conversion        │
              │ Scaling                   │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │    Random Forest Model    │
              │                           │
              │ Final Classification      │
              │ Confidence Score          │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │       Gemini API          │
              │                           │
              │ Explanation Generation    │
              │        ONLY               │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │      Chrome Extension     │
              │                           │
              │ Prediction                │
              │ Confidence                │
              │ Explanation               │
              └───────────────────────────┘
