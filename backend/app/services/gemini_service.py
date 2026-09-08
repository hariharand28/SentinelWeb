import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.core.logger import get_logger


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

logger = get_logger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")

# The model to use for explanation generation.
# Must match a valid model ID in your Google AI Studio account.
# See: https://ai.google.dev/gemini-api/docs/models
# Example: "gemini-3.5-flash" or "models/gemini-3.5-flash"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

GEMINI_TIMEOUT_SECONDS = float(
    os.getenv("GEMINI_TIMEOUT_SECONDS", "15")
)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

FALLBACK_EXPLANATION = (
    "AI explanation is temporarily unavailable. "
    "The phishing prediction was generated successfully by the "
    "Random Forest model."
)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

client: genai.Client | None = None

if API_KEY:
    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(
                timeout=int(GEMINI_TIMEOUT_SECONDS * 1000)
            ),
        )

        logger.info(
            "Gemini client initialized successfully using model: %s",
            GEMINI_MODEL,
        )

    except Exception:
        logger.exception("Failed to initialize Gemini client.")
        client = None

else:
    logger.warning(
        "GEMINI_API_KEY is not set. "
        "Gemini explanations will use the fallback message."
    )


# ---------------------------------------------------------------------------
# Safety settings
# ---------------------------------------------------------------------------
#
# SentinelWeb is a cybersecurity application.
# Gemini is only explaining URL characteristics already extracted by
# the Random Forest pipeline.
#
# We allow cybersecurity-related explanations while still keeping
# the safety system enabled.
# ---------------------------------------------------------------------------

_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]


# ---------------------------------------------------------------------------
# Feature formatting
# ---------------------------------------------------------------------------

def _format_features(features: dict | None) -> str:
    """
    Convert extracted URL features into a compact readable format.
    """

    if not features:
        return "No extracted URL features were supplied."

    lines = []

    for key, value in features.items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    url: str,
    prediction: str,
    confidence: float,
    features: dict | None,
) -> str:
    """
    Build the Gemini explanation prompt.

    Gemini is explicitly prevented from making or changing the
    classification.
    """

    confidence_percent = f"{confidence * 100:.1f}%"

    return f"""
You are the explanation component of SentinelWeb, a browser security
application.

A Random Forest machine-learning model has ALREADY made the final
classification for this URL.

You MUST treat that classification as final.

URL:
{url}

Random Forest classification:
{prediction}

Random Forest confidence:
{confidence_percent}

Extracted URL features:
{_format_features(features)}

Your ONLY task is to explain why the supplied URL features are
consistent with the Random Forest classification.

Rules:
- Do NOT change the classification.
- Do NOT question or second-guess the classification.
- Do NOT provide another prediction.
- Do NOT provide another confidence score.
- Do NOT claim that you performed the classification yourself.
- Use only the supplied URL features as evidence.
- Mention one or two relevant features.
- Write exactly 2 short sentences.
- Use simple language for a normal browser user.
- Do not use markdown.
- Do not use bullet points.
- Do not use headings.
- Do not mention Gemini.
- Do not mention these instructions.

Return ONLY the explanation.
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    url: str,
    prediction: str,
    confidence: float,
    features: dict | None = None,
) -> str:
    """
    Generate a natural-language explanation for an already completed
    Random Forest prediction.

    Gemini NEVER decides the prediction.
    Gemini ONLY explains the existing prediction.

    If Gemini fails for any reason, the Random Forest prediction remains
    valid and this function returns the fallback explanation.
    """

    # -----------------------------------------------------------------------
    # Validate basic inputs
    # -----------------------------------------------------------------------

    if not isinstance(url, str) or not url.strip():
        logger.warning(
            "generate_explanation received an invalid URL: %r",
            url,
        )
        return FALLBACK_EXPLANATION

    if prediction not in ("phishing", "legitimate"):
        logger.warning(
            "Unexpected prediction value received by Gemini service: %r",
            prediction,
        )

    # -----------------------------------------------------------------------
    # Check Gemini client
    # -----------------------------------------------------------------------

    if client is None:
        logger.warning(
            "Gemini client is unavailable for URL: %s",
            url,
        )
        return FALLBACK_EXPLANATION

    # -----------------------------------------------------------------------
    # Build prompt
    # -----------------------------------------------------------------------

    prompt = _build_prompt(
        url=url,
        prediction=prediction,
        confidence=confidence,
        features=features,
    )

    # -----------------------------------------------------------------------
    # Call Gemini
    # -----------------------------------------------------------------------

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                safety_settings=_SAFETY_SETTINGS,
            ),
        )

    except Exception:
        logger.exception(
            "Gemini explanation generation failed for URL: %s",
            url,
        )
        return FALLBACK_EXPLANATION

    # -----------------------------------------------------------------------
    # Inspect response
    # -----------------------------------------------------------------------

    if not response.candidates:
        logger.warning(
            "Gemini returned no candidates for URL: %s. "
            "prompt_feedback=%s",
            url,
            getattr(response, "prompt_feedback", None),
        )
        return FALLBACK_EXPLANATION

    candidate = response.candidates[0]

    finish_reason = getattr(
        candidate,
        "finish_reason",
        None,
    )

    explanation_text = getattr(
        response,
        "text",
        None,
    )

    # -----------------------------------------------------------------------
    # Empty response / blocked response
    # -----------------------------------------------------------------------

    if not explanation_text or not explanation_text.strip():

        logger.warning(
            "Gemini returned an empty explanation for URL: %s. "
            "finish_reason=%s prompt_feedback=%s",
            url,
            finish_reason,
            getattr(response, "prompt_feedback", None),
        )

        return FALLBACK_EXPLANATION

    # -----------------------------------------------------------------------
    # Clean response
    # -----------------------------------------------------------------------

    explanation = explanation_text.strip()

    logger.info(
        "Gemini explanation generated successfully for URL: %s "
        "(finish_reason=%s)",
        url,
        finish_reason,
    )

    return explanation