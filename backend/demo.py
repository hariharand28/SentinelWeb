"""Interactive command-line demonstration for SentinelWeb.

Allows testing the Random Forest predictor and optional Gemini explanation
layer directly from the terminal without starting the web server.
"""

from app.ml.predictor import Predictor
from app.services.gemini_service import generate_explanation


def main() -> None:
    print("=" * 55)
    print("  SentinelWeb — Phishing URL Detection CLI Demo")
    print("=" * 55)
    print("Loading machine learning model artifacts...")

    try:
        predictor = Predictor()
        print("[+] Model and preprocessor loaded successfully.")
    except Exception as exc:
        print(f"[-] Failed to load model artifacts: {exc}")
        return

    while True:
        try:
            url = input("\nEnter Website URL (or 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not url:
            continue

        if url.lower() == "exit":
            print("Goodbye!")
            break

        try:
            result = predictor.predict(url)
            confidence_pct = result["confidence"] * 100

            # Attempt Gemini explanation if configured, fallback gracefully if not
            explanation = "Gemini explanation not generated."
            try:
                explanation = generate_explanation(
                    url=url,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    features=result.get("features", {}),
                )
            except Exception as gemini_err:
                explanation = f"(Explanation unavailable: {gemini_err})"

            print("\n==================== RESULT ====================")
            print(f" Analyzed URL : {url}")
            print(f" Prediction   : {result['prediction'].upper()}")
            print(f" Confidence   : {confidence_pct:.1f}%")
            print(f" AI Context   : {explanation}")
            print("================================================")

        except Exception as exc:
            print(f"Error during analysis: {exc}")


if __name__ == "__main__":
    main()