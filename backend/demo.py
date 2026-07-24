from app.ml.predictor import Predictor

def main():
    predictor = Predictor()

    print("=" * 50)
    print("SentinelWeb - Phishing URL Detector")
    print("=" * 50)

    while True:
        url = input("\nEnter Website URL (or 'exit'): ")

        if url.lower() == "exit":
            print("Goodbye!")
            break

        try:
            result = predictor.predict(url)

            print("\n========== RESULT ==========")
            print("Prediction :", result["prediction"])
            print("Confidence :", result["confidence"])
            print("Risk Level :", result.get("risk", "Unknown"))
            print("Reason     :", result.get("reason", "No explanation"))
            print("============================")

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()