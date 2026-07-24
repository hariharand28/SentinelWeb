from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="SentinelWeb API",
    version="1.0.0",
    description="AI-powered phishing detection API for SentinelWeb.",
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "SentinelWeb API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}