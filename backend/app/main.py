"""SentinelWeb FastAPI application entry point.

Starts the FastAPI application, configures CORS, and loads the ML
Predictor during the application lifespan so that:

1. Model artifacts are loaded exactly once at startup, not at
   module-import time.
2. A missing model file surfaces as a startup error rather than
   crashing the first prediction request.
3. The predictor instance is shared across all requests via app.state.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logger import get_logger
from app.ml.predictor import Predictor

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts on startup; release on shutdown."""
    logger.info("SentinelWeb API starting up — loading ML artifacts...")
    app.state.predictor = Predictor()
    logger.info("ML artifacts loaded. API is ready.")
    yield
    logger.info("SentinelWeb API shutting down.")


app = FastAPI(
    title="SentinelWeb API",
    version="1.0.0",
    description="AI-powered phishing detection API for SentinelWeb.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# Allow the Chrome extension to reach the local backend.
# In production, restrict allow_origins to your deployed domain.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

from app.api.routes import router  # noqa: E402 — import after app creation

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "SentinelWeb API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}