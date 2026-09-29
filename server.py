"""
Backend Web Server for News Bias Detection.

Provides REST API endpoints for inference, sample articles, model evaluation
metrics, and serves the minimal frontend interface.
"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ARTIFACTS_DIR, BASE_DIR, DATA_DIR, METADATA_PATH, MODEL_PATH
from src.predict import BiasPredictor

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Singleton predictor instance initialized lazily or on startup
predictor_instance: Optional[BiasPredictor] = None


def get_predictor() -> BiasPredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = BiasPredictor()
    return predictor_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up predictor on server launch."""
    try:
        get_predictor()
        print("[INFO] BiasPredictor loaded and ready.")
    except Exception as e:
        print(f"[WARNING] BiasPredictor warmup failed: {e}")
    yield


app = FastAPI(
    title="News Bias Detection API",
    description="Feedforward Neural Network with GloVe Embeddings for Political Stance Classification",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    token_count: int
    in_vocab_count: int
    oov_count: int
    coverage_rate: float
    oov_tokens: List[str]


def run_prediction_analysis(text: str) -> Dict[str, Any]:
    """Helper to run inference and extract enriched vocabulary breakdown."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    predictor = get_predictor()
    tokens = predictor.preprocessor.tokenize(text)
    if not tokens:
        raise HTTPException(
            status_code=400,
            detail="No valid tokens found after preprocessing (min word length: 2)."
        )

    # In-vocab vs OOV token breakdown
    in_vocab_list = []
    oov_list = []
    for token in tokens:
        if token in predictor.embedding_manager.embeddings_dict:
            in_vocab_list.append(token)
        else:
            oov_list.append(token)

    raw_result = predictor.predict_text(text)

    # Convert all numpy values to pure Python types for clean serialization
    pred_class = str(raw_result["predicted_class"])
    confidence = float(raw_result["confidence"])
    probabilities = {
        str(k): round(float(v), 4)
        for k, v in raw_result["probabilities"].items()
    }
    token_count = int(raw_result["token_count"])
    in_vocab_count = int(raw_result["in_vocab_count"])
    oov_count = int(raw_result["oov_count"])
    coverage = round((in_vocab_count / token_count * 100), 1) if token_count > 0 else 0.0

    # Get unique OOV tokens for display
    unique_oov = sorted(list(set(oov_list)))[:25]

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "token_count": token_count,
        "in_vocab_count": in_vocab_count,
        "oov_count": oov_count,
        "coverage_rate": coverage,
        "oov_tokens": unique_oov,
    }


# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor_instance is not None,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_api(request: PredictRequest):
    return run_prediction_analysis(request.text)


@app.post("/api/predict-file", response_model=PredictResponse)
async def predict_file_api(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".text", ".md", ".csv")):
        raise HTTPException(status_code=400, detail="Only text files (.txt, .md) are supported.")
    
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="ignore")
    
    return run_prediction_analysis(text)


@app.get("/api/samples")
async def get_sample_articles():
    """Returns curated representative articles across stances for 1-click testing."""
    samples = [
        {
            "id": "sample-1",
            "title": "Healthcare as a Universal Right",
            "category": "Healthcare Policy",
            "expected_stance": "liberal",
            "text": "Access to comprehensive healthcare must be recognized as a fundamental human right rather than a market commodity. Millions of working families are crushed by predatory private insurance premiums and unaffordable prescription drugs. Implementing a single-payer Medicare for All system will eliminate bureaucratic waste, guarantee coverage regardless of employment status, and prioritize patient well-being over corporate pharmaceutical profits."
        },
        {
            "id": "sample-2",
            "title": "Free Enterprise & Regulatory Overhaul",
            "category": "Energy & Economy",
            "expected_stance": "conservative",
            "text": "The proposed regulatory overhaul strikes at the heart of free enterprise and market competition. By imposing centralized federal mandates and heavy compliance costs on independent energy producers, the administration risks stifling domestic capital investment and weakening American energy independence. True economic growth requires deregulation, fiscal discipline, and unleashing the power of private innovation."
        },
        {
            "id": "sample-3",
            "title": "Congressional Budget Office Healthcare Review",
            "category": "Fiscal Analysis",
            "expected_stance": "neutral",
            "text": "The Congressional Budget Office published its latest analysis comparing public health options with private exchange subsidies. The report outlines that while universal coverage options reduce uninsured rates by an estimated twelve percent, aggregate federal outlays would increase over the ten-year budget window. Lawmakers continue debating the trade-offs between deficit impact and expanded Medicaid eligibility."
        },
        {
            "id": "sample-4",
            "title": "Climate Emergency & Renewable Transition",
            "category": "Environment",
            "expected_stance": "liberal",
            "text": "The climate emergency represents an existential catastrophe that demands an immediate transition away from fossil fuels. Fossil fuel corporations have knowingly polluted our atmosphere and deceived the public for decades. We must enact a bold Green New Deal, mandate clean renewable energy standards, and heavily subsidize solar, wind, and public transit to protect frontline environmental justice communities."
        },
        {
            "id": "sample-5",
            "title": "Energy Independence & Pipeline Permitting",
            "category": "Energy Policy",
            "expected_stance": "conservative",
            "text": "Unrealistic climate mandates and radical green regulations threaten American energy independence and inflict devastating energy inflation on working households. Nuclear power, domestic oil extraction, and natural gas pipelines must be expanded through regulatory relief to ensure national security and power the industrial economy."
        },
        {
            "id": "sample-6",
            "title": "International Energy Agency Annual Report",
            "category": "Energy Data",
            "expected_stance": "neutral",
            "text": "The International Energy Agency released its annual global energy review detailing trends in electricity generation. Renewables accounted for forty percent of newly added grid capacity, while natural gas remained the leading baseload power provider. The report highlights grid reliability challenges during peak seasonal transitions."
        }
    ]
    return samples


@app.get("/api/metrics")
async def get_metrics():
    """Returns training metadata and evaluation figures."""
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Metadata not found.")
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "metadata": metadata,
        "has_confusion_matrix": (ARTIFACTS_DIR / "confusion_matrix.png").exists(),
        "has_training_plot": (ARTIFACTS_DIR / "training_history.png").exists(),
    }


# Static artifact serving for plots
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

# Static frontend serving
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found at frontend/index.html"}


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start uvicorn server."""
    print(f"Starting News Bias Detection web app on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
