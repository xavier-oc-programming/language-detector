import json
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

import explainer
from config import MODEL_DIR, TOP_N_LANGUAGES, TOP_N_NGRAMS

app = FastAPI(
    title="Language Detector API",
    description=(
        "Detects the language of any text using character n-gram TF-IDF features. "
        "Returns top 3 languages with confidence scores and an explanation of the "
        "character patterns driving the detection. Supports 22 languages."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

# ── Model loading ────────────────────────────────────────────────────────────

MODEL_LOADED = False
_model = None
_vectorizer = None
_label_encoder = None
_model_name = ""
_metrics = {}
_language_list = []


def _load_models():
    global MODEL_LOADED, _model, _vectorizer, _label_encoder
    global _model_name, _metrics, _language_list

    required = [
        MODEL_DIR / 'best_model.pkl',
        MODEL_DIR / 'vectorizer.pkl',
        MODEL_DIR / 'label_encoder.pkl',
        MODEL_DIR / 'best_model_name.txt',
        MODEL_DIR / 'model_metrics.json',
        MODEL_DIR / 'language_list.json',
    ]
    if not all(p.exists() for p in required):
        print("Model files not found — running in degraded mode (train first)")
        return

    _model = pickle.load(open(MODEL_DIR / 'best_model.pkl', 'rb'))
    _vectorizer = pickle.load(open(MODEL_DIR / 'vectorizer.pkl', 'rb'))
    _label_encoder = pickle.load(open(MODEL_DIR / 'label_encoder.pkl', 'rb'))
    _model_name = (MODEL_DIR / 'best_model_name.txt').read_text().strip()
    _metrics = json.loads((MODEL_DIR / 'model_metrics.json').read_text())
    _language_list = json.loads((MODEL_DIR / 'language_list.json').read_text())
    MODEL_LOADED = True
    print(f"Model loaded: {_model_name}")


_load_models()

# ── Flag emojis ───────────────────────────────────────────────────────────────

FLAG_EMOJIS: dict[str, str] = {
    'English': '🇬🇧', 'Spanish': '🇪🇸', 'French': '🇫🇷',
    'Dutch': '🇳🇱', 'Swedish': '🇸🇪', 'Estonian': '🇪🇪',
    'Indonesian': '🇮🇩', 'Romanian': '🇷🇴', 'Turkish': '🇹🇷',
    'Portugese': '🇵🇹', 'Latin': '🏛️',
    'Russian': '🇷🇺',
    'Arabic': '🇸🇦', 'Persian': '🇮🇷', 'Urdu': '🇵🇰', 'Pushto': '🇦🇫',
    'Hindi': '🇮🇳', 'Tamil': '🇱🇰',
    'Chinese': '🇨🇳', 'Japanese': '🇯🇵', 'Korean': '🇰🇷',
    'Thai': '🇹🇭',
}

# ── Pydantic models ────────────────────────────────────────────────────────────


class DetectionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Text to detect language of (10–5000 characters)",
    )


class LanguageResult(BaseModel):
    rank: int
    language: str
    confidence: float
    confidence_pct: int
    flag_emoji: str


class DetectionResponse(BaseModel):
    detected_language: str
    confidence: float
    confidence_pct: int
    top_languages: list[LanguageResult]
    top_ngrams: list[dict]
    explanation: str
    script_type: str
    model_used: str
    text_length: int


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_name": _model_name,
        "supported_languages": len(_language_list),
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(req: DetectionRequest):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first to generate model files.",
        )

    text = req.text
    X = _vectorizer.transform([text])

    # Get probabilities — all loaded models support predict_proba
    proba = _model.predict_proba(X)[0]
    classes = _label_encoder.classes_

    # Top N languages by confidence
    top_idx = np.argsort(proba)[-TOP_N_LANGUAGES:][::-1]
    top_languages_raw = [
        {
            'rank': i + 1,
            'language': classes[idx],
            'confidence': float(proba[idx]),
        }
        for i, idx in enumerate(top_idx)
    ]

    top_language_results = [
        LanguageResult(
            rank=t['rank'],
            language=t['language'],
            confidence=round(t['confidence'], 4),
            confidence_pct=int(round(t['confidence'] * 100)),
            flag_emoji=FLAG_EMOJIS.get(t['language'], '🏳️'),
        )
        for t in top_languages_raw
    ]

    explanation_data = explainer.explain_detection(
        text=text,
        top_languages=top_languages_raw,
        vectorizer=_vectorizer,
        model=_model,
        label_encoder=_label_encoder,
    )

    best = top_language_results[0]
    return DetectionResponse(
        detected_language=best.language,
        confidence=best.confidence,
        confidence_pct=best.confidence_pct,
        top_languages=top_language_results,
        top_ngrams=explanation_data['top_ngrams'],
        explanation=explanation_data['explanation'],
        script_type=explanation_data['script_type'],
        model_used=_model_name,
        text_length=len(text),
    )


@app.get("/api/languages")
async def languages():
    script_map = explainer.SCRIPT_MAP
    desc_map = explainer.LANGUAGE_DESCRIPTIONS
    return [
        {
            'language': lang,
            'flag': FLAG_EMOJIS.get(lang, '🏳️'),
            'script': script_map.get(lang, 'other'),
            'description': desc_map.get(lang, ''),
        }
        for lang in _language_list
    ]


@app.get("/api/model-info")
async def model_info():
    return _metrics
