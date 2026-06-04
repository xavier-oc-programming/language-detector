# language-detector

Detects the language of any text using character n-gram TF-IDF features.
Identifies 17 languages and explains which character patterns drove the
detection — showing the top n-grams that are most characteristic of the
detected language.

**Live demo → [language-detector-xoc.azurewebsites.net](https://language-detector-xoc.azurewebsites.net)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**API docs → [/docs](https://language-detector-xoc.azurewebsites.net/docs)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**Notebook → notebook.ipynb**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![TF-IDF](https://img.shields.io/badge/features-TF--IDF-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Azure App Service](https://img.shields.io/badge/deploy-Azure%20App%20Service-blue)

---

## 0. Prerequisites

- Python 3.11+
- pip

## 1. Quick start

```bash
git clone https://github.com/xavier-oc-programming/language-detector
cd language-detector
pip install -r requirements.txt

# Train the model (generates models/ and plots/)
python train.py

# Start the API
uvicorn main:app --reload

# Run tests
pytest tests/ -v
```

Open `http://localhost:8000` for the demo UI. API docs at `http://localhost:8000/docs`.

## 2. Project structure

```
language-detector/
├── config.py              # Single source of truth for all constants
├── train.py               # Training script — runs end to end
├── explainer.py           # N-gram → plain-English explanation layer
├── main.py                # FastAPI application
├── Dockerfile             # Container definition
├── startup.txt            # Azure App Service startup command
├── notebook.ipynb         # Full walkthrough notebook
├── requirements.txt
├── portfolio.yaml
├── .github/workflows/ci.yml
├── templates/index.html   # Demo UI
├── tests/test_api.py
├── models/                # Trained model artefacts (committed)
└── plots/                 # Training visualisations (committed)
```

## 3. Dataset

Language Detection Dataset — ~22,000 text samples across 17 languages.

| Language   | Language   | Language  |
|------------|------------|-----------|
| English    | Italian    | Russian   |
| Spanish    | Portuguese | Arabic    |
| French     | Dutch      | Turkish   |
| German     | Swedish    | Greek     |
| Hindi      | Danish     | Tamil     |
| Malayalam  | Kannada    |           |

Source: [basilb2s/language-detection](https://www.kaggle.com/datasets/basilb2s/language-detection) on Kaggle.
If Kaggle is unavailable, `train.py` falls back to a GitHub-hosted CSV mirror, then to a
Wikipedia-generated synthetic dataset.

## 4. Why character n-grams

Word-level features require the model to have seen the exact words before. A word it
has never seen is invisible. Character n-grams have no out-of-vocabulary problem —
every text can be represented as a combination of character sequences, even if the
words are new.

`analyzer='char_wb'` respects word boundaries, which is important: word-boundary
patterns (how words start and end) are highly distinctive across languages. The German
trigram `sch` and the Spanish bigram `qu` are reliable language signals regardless of
which words contain them. `ngram_range=(1, 3)` captures single characters (script
detection), bigrams (letter combinations), and trigrams (morphological patterns).

## 5. Models

Three classifiers trained on the same TF-IDF features and compared by macro F1:

| Model | Notes |
|---|---|
| **LinearSVC** | Standard choice for high-dimensional sparse text; wrapped in `CalibratedClassifierCV` for probability output |
| **LogisticRegression** | Provides calibrated `predict_proba`; multinomial with lbfgs solver |
| **ComplementNB** | Outperforms MultinomialNB on imbalanced text in prior experiments; generative baseline |

The best model by macro F1 is saved and used by the API.

## 6. Results

*TBD — populate after running `train.py`.*

Expected: accuracy above 97%, macro F1 above 0.97. Non-Latin script languages
(Arabic, Russian, Greek, Hindi, Malayalam, Tamil, Kannada) are typically the
easiest to detect due to their distinctive character sets.

## 7. N-gram analysis

The most interesting output of this project is not the accuracy — high accuracy on
language detection is expected with character n-grams. The interesting output is the
n-gram weight analysis: what character sequences did the model learn are most
characteristic of each language?

- **Arabic**: patterns no Latin script language shares
- **Russian**: Cyrillic patterns distinct from Greek
- **Spanish**: `que`, `ción`, `es` — patterns every Spanish speaker recognises
- **German**: `sch`, `ung`, `ein` — morphological fingerprints

See `plots/04_top_ngrams_per_language.png` and Cell 8 of `notebook.ipynb`.

## 8. Visualisations

| Plot | Description |
|---|---|
| `01_language_distribution.png` | Sample count per language — class balance |
| `02_text_length_by_language.png` | Text length distribution by language |
| `04_top_ngrams_per_language.png` | Top 10 distinctive n-grams for 6 languages |
| `05_confusion_matrix.png` | Confusion matrix on test set |
| `06_model_comparison.png` | Accuracy and macro F1 across all three models |
| `07_per_language_f1.png` | Per-language F1 for the best model |

## 9. API Reference

### `POST /detect`

```json
// Request
{ "text": "El rápido zorro marrón salta sobre el perro perezoso." }

// Response
{
  "detected_language": "Spanish",
  "confidence": 0.9873,
  "confidence_pct": 99,
  "top_languages": [
    { "rank": 1, "language": "Spanish", "confidence": 0.9873, "confidence_pct": 99, "flag_emoji": "🇪🇸" },
    { "rank": 2, "language": "Portuguese", "confidence": 0.0094, "confidence_pct": 1, "flag_emoji": "🇵🇹" },
    { "rank": 3, "language": "Italian", "confidence": 0.0018, "confidence_pct": 0, "flag_emoji": "🇮🇹" }
  ],
  "top_ngrams": [
    { "ngram": "que", "weight": 2.14, "interpretation": "'que' is a strong indicator of Spanish" },
    ...
  ],
  "explanation": "The text was detected as Spanish with 99% confidence...",
  "script_type": "Latin",
  "model_used": "LogisticRegression",
  "text_length": 53
}
```

### `GET /api/languages`

Returns all 17 supported languages with flag emoji, script type, and description.

### `GET /api/model-info`

Returns model accuracy, macro F1, and per-language metrics.

### `GET /health`

Returns `{ "status": "ok", "model_loaded": true, ... }`.

Full interactive docs at `/docs` (Swagger UI).

## 10. Deployment — Azure App Service

```bash
# Create resource group and app service plan
az group create --name language-detector-rg --location westeurope
az appservice plan create --name language-detector-plan \
  --resource-group language-detector-rg --sku B1 --is-linux
# Scale to F1 via portal after creation if preferred

# Create web app
az webapp create --name language-detector-xoc \
  --resource-group language-detector-rg \
  --plan language-detector-plan --runtime "PYTHON:3.11"

# Configure startup command
az webapp config set --name language-detector-xoc \
  --resource-group language-detector-rg \
  --startup-file "gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"

# Enable build during deployment
az webapp config appsettings set --name language-detector-xoc \
  --resource-group language-detector-rg \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

# Deploy
cd language-detector
zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*" -x "*.ipynb_checkpoints*"
az webapp deployment source config-zip \
  --name language-detector-xoc \
  --resource-group language-detector-rg \
  --src deploy.zip
```

## 11. CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Run `pytest tests/ -v`

See `.github/workflows/ci.yml`.

## 12. Design decisions

**Why character n-grams over word n-grams for language detection**

Character n-grams have no out-of-vocabulary problem. A word n-gram model cannot
represent text containing words it has never seen during training — a significant
limitation given the vocabulary breadth of 17 languages. Character sequences are
universal: every text is representable as a combination of character n-grams,
regardless of topic or vocabulary. This is why character features dominate language
detection benchmarks.

**Why `char_wb` over `char` analyzer**

`char_wb` inserts word-boundary markers before extracting n-grams, so n-grams never
span word boundaries. This matters because the patterns at word beginnings and endings
are among the most language-discriminative features. German words often end in `-ung`,
Spanish verbs end in `-ar`, `-er`, `-ir`. The `char` analyzer would produce n-grams
that span spaces, conflating word-boundary patterns with mid-word patterns.

**Why three models**

LinearSVC is the standard choice for high-dimensional sparse text features — the same
decision made in the spam classifier project. It trains fast and typically achieves
top accuracy on text classification. LogisticRegression provides calibrated
`predict_proba` scores needed for the confidence display in the UI. ComplementNB
was shown to outperform MultinomialNB on imbalanced text classification in prior work;
including it here tests whether that finding generalises to language detection. The
three models cover discriminative linear, probabilistic linear, and generative
approaches to the same problem.

**Why `explainer.py` follows the same pattern as previous projects**

Four projects in this portfolio implement the same translation-layer principle:
identify the features driving the model's prediction, translate them into plain
English, return them alongside the prediction. This is not a portfolio pattern for
its own sake — it is the design principle that makes ML outputs usable by
non-technical people. `explainer.py` is the language detection version of
`shap_to_language.py` (telco churn), `text_explainer.py` (spam classifier), and
`risk_explainer.py` (credit risk).

**Why this project is personally relevant**

I am bilingual in English and Spanish and have studied French. Language technology is
a natural extension of the NLP work already in the portfolio. Testing the model on
Spanish text — the language I grew up speaking — and seeing `que`, `es`, `ción` come
up as the top signals was the most interesting part of building this. The model
learned them from character statistics, not from grammar rules.

## 13. Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥2.0 | Data loading and cleaning |
| numpy | ≥1.24,<2.0 | Numerical operations |
| scikit-learn | ≥1.3 | TF-IDF, classifiers, metrics |
| matplotlib | ≥3.7 | Training visualisations |
| seaborn | ≥0.12 | Confusion matrix heatmap |
| fastapi | ≥0.110 | REST API framework |
| uvicorn | ≥0.27 | ASGI server |
| gunicorn | ≥21.0 | Production server (Azure) |
| pydantic | ≥2.0 | Request/response validation |
| jinja2 | ≥3.1 | HTML template rendering |
| jupyter | ≥1.0 | Notebook |
| pytest | ≥7.0 | API tests |
| httpx | ≥0.27 | Test client |
| python-multipart | ≥0.0.9 | Form data support |
