"""
explainer.py — translate character n-gram weights into plain-English explanations.

This follows the same translation-layer pattern used in shap_to_language.py
(telco churn), text_explainer.py (spam classifier), and risk_explainer.py
(credit risk). The pattern: identify the features driving the model's decision,
translate them into plain English, return them alongside the prediction.
"""

import numpy as np

from config import TOP_N_NGRAMS

LANGUAGE_DESCRIPTIONS: dict[str, str] = {
    'English': 'Latin script with common patterns like "the", "ing", "ion", and short function words',
    'Spanish': 'Latin script with distinctive patterns like "que", "ción", "es", and vowel-heavy endings',
    'French': 'Latin script with patterns like "que", "tion", "les", "est", and nasal vowel markers',
    'German': 'Latin script with patterns like "sch", "ung", "ein", "ich", and compound-word boundaries',
    'Italian': 'Latin script with patterns like "zione", "are", "gli", and double-consonant clusters',
    'Portuguese': 'Latin script with patterns like "ção", "que", "os", "as", and nasal vowel markers',
    'Dutch': 'Latin script with patterns like "ij", "oe", "aa", and double-vowel combinations',
    'Swedish': 'Latin script with patterns like "och", "att", "är", and the distinctive "å", "ä", "ö"',
    'Danish': 'Latin script with patterns like "og", "er", "de", and the distinctive "æ", "ø", "å"',
    'Russian': 'Cyrillic script — completely different character set from Latin; patterns like "ого", "ает", "ные"',
    'Arabic': 'Arabic script — right-to-left, distinctive connected letter shapes and no vowel markers in most text',
    'Turkish': 'Latin script with distinctive vowel harmony patterns and suffixes like "ları", "inin", "dan"',
    'Greek': 'Greek script — distinctive alphabet different from both Latin and Cyrillic; patterns like "ων", "ει", "αι"',
    'Hindi': 'Devanagari script — distinctive curved characters with horizontal bar; patterns reflect agglutinative morphology',
    'Tamil': 'Tamil script — one of the oldest classical languages; distinctive rounded characters',
    'Kannada': 'Kannada script — South Indian script with rounded, complex characters',
    'Malayalam': 'Malayalam script — one of the most visually distinctive scripts with highly curved, circular characters',
}

SCRIPT_MAP: dict[str, str] = {
    'English': 'Latin', 'Spanish': 'Latin', 'French': 'Latin',
    'German': 'Latin', 'Italian': 'Latin', 'Portuguese': 'Latin',
    'Dutch': 'Latin', 'Swedish': 'Latin', 'Danish': 'Latin',
    'Turkish': 'Latin',
    'Russian': 'Cyrillic',
    'Arabic': 'Arabic',
    'Greek': 'Greek',
    'Hindi': 'Devanagari', 'Tamil': 'Tamil',
    'Kannada': 'Kannada', 'Malayalam': 'Malayalam',
}


def _get_base_model(model):
    """Unwrap CalibratedClassifierCV to access coef_ on the base estimator."""
    if hasattr(model, 'calibrated_classifiers_'):
        return model.calibrated_classifiers_[0].estimator
    return model


def get_top_ngrams(
    text: str,
    predicted_language: str,
    vectorizer,
    model,
    label_encoder,
    top_n: int = TOP_N_NGRAMS,
) -> list[dict]:
    """
    Extract the character n-grams in the input text that most strongly
    indicate the predicted language.

    For LinearSVC or LogisticRegression: uses the model's coef_ matrix.
    The predicted language's row contains the weight for each n-gram feature.
    Intersects those weights with n-grams actually present in the input text.

    Args:
        text: the input text
        predicted_language: the detected language name
        vectorizer: fitted TfidfVectorizer
        model: fitted classifier (or CalibratedClassifierCV wrapper)
        label_encoder: fitted LabelEncoder
        top_n: number of top n-grams to return

    Returns:
        List of dicts: [{"ngram": str, "weight": float, "interpretation": str}]
    """
    base = _get_base_model(model)

    if not hasattr(base, 'coef_'):
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = label_encoder.classes_
    lang_indices = np.where(classes == predicted_language)[0]
    if len(lang_indices) == 0:
        return []
    lang_idx = lang_indices[0]

    # TF-IDF vector for the input text — identifies which features are present
    text_vec = vectorizer.transform([text])
    present_feature_indices = text_vec.nonzero()[1]

    if len(present_feature_indices) == 0:
        return []

    # Language-specific weights for features present in the text
    lang_coef = base.coef_[lang_idx]
    present_weights = lang_coef[present_feature_indices]

    # Sort by weight descending
    top_local_idx = np.argsort(present_weights)[-top_n:][::-1]
    top_feature_idx = present_feature_indices[top_local_idx]

    results = []
    for feat_idx in top_feature_idx:
        ngram = feature_names[feat_idx]
        weight = float(lang_coef[feat_idx])
        if weight <= 0:
            continue
        results.append({
            'ngram': ngram,
            'weight': round(weight, 4),
            'interpretation': f'"{ngram}" is a strong indicator of {predicted_language}',
        })

    return results[:top_n]


def explain_detection(
    text: str,
    top_languages: list[dict],
    vectorizer,
    model,
    label_encoder,
) -> dict:
    """
    Generate a plain-English explanation of the language detection result.

    Args:
        text: input text
        top_languages: list of {language, confidence, rank} dicts
        vectorizer: fitted TfidfVectorizer
        model: fitted classifier
        label_encoder: fitted LabelEncoder

    Returns:
        {
          "top_ngrams": list of top n-gram dicts for the predicted language,
          "language_description": str from LANGUAGE_DESCRIPTIONS,
          "explanation": str — one paragraph plain-English explanation,
          "script_type": "Latin" / "Cyrillic" / "Arabic" / "Devanagari" / etc.
        }
    """
    predicted_language = top_languages[0]['language']
    confidence = top_languages[0]['confidence']

    top_ngrams = get_top_ngrams(
        text, predicted_language, vectorizer, model, label_encoder
    )

    language_description = LANGUAGE_DESCRIPTIONS.get(
        predicted_language,
        f'{predicted_language} — distinctive character patterns'
    )
    script_type = SCRIPT_MAP.get(predicted_language, 'other')

    if top_ngrams:
        ngram_strs = ', '.join(f"'{g['ngram']}'" for g in top_ngrams[:3])
        explanation = (
            f"The text was detected as {predicted_language} with "
            f"{confidence * 100:.0f}% confidence based on "
            f"{len(top_ngrams)} character pattern{'s' if len(top_ngrams) != 1 else ''}. "
            f"The strongest signals are: {ngram_strs}. "
            f"{language_description}."
        )
    else:
        explanation = (
            f"The text was detected as {predicted_language} with "
            f"{confidence * 100:.0f}% confidence. "
            f"{language_description}."
        )

    return {
        'top_ngrams': top_ngrams,
        'language_description': language_description,
        'explanation': explanation,
        'script_type': script_type,
    }
