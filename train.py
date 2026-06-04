import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    ANALYZER,
    MAX_FEATURES,
    MODEL_DIR,
    NGRAM_RANGE,
    PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

warnings.filterwarnings('ignore')

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    print("Loading dataset...")

    # Try kagglehub first
    try:
        import kagglehub
        path = kagglehub.dataset_download('basilb2s/language-detection')
        csv_files = list(Path(path).rglob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            print(f"Loaded from kagglehub: {csv_files[0]}")
            return df
    except Exception as e:
        print(f"kagglehub unavailable: {e}")

    # Try multiple GitHub mirrors in order
    fallback_urls = [
        'https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv',
        'https://raw.githubusercontent.com/lyteabovenyte/NLP_Classification/main/Language_Detection.csv',
    ]
    for url in fallback_urls:
        try:
            df = pd.read_csv(url)
            print(f"Loaded from {url}: {len(df)} rows")
            return df
        except Exception as e:
            print(f"URL unavailable ({url}): {e}")

    # Fall back to synthetic dataset using Wikipedia
    print("Generating synthetic dataset from Wikipedia...")
    return generate_synthetic_dataset()


def generate_synthetic_dataset() -> pd.DataFrame:
    import wikipedia

    languages_wiki = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
        'Italian': 'it', 'Portuguese': 'pt', 'Dutch': 'nl', 'Swedish': 'sv',
        'Danish': 'da', 'Russian': 'ru', 'Arabic': 'ar', 'Turkish': 'tr',
        'Greek': 'el', 'Hindi': 'hi', 'Tamil': 'ta', 'Kannada': 'kn',
        'Malayalam': 'ml',
    }

    topics = ['Science', 'History', 'Geography', 'Technology', 'Culture',
              'Politics', 'Economics', 'Sports', 'Mathematics', 'Philosophy']

    records = []
    samples_per_language = 1000

    for lang_name, lang_code in languages_wiki.items():
        wikipedia.set_lang(lang_code)
        collected = []
        for topic in topics:
            if len(collected) >= samples_per_language:
                break
            try:
                results = wikipedia.search(topic, results=5)
                for title in results:
                    if len(collected) >= samples_per_language:
                        break
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        text = page.content
                        # Split into paragraphs
                        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
                        for para in paragraphs[:10]:
                            if len(collected) >= samples_per_language:
                                break
                            collected.append(para[:500])
                    except Exception:
                        continue
            except Exception:
                continue

        for text in collected[:samples_per_language]:
            records.append({'Text': text, 'Language': lang_name})
        print(f"  {lang_name}: {min(len(collected), samples_per_language)} samples")

    df = pd.DataFrame(records)
    print(f"Synthetic dataset: {len(df)} samples across {df['Language'].nunique()} languages")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names
    df.columns = df.columns.str.strip()
    if 'language' in df.columns and 'Language' not in df.columns:
        df = df.rename(columns={'language': 'Language'})
    if 'text' in df.columns and 'Text' not in df.columns:
        df = df.rename(columns={'text': 'Text'})

    df = df.dropna(subset=['Text', 'Language'])
    df['Text'] = df['Text'].str.strip()
    df['Language'] = df['Language'].str.strip()
    df = df[df['Text'].str.len() > 0]
    return df.reset_index(drop=True)


def print_class_distribution(df: pd.DataFrame):
    print("\n--- Class distribution ---")
    dist = df['Language'].value_counts()
    for lang, count in dist.items():
        print(f"  {lang:15s}: {count:5d}")
    print(f"\nTotal samples : {len(df)}")
    print(f"Languages     : {df['Language'].nunique()}")


def plot_language_distribution(df: pd.DataFrame):
    dist = df['Language'].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dist)))
    dist.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Sample Count by Language', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Language')
    plt.tight_layout()
    path = PLOTS_DIR / '01_language_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_text_length(df: pd.DataFrame):
    df = df.copy()
    df['text_length'] = df['Text'].str.len()
    order = df.groupby('Language')['text_length'].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    df.boxplot(column='text_length', by='Language', ax=ax,
               vert=False, showfliers=False,
               order=order)
    ax.set_title('Text Length Distribution by Language', fontsize=14, fontweight='bold')
    plt.suptitle('')
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Language')
    plt.tight_layout()
    path = PLOTS_DIR / '02_text_length_by_language.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def print_sample_texts(df: pd.DataFrame):
    print("\n--- One sample text per language ---")
    for lang in sorted(df['Language'].unique()):
        sample = df[df['Language'] == lang]['Text'].iloc[0]
        preview = sample[:120].replace('\n', ' ')
        print(f"\n[{lang}]\n  {preview}...")


def train_models(X_train, y_train, label_encoder):
    models = {}

    print("\nTraining LinearSVC...")
    svc = LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
    svc.fit(X_train, y_train)
    # Wrap with CalibratedClassifierCV so we get predict_proba
    svc_cal = CalibratedClassifierCV(svc, cv='prefit')
    svc_cal.fit(X_train, y_train)
    models['LinearSVC'] = svc_cal

    print("Training LogisticRegression...")
    lr = LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000,
        multi_class='multinomial', solver='lbfgs', n_jobs=-1
    )
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr

    print("Training ComplementNB...")
    nb = ComplementNB()
    nb.fit(X_train, y_train)
    models['ComplementNB'] = nb

    return models


def evaluate_models(models, X_test, y_test, label_encoder):
    results = {}
    print("\n--- Model evaluation ---")
    header = f"{'Model':20s}  {'Accuracy':>10s}  {'Macro F1':>10s}"
    print(header)
    print('-' * len(header))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {'accuracy': acc, 'macro_f1': f1, 'model': model, 'y_pred': y_pred}
        print(f"{name:20s}  {acc:10.4f}  {f1:10.4f}")

    return results


def select_best_model(results):
    best_name = max(results, key=lambda k: results[k]['macro_f1'])
    print(f"\nBest model: {best_name} (macro F1 = {results[best_name]['macro_f1']:.4f})")
    return best_name, results[best_name]['model']


def plot_confusion_matrix(y_test, y_pred, label_encoder):
    labels = label_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title('Confusion Matrix (normalised) — Best Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = PLOTS_DIR / '05_confusion_matrix.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_model_comparison(results):
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    f1s = [results[n]['macro_f1'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label='Accuracy', color='#7C3AED', alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s, width, label='Macro F1', color='#A78BFA', alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Accuracy and Macro F1', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.bar_label(bars1, fmt='%.4f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.4f', padding=3, fontsize=9)
    plt.tight_layout()
    path = PLOTS_DIR / '06_model_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_language_f1(y_test, y_pred, label_encoder):
    report = classification_report(y_test, y_pred,
                                   target_names=label_encoder.classes_,
                                   output_dict=True)
    langs = label_encoder.classes_
    f1s = [report[lang]['f1-score'] for lang in langs]
    order = np.argsort(f1s)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#EF4444' if f < 0.9 else '#7C3AED' for f in [f1s[i] for i in order]]
    ax.barh([langs[i] for i in order], [f1s[i] for i in order], color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score per Language — Best Model', fontsize=13, fontweight='bold')
    ax.axvline(x=0.9, color='gray', linestyle='--', linewidth=0.8, label='0.90 threshold')
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / '07_per_language_f1.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_ngrams(vectorizer, model, label_encoder):
    # Works for models with coef_ (LinearSVC base estimator or LogisticRegression)
    try:
        # If model is CalibratedClassifierCV, dig into the estimator
        if hasattr(model, 'calibrated_classifiers_'):
            base = model.calibrated_classifiers_[0].estimator
        else:
            base = model

        if not hasattr(base, 'coef_'):
            print("Skipping n-gram plot — model has no coef_ attribute")
            return

        coef = base.coef_
        feature_names = np.array(vectorizer.get_feature_names_out())
        labels = label_encoder.classes_

        # Pick 6 most common languages to keep the plot readable
        featured = ['English', 'Spanish', 'French', 'German', 'Arabic', 'Russian']
        featured = [l for l in featured if l in labels]
        indices = [np.where(labels == l)[0][0] for l in featured]

        top_n = 10
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for ax, lang, idx in zip(axes, featured, indices):
            top_idx = np.argsort(coef[idx])[-top_n:][::-1]
            top_ngrams = feature_names[top_idx]
            top_weights = coef[idx][top_idx]
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
            ax.barh(top_ngrams[::-1], top_weights[::-1], color=colors[::-1])
            ax.set_title(f'{lang}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Coefficient weight')

        plt.suptitle('Top 10 Most Distinctive Character N-grams per Language',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = PLOTS_DIR / '04_top_ngrams_per_language.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")
    except Exception as e:
        print(f"Could not generate n-gram plot: {e}")


def save_artifacts(vectorizer, best_model, best_name, label_encoder, results, df):
    pickle.dump(best_model, open(MODEL_DIR / 'best_model.pkl', 'wb'))
    pickle.dump(vectorizer, open(MODEL_DIR / 'vectorizer.pkl', 'wb'))
    pickle.dump(label_encoder, open(MODEL_DIR / 'label_encoder.pkl', 'wb'))

    (MODEL_DIR / 'best_model_name.txt').write_text(best_name)

    languages = sorted(df['Language'].unique().tolist())
    (MODEL_DIR / 'language_list.json').write_text(json.dumps(languages, indent=2))

    # Per-language metrics from best model
    best_result = results[best_name]
    from sklearn.metrics import classification_report as cr
    report = cr(best_result['y_test'], best_result['y_pred'],
                target_names=label_encoder.classes_, output_dict=True)

    metrics = {
        'model_name': best_name,
        'accuracy': round(best_result['accuracy'], 4),
        'macro_f1': round(best_result['macro_f1'], 4),
        'per_language': {
            lang: {
                'precision': round(report[lang]['precision'], 4),
                'recall': round(report[lang]['recall'], 4),
                'f1': round(report[lang]['f1-score'], 4),
            }
            for lang in label_encoder.classes_
        },
        'all_models': {
            name: {
                'accuracy': round(r['accuracy'], 4),
                'macro_f1': round(r['macro_f1'], 4),
            }
            for name, r in results.items()
        }
    }
    (MODEL_DIR / 'model_metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved models to {MODEL_DIR}/")
    for f in MODEL_DIR.iterdir():
        print(f"  {f.name}")


def print_summary(results, best_name, label_encoder, vectorizer, best_model):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\nAll models:")
    for name, r in results.items():
        marker = " <-- WINNER" if name == best_name else ""
        print(f"  {name:20s}  acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}{marker}")

    # Easiest / hardest language
    best_result = results[best_name]
    from sklearn.metrics import classification_report as cr
    report = cr(best_result['y_test'], best_result['y_pred'],
                target_names=label_encoder.classes_, output_dict=True)
    lang_f1 = {l: report[l]['f1-score'] for l in label_encoder.classes_}
    easiest = max(lang_f1, key=lang_f1.get)
    hardest = min(lang_f1, key=lang_f1.get)
    print(f"\nEasiest to detect : {easiest} (F1 = {lang_f1[easiest]:.4f})")
    print(f"Hardest to detect : {hardest} (F1 = {lang_f1[hardest]:.4f})")

    # Top n-grams per language (3 most common)
    try:
        if hasattr(best_model, 'calibrated_classifiers_'):
            base = best_model.calibrated_classifiers_[0].estimator
        else:
            base = best_model

        if hasattr(base, 'coef_'):
            feature_names = np.array(vectorizer.get_feature_names_out())
            labels = label_encoder.classes_
            print("\nMost distinctive n-grams per language (top 3):")
            for i, lang in enumerate(labels):
                top_idx = np.argsort(base.coef_[i])[-3:][::-1]
                top_ngrams = feature_names[top_idx]
                print(f"  {lang:15s}: {', '.join(repr(n) for n in top_ngrams)}")
    except Exception:
        pass

    print(f"\nModel files in {MODEL_DIR}/")


def main():
    df_raw = load_data()
    df = clean_data(df_raw)
    print_class_distribution(df)
    print_sample_texts(df)

    plot_language_distribution(df)
    plot_text_length(df)

    print("\nVectorizing...")
    vectorizer = TfidfVectorizer(
        analyzer=ANALYZER,
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Language'])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['Text'], y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    models = train_models(X_train, y_train, label_encoder)
    results = evaluate_models(models, X_test, y_test, label_encoder)

    # Attach y_test to results for later use
    for name in results:
        results[name]['y_test'] = y_test

    best_name, best_model = select_best_model(results)

    best_y_pred = results[best_name]['y_pred']
    plot_confusion_matrix(y_test, best_y_pred, label_encoder)
    plot_model_comparison(results)
    plot_per_language_f1(y_test, best_y_pred, label_encoder)
    plot_top_ngrams(vectorizer, best_model, label_encoder)

    save_artifacts(vectorizer, best_model, best_name, label_encoder, results, df)
    print_summary(results, best_name, label_encoder, vectorizer, best_model)


if __name__ == '__main__':
    main()
