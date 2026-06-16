from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "doc_type_clf.joblib"


def build_pipeline(use_svm: bool = True):
    clf = LinearSVC() if use_svm else LogisticRegression(max_iter=3000)

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True
        )),
        ("clf", clf),
    ])


def train_doc_classifier(
    texts,
    labels,
    groups,
    model_path: Path = DEFAULT_MODEL_PATH,
    use_svm: bool = True,
    test_size: float = 0.2,
):
    pipe = build_pipeline(use_svm=use_svm)

    print(f"[INFO] Örnek sayısı: {len(texts)}")
    print(f"[INFO] Sınıf sayısı: {len(set(labels))}")
    print(f"[INFO] PDF grup sayısı: {len(set(groups))}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(texts, labels, groups=groups))

    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    train_files = {groups[i] for i in train_idx}
    test_files = {groups[i] for i in test_idx}

    leakage = train_files.intersection(test_files)
    if leakage:
        raise RuntimeError(f"Veri sızıntısı var! Ortak PDF'ler: {leakage}")

    print(f"\n[INFO] Train chunk: {len(X_train)} | Test chunk: {len(X_test)}")
    print(f"[INFO] Train PDF: {len(train_files)} | Test PDF: {len(test_files)}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))

    final_model = build_pipeline(use_svm=use_svm)
    final_model.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    print(f"\n[OK] Final model tüm eğitim verisiyle eğitildi.")
    print(f"[OK] Model kaydedildi: {model_path}")

    return final_model


def load_doc_classifier(model_path: Path = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict_doc_type(text: str, model_path: Path = DEFAULT_MODEL_PATH):
    text = text or ""

    if len(text.strip()) < 300:
        return "image_only_or_empty"

    model = load_doc_classifier(model_path)
    return model.predict([text])[0]