# app/document_classifier.py

from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Proje kökünü ve model path'ini otomatik bul
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "doc_type_clf.joblib"


def train_doc_classifier(texts, labels, model_path: Path = DEFAULT_MODEL_PATH, use_svm: bool = True):
    """
    texts: her eleman 1 belgenin full metni
    labels: her belge için tür label'ı (örn: 'tournament_report')
    """

    if use_svm:
        clf = LinearSVC()
    else:
        clf = LogisticRegression(max_iter=2000)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )),
        ("clf", clf),
    ])

    n_samples = len(texts)
    n_classes = len(set(labels))

    # Dataset çok küçükse train/test split yapmıyoruz
    if n_samples < 3 or n_classes < 2:
        print(f"[Uyarı] Dataset çok küçük (n_samples={n_samples}, n_classes={n_classes}).")
        print("Train/test split YAPMADAN tüm veriyi kullanarak modeli eğitiyorum.\n")

        pipe.fit(texts, labels)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
        print(f"[OK] Model kaydedildi: {model_path}")
        return

    # Normal durumda train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"[OK] Model kaydedildi: {model_path}")


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict_doc_type(text: str, model=None, model_path: Path = DEFAULT_MODEL_PATH) -> str:
    """
    Tek bir belgenin metnini alıp tür label'ını döndürür.
    """
    if model is None:
        model = load_model(model_path)
    return model.predict([text])[0]
