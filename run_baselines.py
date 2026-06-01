import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_utils import LABEL_NAMES, get_scores, get_xy, load_liar


def make_models():
    logreg = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    svm = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )),
        ("clf", LinearSVC(class_weight="balanced", random_state=42)),
    ])

    return {
        "TF-IDF + Logistic Regression": logreg,
        "TF-IDF + Linear SVM": svm,
    }


def save_matrix(name, y_true, y_pred, out_dir):
    file_name = name.lower().replace(" + ", "_").replace(" ", "_").replace("-", "")
    fig = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        cmap="Blues",
        colorbar=False,
    )
    fig.ax_.set_title(name)
    fig.figure_.tight_layout()
    fig.figure_.savefig(out_dir / f"confusion_matrix_{file_name}.png", dpi=180)
    plt.close(fig.figure_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    out_dir = project_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    train, valid, test = load_liar(project_dir)
    x_train, y_train = get_xy(train)
    x_valid, y_valid = get_xy(valid)
    x_test, y_test = get_xy(test)

    rows = []
    reports = []

    for model_name, model in make_models().items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)

        valid_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)

        rows.append({"Model": model_name, "Split": "valid", **get_scores(y_valid, valid_pred)})
        rows.append({"Model": model_name, "Split": "test", **get_scores(y_test, test_pred)})

        reports.append(f"\n## {model_name}\n")
        reports.append("Validation\n")
        reports.append(classification_report(y_valid, valid_pred, target_names=["fake", "real"], zero_division=0))
        reports.append("\nTest\n")
        reports.append(classification_report(y_test, test_pred, target_names=["fake", "real"], zero_division=0))

        save_matrix(model_name, y_test, test_pred, out_dir)

    results = pd.DataFrame(rows)
    results.to_csv(project_dir / "results_baselines.csv", index=False)
    (out_dir / "baseline_classification_reports.md").write_text("\n".join(reports), encoding="utf-8")

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
