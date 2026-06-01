import argparse
from pathlib import Path

import pandas as pd
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.data_utils import load_liar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--lime-samples", type=int, default=1000)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    out_dir = project_dir / "outputs" / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test = load_liar(project_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=128,
    )

    def predict_proba(texts):
        raw_predictions = clf(list(texts))
        probs = []
        for prediction in raw_predictions:
            scores = {item["label"].lower(): item["score"] for item in prediction}
            fake_score = scores.get("fake", scores.get("label_0", 0.0))
            real_score = scores.get("real", scores.get("label_1", 0.0))
            probs.append([fake_score, real_score])
        return pd.DataFrame(probs).to_numpy()

    explainer = LimeTextExplainer(class_names=["fake", "real"])
    examples = test.head(args.num_samples)
    rows = []

    for test_index, row in examples.iterrows():
        explanation = explainer.explain_instance(
            row["statement"],
            predict_proba,
            num_features=10,
            num_samples=args.lime_samples,
        )

        html_file = out_dir / f"lime_test_{test_index}.html"
        explanation.save_to_file(str(html_file))

        rows.append({
            "test_index": test_index,
            "true_label": "real" if row["label_binary"] == 1 else "fake",
            "statement": row["statement"],
            "explanation_file": str(html_file),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "lime_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
