import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data_utils import load_liar


class LiarDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.tokens = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(values[index]) for key, values in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    project_dir = Path(args.project_dir).resolve()
    train, valid, test = load_liar(project_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "fake", 1: "real"},
        label2id={"fake": 0, "real": 1},
    )

    train_data = LiarDataset(train["statement"], train["label_binary"], tokenizer, args.max_length)
    valid_data = LiarDataset(valid["statement"], valid["label_binary"], tokenizer, args.max_length)
    test_data = LiarDataset(test["statement"], test["label_binary"], tokenizer, args.max_length)

    model_dir = project_dir / "models" / args.model_name.replace("/", "_")
    train_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    valid_scores = trainer.evaluate(valid_data, metric_key_prefix="valid")
    test_scores = trainer.evaluate(test_data, metric_key_prefix="test")

    rows = []
    for split, scores in [("valid", valid_scores), ("test", test_scores)]:
        rows.append({
            "Model": args.model_name,
            "Split": split,
            "Accuracy": scores[f"{split}_accuracy"],
            "Macro Precision": scores[f"{split}_macro_precision"],
            "Macro Recall": scores[f"{split}_macro_recall"],
            "Macro F1": scores[f"{split}_macro_f1"],
        })

    results = pd.DataFrame(rows)
    results_path = project_dir / "outputs" / "results_transformer.csv"
    results_path.parent.mkdir(exist_ok=True)
    results.to_csv(results_path, index=False)

    trainer.save_model(model_dir / "best_model")
    tokenizer.save_pretrained(model_dir / "best_model")

    print(results.to_string(index=False))
    print(f"Model saved in: {model_dir / 'best_model'}")


if __name__ == "__main__":
    main()
