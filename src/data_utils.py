from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Column names from the LIAR README. The original files do not include headers.
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]

FAKE = {"false", "barely-true", "pants-fire"}
REAL = {"true", "mostly-true", "half-true"}
LABEL_NAMES = {0: "fake", 1: "real"}


def to_binary(label):
    if label in FAKE:
        return 0
    if label in REAL:
        return 1
    raise ValueError(f"Unexpected label: {label}")


def read_split(project_dir, split_name, binary=True):
    file_path = Path(project_dir) / f"{split_name}.tsv"
    data = pd.read_csv(file_path, sep="\t", header=None, names=LIAR_COLUMNS)
    data["statement"] = data["statement"].fillna("").astype(str)

    if binary:
        data["label_binary"] = data["label"].apply(to_binary)

    return data


def load_liar(project_dir, binary=True):
    train = read_split(project_dir, "train", binary=binary)
    valid = read_split(project_dir, "valid", binary=binary)
    test = read_split(project_dir, "test", binary=binary)
    return train, valid, test


def get_xy(data, binary=True):
    y_column = "label_binary" if binary else "label"
    return data["statement"], data[y_column]


def get_scores(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
    }
