# Fake news classification with LIAR

Code for my I2R project. I used the LIAR dataset to compare two simple baselines with a fine-tuned DistilBERT model. At the end I also used LIME to inspect a few predictions.

The original LIAR labels were grouped like this:

- fake: `pants-fire`, `false`, `barely-true`
- real: `half-true`, `mostly-true`, `true`

## Files

- `implementation_jupyter.ipynb`: notebook I used to run the work step by step.
- `src/data_utils.py`: loading the LIAR files and converting the labels.
- `run_baselines.py`: TF-IDF with Logistic Regression and Linear SVM.
- `train_transformer.py`: fine-tuning for `distilbert-base-uncased`.
- `explain_lime.py`: LIME explanations for a few test examples.
- `outputs/`: result files used in the report.
- `figures/`: confusion matrices.

## Dataset

The three LIAR files have to be in the same folder as the scripts:

```text
train.tsv
valid.tsv
test.tsv
```

I kept the original train/validation/test split.

## Running it

Install dependencies:

```bash
pip install -r requirements.txt
```

Baselines:

```bash
python run_baselines.py --project-dir .
```

DistilBERT:

```bash
python train_transformer.py --project-dir . --model-name distilbert-base-uncased --epochs 3 --batch-size 8 --learning-rate 2e-5
```

LIME explanations:

```bash
python explain_lime.py --project-dir . --model-dir ./models/distilbert-base-uncased/best_model --num-samples 3
```

## Results used in the report

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.6038 | 0.5985 |
| TF-IDF + Linear SVM | 0.6101 | 0.6045 |
| DistilBERT | 0.6480 | 0.6345 |

I did not include the trained checkpoints in this folder because they are large and can be generated again with the training script.
