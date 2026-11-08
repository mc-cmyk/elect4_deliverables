"""ReviewLens - Part 1: grid search for the sentiment classifier.

Evaluates 2 vectorizers x 4 n-gram ranges x 3 models = 24 configurations with
5-fold stratified cross-validation, plus an AFINN lexicon baseline (25 rows).
The vectorizer lives inside the Pipeline and is cloned fresh for every fold, so
vocabulary and IDF weights are learned from the training fold only - the
validation fold never leaks into feature construction.

Outputs (written to output/):
  cv_results.csv        all 25 configurations, sorted by macro-F1
  predictions_test.csv  held-out test set with the best model's predictions
  best_model.joblib     best pipeline refitted on the full training set
  best_config.json      chosen configuration and its test-set scores

Run:  python train_grid.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from afinn import Afinn
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from text_prep import normalize, tokenize

OUT = Path(__file__).parent / "output"
LABELS = ["Negative", "Neutral", "Positive"]
SEED = 42
N_FOLDS = 5

VECTORIZERS = {"CountVectorizer": CountVectorizer, "TfidfVectorizer": TfidfVectorizer}
NGRAMS = [(1, 1), (2, 2), (3, 3), (1, 3)]
MODELS = {
    "MultinomialNB": lambda: MultinomialNB(),
    "LogisticRegression": lambda: LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=SEED),
    "LinearSVC": lambda: LinearSVC(class_weight="balanced", random_state=SEED),
}
AFINN_NOTE = ("AFINN is an English-only lexicon; Tagalog/Bisaya/Taglish words "
              "are unknown to it, so it under-performs on this dataset.")


def build_pipeline(vec_name, ngram, model_name):
    vectorizer = VECTORIZERS[vec_name](
        preprocessor=normalize,   # lowercase, strip URLs, collapse repeats
        tokenizer=tokenize,       # emoji-aware, stopwords out, negations kept
        token_pattern=None,
        lowercase=False,          # normalize() already lowercases
        ngram_range=ngram,
    )
    return Pipeline([("vec", vectorizer), ("clf", MODELS[model_name]())])


def fold_scores(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{lab.lower()}": s for lab, s in zip(LABELS, per_class)},
    }


def summarize(name, fold_rows, fit_seconds, **meta):
    df = pd.DataFrame(fold_rows)
    row = {"config": name, **meta}
    for col in df.columns:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std(ddof=1)
    row["fit_seconds"] = round(fit_seconds, 2)
    return row


class AfinnClassifier:
    """Sum of AFINN word scores: >0 Positive, <0 Negative, 0 Neutral."""

    def __init__(self):
        self.afinn = Afinn(language="en", emoticons=False)

    def predict(self, texts):
        return np.array([self._label(self.afinn.score(normalize(t))) for t in texts])

    def n_without_known_words(self, texts):
        return sum(1 for t in texts if not self.afinn.find_all(normalize(t)))

    @staticmethod
    def _label(score):
        return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def run_grid(X, y, folds):
    rows = []
    total = len(VECTORIZERS) * len(NGRAMS) * len(MODELS) + 1  # +1 for AFINN
    i = 0
    for vec_name in VECTORIZERS:
        for ngram in NGRAMS:
            for model_name in MODELS:
                i += 1
                name = f"{vec_name} {ngram} + {model_name}"
                template = build_pipeline(vec_name, ngram, model_name)
                start, fold_rows = time.perf_counter(), []
                for train_idx, val_idx in folds:
                    pipe = clone(template)  # fresh, unfitted vectorizer per fold
                    pipe.fit(X[train_idx], y[train_idx])
                    fold_rows.append(fold_scores(y[val_idx], pipe.predict(X[val_idx])))
                row = summarize(name, fold_rows, time.perf_counter() - start,
                                vectorizer=vec_name, ngram_range=str(ngram),
                                model=model_name, note="")
                rows.append(row)
                print(f"  [{i:2d}/{total}] {name:<52} macro-F1 "
                      f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    return rows


def run_afinn(X, y, folds):
    afinn = AfinnClassifier()
    start, fold_rows = time.perf_counter(), []
    for _, val_idx in folds:  # nothing to fit; score the same validation folds
        fold_rows.append(fold_scores(y[val_idx], afinn.predict(X[val_idx])))
    row = summarize("AFINN lexicon (baseline)", fold_rows, time.perf_counter() - start,
                    vectorizer="AFINN lexicon", ngram_range="(1, 1)",
                    model="AFINN word-score sum", note=AFINN_NOTE)
    preds = afinn.predict(X)
    zero_unknown = afinn.n_without_known_words(X)
    zero_total = int(sum(afinn.afinn.score(normalize(t)) == 0 for t in X))
    print(f"  [25/25] {'AFINN lexicon (baseline)':<52} macro-F1 "
          f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    return row, {"zero_score_total": zero_total,
                 "zero_no_known_words": zero_unknown,
                 "n": len(X),
                 "pred_neutral": int((preds == "Neutral").sum())}


def main():
    train = pd.read_csv(OUT / "jollibee_train.csv", encoding="utf-8-sig")
    test = pd.read_csv(OUT / "jollibee_test.csv", encoding="utf-8-sig")
    X, y = train["text_clean"].astype(str).to_numpy(), train["sentiment"].to_numpy()
    X_test, y_test = test["text_clean"].astype(str).to_numpy(), test["sentiment"].to_numpy()

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    print(f"ReviewLens grid search - {len(X):,} training reviews, {N_FOLDS}-fold CV\n")
    rows = run_grid(X, y, folds)
    afinn_row, afinn_stats = run_afinn(X, y, folds)
    rows.append(afinn_row)

    cv = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    cv.insert(0, "rank", range(1, len(cv) + 1))
    ml = cv[cv["vectorizer"] != "AFINN lexicon"]
    best = ml.iloc[0]  # the lexicon is a reference point, not a candidate
    cv["is_best"] = cv["config"] == best["config"]
    cv.to_csv(OUT / "cv_results.csv", index=False)

    # Refit the winner on all training data, then score the test set exactly once.
    ngram = tuple(int(n) for n in best["ngram_range"].strip("()").split(","))
    model = build_pipeline(best["vectorizer"], ngram, best["model"]).fit(X, y)
    pred = model.predict(X_test)
    joblib.dump(model, OUT / "best_model.joblib")

    pred_df = test[["uuid", "Branch", "City", "review_date", "rating", "sentiment",
                    "text_raw", "text_clean", "word_count", "is_short",
                    "rider_blamed", "branch_usable"]].copy()
    pred_df["predicted"] = pred
    pred_df["correct"] = pred_df["predicted"] == pred_df["sentiment"]
    pred_df.to_csv(OUT / "predictions_test.csv", index=False)

    test_acc = accuracy_score(y_test, pred)
    test_f1 = f1_score(y_test, pred, labels=LABELS, average="macro")
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    majority_label = pd.Series(y).value_counts().idxmax()
    majority_acc = float((y_test == majority_label).mean())
    short_mask = test["is_short"].astype(bool).to_numpy()
    short_acc = float((pred[short_mask] == y_test[short_mask]).mean())
    long_acc = float((pred[~short_mask] == y_test[~short_mask]).mean())

    config = {
        "config": best["config"],
        "vectorizer": best["vectorizer"],
        "ngram_range": list(ngram),
        "model": best["model"],
        "cv_macro_f1_mean": float(best["macro_f1_mean"]),
        "cv_macro_f1_std": float(best["macro_f1_std"]),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_report": classification_report(y_test, pred, labels=LABELS, output_dict=True),
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "majority_baseline": {"label": majority_label, "accuracy": majority_acc},
        "short_vs_long_accuracy": {
            "short": {"n": int(short_mask.sum()), "accuracy": short_acc},
            "long": {"n": int((~short_mask).sum()), "accuracy": long_acc},
        },
        "afinn": {**afinn_stats, "note": AFINN_NOTE},
        "n_train": len(X), "n_test": len(X_test), "cv_folds": N_FOLDS, "seed": SEED,
    }
    (OUT / "best_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- summary
    line = "=" * 92
    print(f"\n{line}\nCROSS-VALIDATION RESULTS ({N_FOLDS}-fold, sorted by macro-F1)\n{line}")
    table = cv[["rank", "config", "macro_f1_mean", "macro_f1_std", "accuracy_mean",
                "f1_negative_mean", "f1_neutral_mean", "f1_positive_mean"]].copy()
    table["macro-F1"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(table.macro_f1_mean, table.macro_f1_std)]
    table = table.rename(columns={"accuracy_mean": "acc", "f1_negative_mean": "F1 neg",
                                  "f1_neutral_mean": "F1 neu", "f1_positive_mean": "F1 pos"})
    table["config"] = [c + ("  <-- best" if c == best["config"] else "") for c in table["config"]]
    print(table[["rank", "config", "macro-F1", "acc", "F1 neg", "F1 neu", "F1 pos"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nAFINN baseline: {afinn_stats['zero_score_total']:,} of {afinn_stats['n']:,} "
          f"training reviews scored 0 (-> Neutral); {afinn_stats['zero_no_known_words']:,} of them "
          f"contain no AFINN word at all, mostly Tagalog/Bisaya/Taglish text.\n  Note: {AFINN_NOTE}")

    print(f"\n{line}\nBEST MODEL ON HELD-OUT TEST SET ({len(X_test):,} reviews)\n{line}")
    print(f"Config: {best['config']}")
    print(f"CV macro-F1: {best['macro_f1_mean']:.3f} ± {best['macro_f1_std']:.3f}   "
          f"Test macro-F1: {test_f1:.3f}   Test accuracy: {test_acc:.1%}\n")
    print(classification_report(y_test, pred, labels=LABELS, digits=3))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {l}" for l in LABELS],
                       columns=[f"pred {l}" for l in LABELS]).to_string())
    print(f"\nMajority-class baseline (always '{majority_label}'): {majority_acc:.1%} accuracy")
    print(f"Short reviews (<3 words): {short_acc:.1%} accuracy on {short_mask.sum():,} reviews")
    print(f"Longer reviews:           {long_acc:.1%} accuracy on {(~short_mask).sum():,} reviews")
    print("Neutral (3-star) has the weakest F1: ratings are U-shaped and 3-star text "
          "reads as mildly positive or negative.")
    print(f"\nSaved to {OUT}: cv_results.csv, predictions_test.csv, best_model.joblib, best_config.json")


if __name__ == "__main__":
    main()
