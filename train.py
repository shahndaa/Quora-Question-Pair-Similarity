#!/usr/bin/env python3
"""
Train the Quora Question Pair Similarity models.

Usage:
    python train.py --data data/train.csv --model classical
    python train.py --data data/train.csv --model deep
    python train.py --data data/train.csv --model both
    python train.py --data data/train.csv --model both --sample-frac 0.2

Outputs (written to models/):
    - tfidf_vectorizer.joblib      (fitted TF-IDF vectorizer used for features)
    - scaler.joblib                (StandardScaler for classical model features)
    - classical_best_model.joblib  (best classical model by F1 score)
    - classical_results.json       (metrics for every classical model tried)
    - tokenizer.joblib             (Keras Tokenizer used by the deep model)
    - deep_model.keras             (trained Siamese LSTM, if --model deep/both)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from preprocessing import preprocess_dataframe  # noqa: E402
from features import TfidfSimilarityFeaturizer, build_feature_matrix  # noqa: E402
from embeddings import load_glove, build_embedding_similarity_feature, build_embedding_matrix  # noqa: E402
from models_classical import train_and_evaluate_all, save_artifact  # noqa: E402
# NOTE: models_deep (and therefore tensorflow) is imported lazily, only when
# --model deep/both is actually requested. This lets `--model classical` run
# in environments where tensorflow isn't installed (e.g. Python versions
# TensorFlow doesn't yet publish wheels for).


def parse_args():
    p = argparse.ArgumentParser(description="Train Quora Question Pair Similarity models")
    p.add_argument("--data", type=str, default="data/train.csv", help="Path to the training CSV")
    p.add_argument("--model", type=str, choices=["classical", "deep", "both"], default="both")
    p.add_argument("--glove", type=str, default="glove-wiki-gigaword-100",
                    help="gensim pretrained embedding name (50/100/200/300-dim gigaword variants)")
    p.add_argument("--glove-dim", type=int, default=100)
    p.add_argument("--max-len", type=int, default=30, help="Max tokens per question (deep model)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--sample-frac", type=float, default=None,
                    help="Optionally train on a random fraction of the data (useful for quick iteration)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="models")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["question1", "question2", "is_duplicate"]).reset_index(drop=True)
    if args.sample_frac:
        df = df.sample(frac=args.sample_frac, random_state=args.random_state).reset_index(drop=True)
    print(f"    -> {len(df):,} rows, duplicate rate = {df['is_duplicate'].mean():.2%}")

    print("[2/5] Preprocessing text ...")
    t0 = time.time()
    df = preprocess_dataframe(df)
    print(f"    -> done in {time.time() - t0:.1f}s")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["is_duplicate"]
    )

    if args.model in ("classical", "both"):
        print("[3/5] Building classical features (statistical + fuzzy + TF-IDF + embedding cosine) ...")
        tfidf_feat = TfidfSimilarityFeaturizer()
        tfidf_feat.fit(train_df["question1_clean"], train_df["question2_clean"])

        kv = load_glove(args.glove)
        emb_sim_train = build_embedding_similarity_feature(train_df, kv, args.glove_dim)
        emb_sim_test = build_embedding_similarity_feature(test_df, kv, args.glove_dim)

        X_train = build_feature_matrix(train_df, tfidf_feat, emb_sim_train)
        X_test = build_feature_matrix(test_df, tfidf_feat, emb_sim_test)
        y_train = train_df["is_duplicate"].values
        y_test = test_df["is_duplicate"].values

        print("    -> training classical model zoo ...")
        results, scaler = train_and_evaluate_all(X_train, y_train, X_test, y_test, args.random_state)

        print("\n    Classical model results (sorted by F1):")
        results_summary = {}
        for r in results:
            print(f"      {r.name:<22} " + " | ".join(f"{k}={v:.4f}" for k, v in r.metrics.items()))
            results_summary[r.name] = r.metrics

        # Guard against saving an unreasonably large model file (e.g. a very
        # deep, unbounded Random Forest can serialize to gigabytes, which
        # breaks a normal `git push` to GitHub). Fall back to the next-best
        # model by F1 if the top one is too large.
        max_model_mb = 95
        best = None
        for candidate in results:
            tmp_path = out_dir / "_size_check.joblib"
            save_artifact(candidate.model, tmp_path)
            size_mb = tmp_path.stat().st_size / (1024 * 1024)
            tmp_path.unlink()
            if size_mb <= max_model_mb:
                best = candidate
                print(f"    -> selected '{candidate.name}' ({size_mb:.1f} MB, within {max_model_mb} MB limit)")
                break
            else:
                print(f"    -> skipping '{candidate.name}': {size_mb:.1f} MB exceeds {max_model_mb} MB limit")
        if best is None:
            best = results[0]
            print(f"    -> WARNING: all models exceed {max_model_mb} MB, keeping best anyway ('{best.name}')")

        save_artifact(best.model, out_dir / "classical_best_model.joblib")
        save_artifact(scaler, out_dir / "scaler.joblib")
        save_artifact(tfidf_feat, out_dir / "tfidf_featurizer.joblib")
        with open(out_dir / "classical_results.json", "w") as f:
            json.dump({"best_model": best.name, "all_results": results_summary}, f, indent=2)
        print(f"    -> saved best classical model ('{best.name}') and artifacts to {out_dir}/")

    if args.model in ("deep", "both"):
        from models_deep import build_tokenizer, texts_to_padded, build_siamese_lstm, train_siamese_lstm

        print("[4/5] Training Siamese LSTM ...")
        tokenizer = build_tokenizer(
            pd.concat([train_df["question1_emb"], train_df["question2_emb"]])
        )
        q1_train = texts_to_padded(tokenizer, train_df["question1_emb"], args.max_len)
        q2_train = texts_to_padded(tokenizer, train_df["question2_emb"], args.max_len)
        q1_test = texts_to_padded(tokenizer, test_df["question1_emb"], args.max_len)
        q2_test = texts_to_padded(tokenizer, test_df["question2_emb"], args.max_len)

        kv = load_glove(args.glove)
        embedding_matrix = build_embedding_matrix(tokenizer.word_index, kv, args.glove_dim)

        model = build_siamese_lstm(
            vocab_size=embedding_matrix.shape[0],
            embedding_dim=args.glove_dim,
            max_len=args.max_len,
            embedding_matrix=embedding_matrix,
        )
        y_train = train_df["is_duplicate"].values
        y_test = test_df["is_duplicate"].values
        train_siamese_lstm(
            model, q1_train, q2_train, y_train, q1_test, q2_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size,
        )

        print("    -> training finished, saving model immediately (before final evaluation) ...")
        model.save(out_dir / "deep_model.keras")
        save_artifact(tokenizer, out_dir / "tokenizer.joblib")
        print(f"    -> saved deep model + tokenizer to {out_dir}/")

        eval_results = model.evaluate(
            {"question1": q1_test, "question2": q2_test}, y_test, verbose=0, return_dict=True
        )
        print(f"    -> Siamese LSTM test metrics: {eval_results}")

        with open(out_dir / "deep_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

    print("[5/5] Done.")


if __name__ == "__main__":
    main()
