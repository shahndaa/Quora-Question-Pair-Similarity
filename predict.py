#!/usr/bin/env python3
"""
Predict whether two questions are duplicates, using the saved models from
train.py.

Usage:
    python predict.py "How do I learn Python?" "What is the best way to learn Python?"
    python predict.py "..." "..." --model deep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from preprocessing import clean_for_classical, clean_for_embeddings  # noqa: E402
from features import build_statistical_and_fuzzy_features  # noqa: E402
from embeddings import load_glove, sentence_to_avg_vector, cosine  # noqa: E402
from models_classical import load_artifact  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Predict duplicate probability for a question pair")
    p.add_argument("question1", type=str)
    p.add_argument("question2", type=str)
    p.add_argument("--model", choices=["classical", "deep"], default="classical")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--glove", type=str, default="glove-wiki-gigaword-100")
    p.add_argument("--glove-dim", type=int, default=100)
    p.add_argument("--max-len", type=int, default=30)
    return p.parse_args()


def predict_classical(q1: str, q2: str, model_dir: Path, glove_name: str, glove_dim: int) -> float:
    model = load_artifact(model_dir / "classical_best_model.joblib")
    scaler = load_artifact(model_dir / "scaler.joblib")
    tfidf_feat = load_artifact(model_dir / "tfidf_featurizer.joblib")

    q1_clean, q2_clean = clean_for_classical(q1), clean_for_classical(q2)
    row_df = pd.DataFrame({"question1_clean": [q1_clean], "question2_clean": [q2_clean]})

    stat_fuzzy = build_statistical_and_fuzzy_features(row_df, "question1_clean", "question2_clean")
    tfidf_sim = tfidf_feat.transform(row_df["question1_clean"], row_df["question2_clean"])

    kv = load_glove(glove_name)
    v1 = sentence_to_avg_vector(clean_for_embeddings(q1), kv, glove_dim)
    v2 = sentence_to_avg_vector(clean_for_embeddings(q2), kv, glove_dim)
    emb_sim = cosine(v1, v2)

    features = stat_fuzzy.copy()
    features["tfidf_cosine_sim"] = tfidf_sim.flatten()
    features["embedding_cosine_sim"] = emb_sim

    X = scaler.transform(features)
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


def predict_deep(q1: str, q2: str, model_dir: Path, max_len: int) -> float:
    import tensorflow as tf
    from models_deep import texts_to_padded

    model = tf.keras.models.load_model(model_dir / "deep_model.keras")
    tokenizer = load_artifact(model_dir / "tokenizer.joblib")

    q1_pad = texts_to_padded(tokenizer, [clean_for_embeddings(q1)], max_len)
    q2_pad = texts_to_padded(tokenizer, [clean_for_embeddings(q2)], max_len)

    proba = model.predict({"question1": q1_pad, "question2": q2_pad}, verbose=0)[0, 0]
    return float(proba)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    if args.model == "classical":
        proba = predict_classical(args.question1, args.question2, model_dir, args.glove, args.glove_dim)
    else:
        proba = predict_deep(args.question1, args.question2, model_dir, args.max_len)

    label = "DUPLICATE" if proba >= 0.5 else "NOT duplicate"
    print(f"\nQ1: {args.question1}")
    print(f"Q2: {args.question2}")
    print(f"\nPrediction ({args.model} model): {label}  (probability = {proba:.3f})")


if __name__ == "__main__":
    main()
