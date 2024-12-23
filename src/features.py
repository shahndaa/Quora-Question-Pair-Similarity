"""
Feature engineering for the Quora Question Pair Similarity project.

Produces a rich feature matrix combining:
  - Simple statistical features (lengths, word counts, common words)
  - Fuzzy string-matching features (rapidfuzz)
  - TF-IDF cosine similarity
  - Averaged word-embedding cosine similarity (semantic signal)

These are exactly the kind of hand-crafted features that performed well in
the real Kaggle competition (see project README for references), used here
on top of a single TF-IDF cosine value like the old version of this repo did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _safe_div(a, b):
    return a / b if b else 0.0


def statistical_features(q1: str, q2: str) -> dict:
    """Cheap, fast features based on lengths and token overlap."""
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    len_q1, len_q2 = len(q1), len(q2)
    n_words_q1, n_words_q2 = len(q1_tokens), len(q2_tokens)

    common_tokens = set(q1_tokens) & set(q2_tokens)
    total_unique_tokens = set(q1_tokens) | set(q2_tokens)

    return {
        "len_q1": len_q1,
        "len_q2": len_q2,
        "len_diff": abs(len_q1 - len_q2),
        "n_words_q1": n_words_q1,
        "n_words_q2": n_words_q2,
        "n_words_diff": abs(n_words_q1 - n_words_q2),
        "common_word_count": len(common_tokens),
        "common_word_ratio_min": _safe_div(len(common_tokens), min(n_words_q1, n_words_q2) or 1),
        "common_word_ratio_max": _safe_div(len(common_tokens), max(n_words_q1, n_words_q2) or 1),
        "jaccard_similarity": _safe_div(len(common_tokens), len(total_unique_tokens) or 1),
        "first_word_same": int(bool(q1_tokens) and bool(q2_tokens) and q1_tokens[0] == q2_tokens[0]),
        "last_word_same": int(bool(q1_tokens) and bool(q2_tokens) and q1_tokens[-1] == q2_tokens[-1]),
    }


def fuzzy_features(q1: str, q2: str) -> dict:
    """Fuzzy string-matching ratios (rapidfuzz, a fast C++ implementation
    of the fuzzywuzzy API) that catch near-duplicate phrasing."""
    return {
        "fuzz_ratio": fuzz.ratio(q1, q2),
        "fuzz_partial_ratio": fuzz.partial_ratio(q1, q2),
        "fuzz_token_sort_ratio": fuzz.token_sort_ratio(q1, q2),
        "fuzz_token_set_ratio": fuzz.token_set_ratio(q1, q2),
    }


def build_statistical_and_fuzzy_features(df: pd.DataFrame, q1_col: str, q2_col: str) -> pd.DataFrame:
    """Vectorized-ish construction of the row-wise features above."""
    rows = []
    for q1, q2 in zip(df[q1_col], df[q2_col]):
        row = {}
        row.update(statistical_features(q1, q2))
        row.update(fuzzy_features(q1, q2))
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


class TfidfSimilarityFeaturizer:
    """Fits a single shared TF-IDF vocabulary across both question columns,
    then produces a cosine-similarity feature per row. Fitting on both
    columns together (not just q1) avoids a vocabulary mismatch between
    q1/q2 at transform time."""

    def __init__(self, ngram_range=(1, 2), max_features=50_000):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self._fitted = False

    def fit(self, q1_series: pd.Series, q2_series: pd.Series):
        combined = pd.concat([q1_series, q2_series], ignore_index=True)
        self.vectorizer.fit(combined)
        self._fitted = True
        return self

    def transform(self, q1_series: pd.Series, q2_series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        tfidf_q1 = self.vectorizer.transform(q1_series)
        tfidf_q2 = self.vectorizer.transform(q2_series)
        # Row-wise cosine similarity between q1_i and q2_i (not a full matrix)
        numer = tfidf_q1.multiply(tfidf_q2).sum(axis=1)
        norm_q1 = np.sqrt(tfidf_q1.multiply(tfidf_q1).sum(axis=1))
        norm_q2 = np.sqrt(tfidf_q2.multiply(tfidf_q2).sum(axis=1))
        denom = np.multiply(norm_q1, norm_q2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.asarray(numer).flatten() / np.asarray(denom).flatten()
        sim = np.nan_to_num(sim, nan=0.0)
        return sim.reshape(-1, 1)


def build_feature_matrix(
    df: pd.DataFrame,
    tfidf_featurizer: TfidfSimilarityFeaturizer,
    embedding_sim: np.ndarray | None = None,
    q1_clean_col: str = "question1_clean",
    q2_clean_col: str = "question2_clean",
) -> pd.DataFrame:
    """Assemble the full feature matrix used by the classical models."""
    stat_fuzzy = build_statistical_and_fuzzy_features(df, q1_clean_col, q2_clean_col)
    tfidf_sim = tfidf_featurizer.transform(df[q1_clean_col], df[q2_clean_col])

    features = stat_fuzzy.copy()
    features["tfidf_cosine_sim"] = tfidf_sim.flatten()

    if embedding_sim is not None:
        features["embedding_cosine_sim"] = embedding_sim.flatten()

    return features
