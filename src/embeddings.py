"""
Word-embedding utilities for the Quora Question Pair Similarity project.

We use pretrained GloVe vectors (via gensim's downloader, which caches the
model locally after first use) for two things:
  1. A fast "averaged embedding cosine similarity" feature that feeds the
     classical models -> captures semantic similarity that plain TF-IDF
     (word-overlap based) cannot see, e.g. "purchase" vs "buy".
  2. An embedding matrix used to initialize the Embedding layer of the
     Siamese LSTM deep-learning model (see model_deep.py).

Note: the first call to load_glove() downloads ~66MB (50d) or more for
larger dimensions. It is cached under ~/gensim-data after the first run.
"""
from __future__ import annotations

import numpy as np

_GLOVE_CACHE = {}


def load_glove(name: str = "glove-wiki-gigaword-100"):
    """Load (and cache in-process) a pretrained GloVe model via gensim.

    Common options: glove-wiki-gigaword-50/100/200/300.
    """
    if name not in _GLOVE_CACHE:
        import gensim.downloader as api
        _GLOVE_CACHE[name] = api.load(name)
    return _GLOVE_CACHE[name]


def sentence_to_avg_vector(text: str, kv, dim: int) -> np.ndarray:
    """Average the word vectors of every in-vocabulary token in `text`.
    Returns a zero vector if none of the tokens are known."""
    if not text:
        return np.zeros(dim, dtype=np.float32)
    vectors = [kv[tok] for tok in text.split() if tok in kv]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_embedding_similarity_feature(
    df, kv, dim: int, q1_col: str = "question1_emb", q2_col: str = "question2_emb"
) -> np.ndarray:
    """Row-wise cosine similarity between averaged GloVe vectors of q1/q2."""
    sims = np.zeros(len(df), dtype=np.float32)
    for i, (q1, q2) in enumerate(zip(df[q1_col], df[q2_col])):
        v1 = sentence_to_avg_vector(q1, kv, dim)
        v2 = sentence_to_avg_vector(q2, kv, dim)
        sims[i] = cosine(v1, v2)
    return sims


def build_embedding_matrix(word_index: dict, kv, dim: int) -> np.ndarray:
    """Build a Keras-Embedding-layer-ready matrix: row i = vector for the
    word whose tokenizer id is i. Words not found in GloVe get a zero row
    (Keras will still learn something for them via fine-tuning, if enabled).
    `word_index` is a {word: index} mapping, e.g. from keras Tokenizer.
    """
    vocab_size = len(word_index) + 1  # +1 for the padding/reserved index 0
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    n_found = 0
    for word, idx in word_index.items():
        if word in kv:
            matrix[idx] = kv[word]
            n_found += 1
    return matrix
