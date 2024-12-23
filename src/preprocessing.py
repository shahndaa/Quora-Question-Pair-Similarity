"""
Text preprocessing utilities for the Quora Question Pair Similarity project.

Two preprocessing "modes" are provided because different downstream models
want different things:
  - `clean_for_classical`: aggressive cleaning (lowercase, punctuation/number
    normalization, stopword removal, lemmatization) -> good for TF-IDF /
    bag-of-words style features.
  - `clean_for_embeddings`: lighter cleaning (lowercase, punctuation
    normalization only) -> keeps stopwords and word order, which matters for
    embedding-based / deep learning models.
"""
from __future__ import annotations

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if they're not already present."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_data()

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Common English contractions -> expanded form. Expanding these BEFORE
# stripping punctuation avoids turning "what's" into the meaningless "what s".
_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
}

_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
_MULTI_SPACE_RE = re.compile(r"\s+")
_MATH_RE = re.compile(r"\[math\].*?\[/math\]", flags=re.IGNORECASE | re.DOTALL)


def _expand_contractions(text: str) -> str:
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def _base_clean(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _MATH_RE.sub(" ", text)          # strip Quora's [math]...[/math] LaTeX blocks
    text = _expand_contractions(text)
    text = text.translate(_PUNCT_TABLE)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=200_000)
def clean_for_classical(text: str) -> str:
    """Lowercase, expand contractions, strip punctuation/digits, remove
    stopwords, and lemmatize. Intended for TF-IDF / bag-of-words features."""
    text = _base_clean(text)
    if not text:
        return ""
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in text.split()
        if tok not in _STOPWORDS and not tok.isdigit()
    ]
    return " ".join(tokens)


@lru_cache(maxsize=200_000)
def clean_for_embeddings(text: str) -> str:
    """Lighter cleaning that preserves word order and stopwords, which
    matters for sequence models (LSTM) and averaged word embeddings."""
    return _base_clean(text)


def preprocess_dataframe(df, q1_col: str = "question1", q2_col: str = "question2"):
    """Return a copy of `df` with four new columns: cleaned classical +
    cleaned embedding text for both questions."""
    df = df.copy()
    df[f"{q1_col}_clean"] = df[q1_col].apply(clean_for_classical)
    df[f"{q2_col}_clean"] = df[q2_col].apply(clean_for_classical)
    df[f"{q1_col}_emb"] = df[q1_col].apply(clean_for_embeddings)
    df[f"{q2_col}_emb"] = df[q2_col].apply(clean_for_embeddings)
    return df
