import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from preprocessing import clean_for_classical, clean_for_embeddings
from features import (
    statistical_features,
    fuzzy_features,
    TfidfSimilarityFeaturizer,
    build_feature_matrix,
)


def test_clean_for_classical_removes_stopwords_and_lemmatizes():
    result = clean_for_classical("What are the Cats doing?")
    assert "the" not in result.split()
    assert "cat" in result  # "cats" -> lemma "cat"


def test_clean_for_classical_handles_non_string():
    assert clean_for_classical(None) == ""
    assert clean_for_classical(float("nan")) == ""


def test_clean_for_embeddings_keeps_stopwords():
    result = clean_for_embeddings("What is the capital of France?")
    assert "is" in result.split()
    assert "the" in result.split()


def test_statistical_features_identical_questions():
    feats = statistical_features("how do i learn python", "how do i learn python")
    assert feats["common_word_count"] == 5
    assert feats["jaccard_similarity"] == 1.0
    assert feats["len_diff"] == 0


def test_statistical_features_disjoint_questions():
    feats = statistical_features("apple banana", "car truck")
    assert feats["common_word_count"] == 0
    assert feats["jaccard_similarity"] == 0.0


def test_fuzzy_features_identical_strings_score_100():
    feats = fuzzy_features("hello world", "hello world")
    assert feats["fuzz_ratio"] == 100
    assert feats["fuzz_token_sort_ratio"] == 100


def test_tfidf_similarity_identical_questions_score_high():
    featurizer = TfidfSimilarityFeaturizer()
    q1 = pd.Series(["how do i learn python fast", "what is the best pizza"])
    q2 = pd.Series(["how do i learn python fast", "completely unrelated sentence"])
    featurizer.fit(q1, q2)
    sims = featurizer.transform(q1, q2)
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert sims[1, 0] < sims[0, 0]


def test_build_feature_matrix_shape():
    df = pd.DataFrame({
        "question1_clean": ["learn python fast", "best pizza place"],
        "question2_clean": ["learn python quickly", "worst pizza place"],
    })
    featurizer = TfidfSimilarityFeaturizer()
    featurizer.fit(df["question1_clean"], df["question2_clean"])
    features = build_feature_matrix(df, featurizer)
    assert len(features) == 2
    assert "tfidf_cosine_sim" in features.columns
    assert "common_word_count" in features.columns
