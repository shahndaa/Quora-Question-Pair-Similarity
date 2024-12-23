"""
Streamlit demo for the Quora Question Pair Similarity project.

Run with:
    streamlit run app.py

Lets you type two questions and see whether the trained model(s) think
they're duplicates, along with the probability and the underlying
similarity features (for the classical model).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from preprocessing import clean_for_classical, clean_for_embeddings
from features import build_statistical_and_fuzzy_features
from embeddings import load_glove, sentence_to_avg_vector, cosine
from models_classical import load_artifact

MODEL_DIR = Path("models")
GLOVE_NAME = "glove-wiki-gigaword-100"
GLOVE_DIM = 100
MAX_LEN = 30


def _deep_learning_available() -> bool:
    """Check whether TensorFlow is importable in this environment. It's an
    optional dependency (see requirements-deep.txt) so the app, and the
    lightweight classical model, keep working even where TensorFlow has no
    wheels available (e.g. very new Python versions)."""
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


DEEP_LEARNING_AVAILABLE = _deep_learning_available()

st.set_page_config(page_title="Quora Question Pair Similarity", page_icon="❓", layout="centered")


@st.cache_resource(show_spinner="Loading models...")
def load_classical_artifacts():
    model = load_artifact(MODEL_DIR / "classical_best_model.joblib")
    scaler = load_artifact(MODEL_DIR / "scaler.joblib")
    tfidf_feat = load_artifact(MODEL_DIR / "tfidf_featurizer.joblib")
    kv = load_glove(GLOVE_NAME)
    return model, scaler, tfidf_feat, kv


@st.cache_resource(show_spinner="Loading deep model...")
def load_deep_artifacts():
    import tensorflow as tf
    from models_deep import AbsDifference  # noqa: F401 (ensures custom layer is registered)

    model = tf.keras.models.load_model(MODEL_DIR / "deep_model.keras")
    tokenizer = load_artifact(MODEL_DIR / "tokenizer.joblib")
    return model, tokenizer


def predict_classical(q1: str, q2: str):
    model, scaler, tfidf_feat, kv = load_classical_artifacts()

    q1_clean, q2_clean = clean_for_classical(q1), clean_for_classical(q2)
    row_df = pd.DataFrame({"question1_clean": [q1_clean], "question2_clean": [q2_clean]})

    stat_fuzzy = build_statistical_and_fuzzy_features(row_df, "question1_clean", "question2_clean")
    tfidf_sim = tfidf_feat.transform(row_df["question1_clean"], row_df["question2_clean"])

    v1 = sentence_to_avg_vector(clean_for_embeddings(q1), kv, GLOVE_DIM)
    v2 = sentence_to_avg_vector(clean_for_embeddings(q2), kv, GLOVE_DIM)
    emb_sim = cosine(v1, v2)

    features = stat_fuzzy.copy()
    features["tfidf_cosine_sim"] = tfidf_sim.flatten()
    features["embedding_cosine_sim"] = emb_sim

    X = scaler.transform(features)
    proba = model.predict_proba(X)[0, 1]
    return proba, features.iloc[0]


def predict_deep(q1: str, q2: str):
    from models_deep import texts_to_padded

    model, tokenizer = load_deep_artifacts()
    q1_pad = texts_to_padded(tokenizer, [clean_for_embeddings(q1)], MAX_LEN)
    q2_pad = texts_to_padded(tokenizer, [clean_for_embeddings(q2)], MAX_LEN)
    proba = model.predict({"question1": q1_pad, "question2": q2_pad}, verbose=0)[0, 0]
    return float(proba)


st.title("❓ Quora Question Pair Similarity")
st.caption("Type two questions below to check whether they're asking the same thing.")

col1, col2 = st.columns(2)
with col1:
    question1 = st.text_area("Question 1", "How do I learn Python quickly?", height=100)
with col2:
    question2 = st.text_area("Question 2", "What is the fastest way to learn Python?", height=100)

use_deep = st.checkbox(
    "Also try the deep learning model (Siamese LSTM)",
    value=False,
    disabled=not DEEP_LEARNING_AVAILABLE,
    help=(
        "Off by default to keep the app fast to load. The classical model "
        "(TF-IDF + engineered features + XGBoost) is used by default and is "
        "lightweight. Enabling this downloads and loads a larger language "
        "model on first use, which takes longer."
        if DEEP_LEARNING_AVAILABLE
        else "Unavailable in this environment: TensorFlow isn't installed here "
        "(see requirements-deep.txt to enable it locally)."
    ),
)
if not DEEP_LEARNING_AVAILABLE:
    st.caption(
        "ℹ️ Running in classical-only mode: TensorFlow isn't available in this "
        "deployment, so only the classical (XGBoost) model is active."
    )

if st.button("Check similarity", type="primary"):
    if not question1.strip() or not question2.strip():
        st.warning("Please enter both questions.")
    else:
        try:
            with st.spinner("Scoring with the classical model..."):
                proba_classical, feature_row = predict_classical(question1, question2)

            label = "🟢 Likely DUPLICATE" if proba_classical >= 0.5 else "🔴 Likely NOT a duplicate"
            st.subheader(f"Classical model: {label}")
            st.progress(min(max(proba_classical, 0.0), 1.0))
            st.write(f"Duplicate probability: **{proba_classical:.1%}**")

            with st.expander("See underlying similarity features"):
                st.dataframe(feature_row.to_frame("value"))

            if use_deep:
                st.markdown("---")
                with st.spinner("Loading the deep learning model (first use only) and scoring..."):
                    proba_deep = predict_deep(question1, question2)

                label_deep = "🟢 Likely DUPLICATE" if proba_deep >= 0.5 else "🔴 Likely NOT a duplicate"
                st.subheader(f"Deep learning model (Siamese LSTM): {label_deep}")
                st.progress(min(max(proba_deep, 0.0), 1.0))
                st.write(f"Duplicate probability: **{proba_deep:.1%}**")
        except FileNotFoundError:
            st.error(
                "No trained model found in `models/`. Run `python train.py --data data/train.csv "
                "--model both` first, then reload this app."
            )

st.markdown("---")
st.caption(
    "Model artifacts are loaded from the `models/` folder produced by `train.py`. "
    "See the README for how to train on the full Quora Question Pairs dataset."
)
