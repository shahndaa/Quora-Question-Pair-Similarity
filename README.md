# Quora Question Pair Similarity

A natural language processing project for detecting duplicate question pairs on Quora. The system determines whether two differently-worded questions are asking the same thing, using a combination of classical machine learning on hand-crafted linguistic features and a deep learning Siamese network, with an interactive demo application for testing predictions in real time.

Repository: https://github.com/shahndaa/Quora-Question-Pair-Similarity

---

## Table of Contents

- [Overview](#overview)
- [Motivation and Problem Statement](#motivation-and-problem-statement)
- [Methodology](#methodology)
  - [Text Preprocessing](#1-text-preprocessing)
  - [Feature Engineering](#2-feature-engineering)
  - [Classical Machine Learning Models](#3-classical-machine-learning-models)
  - [Deep Learning: Siamese LSTM](#4-deep-learning-siamese-lstm)
  - [Interactive Demo Application](#5-interactive-demo-application)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Running the Demo Application](#running-the-demo-application)
  - [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Evaluation Methodology](#evaluation-methodology)
- [Testing](#testing)
- [Technology Stack](#technology-stack)

---

## Overview

Given two questions submitted independently to Quora, the objective is to predict whether they carry the same underlying intent, even when they are phrased differently. This is a binary classification problem (`is_duplicate ∈ {0, 1}`) based on the dataset originally released by Quora and used in the associated [Kaggle competition](https://www.kaggle.com/c/quora-question-pairs), which contains approximately 404,000 labeled question pairs with a duplicate rate of roughly 37 percent.

Solving this problem well requires more than surface-level string comparison. Two questions can be lexically very different while expressing an identical need ("How can I lose weight fast?" versus "What is the quickest way to shed pounds?"), and conversely, two questions can share many of the same words while asking about entirely different things. The approach in this repository therefore combines several complementary signals: statistical and lexical overlap features, fuzzy string matching, term-frequency similarity, semantic similarity from word embeddings, and a sequence model capable of learning contextual patterns directly from the data.

## Motivation and Problem Statement

Duplicate question detection is a practically significant problem for any community question-answering platform. When users submit questions that already have established answers elsewhere on the platform, failing to identify the duplication leads to fragmented knowledge, redundant effort from contributors, and a degraded experience for users searching for answers. Automating this detection allows a platform to merge duplicate threads, surface existing answers immediately, and reduce the burden on human moderators.

From a technical standpoint, the task is a well-studied benchmark in natural language processing for evaluating sentence-pair similarity and semantic matching techniques, since it requires reasoning about paraphrase, entailment, and topical overlap rather than exact or near-exact string matching.

## Methodology

### 1. Text Preprocessing

Two separate preprocessing pipelines are used, since different downstream models benefit from different representations of the text (implemented in `src/preprocessing.py`):

- **Aggressive cleaning**, used for statistical and TF-IDF-based features: lowercasing, expansion of contractions, removal of embedded LaTeX math blocks (a quirk present in the raw Quora data), punctuation and digit stripping, stopword removal, and lemmatization. This representation discards function words and inflectional variation to emphasize shared content words.
- **Light cleaning**, used for the sequence model: lowercasing and punctuation normalization only. Word order and stopwords are preserved, since the LSTM relies on sequential structure that the aggressive pipeline would destroy.

### 2. Feature Engineering

The classical pipeline (`src/features.py`) constructs a feature matrix per question pair rather than relying on a single similarity score, which was the primary limitation of the previous version of this project. The features include:

- **Length-based features**: character length, word count, and their absolute differences between the two questions.
- **Lexical overlap features**: count of shared words, ratio of shared words to the shorter and longer question, Jaccard similarity, and whether the first or last word matches.
- **Fuzzy string-matching features**, computed with [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz): simple ratio, partial ratio, token sort ratio, and token set ratio, which are robust to reordering and partial overlaps.
- **TF-IDF cosine similarity**: a shared TF-IDF vocabulary (unigrams and bigrams) is fit across both questions jointly, and the cosine similarity between each question pair's TF-IDF vectors is computed row-wise.
- **Semantic embedding similarity**: pretrained GloVe word vectors are averaged over each question's tokens, and the cosine similarity between the two resulting vectors is used as a feature. This captures semantic relatedness that purely lexical methods miss, such as the relationship between "purchase" and "buy."

### 3. Classical Machine Learning Models

Four models are trained and compared on the engineered feature matrix (`src/models_classical.py`):

- Logistic Regression (linear baseline, class-weighted for the dataset's imbalance)
- Random Forest
- Gradient Boosting
- XGBoost

Each model is evaluated on a held-out test split using accuracy, precision, recall, F1 score, ROC-AUC, and log loss. The best-performing model by F1 score, rather than accuracy alone, is selected and persisted, since accuracy is a misleading metric on a dataset where the two classes are not evenly represented.

### 4. Deep Learning: Siamese LSTM

A Siamese neural network is implemented in `src/models_deep.py`. Both questions in a pair are passed through the same embedding layer (initialized with pretrained GloVe vectors) and the same bidirectional LSTM encoder, meaning the model learns a single, consistent way of representing a question rather than two independent representations. The two resulting encodings are then combined through their absolute difference and their elementwise product before being passed through a small fully connected classification head with dropout regularization.

This weight-sharing design is standard for symmetric sentence-pair tasks: it substantially reduces the number of parameters relative to encoding each question with a separate network, and it enforces that the similarity judgment is based on a consistent representation space for both questions.

### 5. Interactive Demo Application

A [Streamlit](https://streamlit.io) application (`app.py`) allows a user to enter two questions directly and receive a live prediction, the associated probability, and, for the classical model, a breakdown of the underlying feature values that produced the prediction. This is intended both as a way to sanity-check the trained models and as a lightweight demonstration of the end-to-end system.

## Project Structure

```
Quora-Question-Pair-Similarity/
├── data/
│   ├── sample_dataset.csv            Small 250-row sample for quick pipeline checks
│   └── train.csv                     Full 404K-row training data (not committed to
│                                       git; download it yourself, see Dataset below)
├── models/                           Trained artifacts (included, trained on the
│   │                                   full 404K-row dataset; see Evaluation below)
│   ├── classical_best_model.joblib   Best classical model (XGBoost)
│   ├── scaler.joblib                 Feature scaler used before prediction
│   ├── tfidf_featurizer.joblib       Fitted TF-IDF vectorizer/similarity featurizer
│   ├── classical_results.json        Evaluation metrics for every classical model tried
│   ├── deep_model.keras              Trained Siamese LSTM
│   ├── tokenizer.joblib              Keras tokenizer used by the deep model
│   └── deep_results.json             Evaluation metrics for the deep model
├── src/
│   ├── preprocessing.py       Text cleaning (classical and embedding variants)
│   ├── features.py            Statistical, fuzzy, and TF-IDF feature construction
│   ├── embeddings.py          GloVe loading, embedding similarity, embedding matrix
│   ├── models_classical.py    Classical model training and evaluation
│   └── models_deep.py         Siamese LSTM architecture and training loop
├── tests/
│   └── test_features.py       Unit tests for the feature engineering functions
├── train.py                   Command-line entry point for training
├── predict.py                 Command-line entry point for single-pair prediction
├── app.py                     Streamlit demo application
├── requirements.txt           Core Python dependencies (no TensorFlow)
├── requirements-deep.txt      Optional: adds TensorFlow for the Siamese LSTM
└── README.md
```

## Installation

```bash
git clone https://github.com/shahndaa/Quora-Question-Pair-Similarity.git
cd Quora-Question-Pair-Similarity
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs everything needed for the classical pipeline, prediction, and the demo app. The Siamese LSTM (TensorFlow) is an optional extra, kept separate because TensorFlow's wheel availability lags behind new Python releases and can otherwise block installation on some platforms. To also use the deep learning model:

```bash
pip install -r requirements-deep.txt
```

The first run will also download NLTK corpora (stopwords, WordNet) and a pretrained GloVe embedding model via `gensim`, both of which are cached locally afterward.

## Dataset

A small sample (`data/sample_dataset.csv`, 250 rows) is included so that the full pipeline can be exercised immediately without any external download. The models shipped in `models/` (see [Evaluation Methodology](#evaluation-methodology)) were trained on the full 404,348-row dataset, not on this sample; the sample is provided only for quickly verifying the code runs end-to-end.

To retrain on the full dataset yourself:

1. Create a Kaggle account if you do not already have one.
2. Download the data from the [Quora Question Pairs competition page](https://www.kaggle.com/c/quora-question-pairs), or from the equivalent [dataset page](https://www.kaggle.com/datasets/quora/question-pairs-dataset).
3. Extract the archive and place the resulting CSV at `data/train.csv`. The expected columns are `id, qid1, qid2, question1, question2, is_duplicate`.

## Usage

### Training

```bash
# Train both the classical model zoo and the Siamese LSTM
python train.py --data data/train.csv --model both

# Train only the classical models (faster, no GPU required)
python train.py --data data/train.csv --model classical

# Train only the deep learning model
python train.py --data data/train.csv --model deep

# Train on a random subset of the data for faster iteration during development
python train.py --data data/train.csv --model both --sample-frac 0.1
```

Trained artifacts are written to `models/`, including the best classical model and its supporting scaler and TF-IDF featurizer, the trained Siamese LSTM and its tokenizer, and JSON files recording the full evaluation metrics for every model that was trained.

### Prediction

```bash
python predict.py "How do I learn Python?" "What is the best way to learn Python?"
python predict.py "How do I learn Python?" "How do I bake a cake?" --model deep
```

### Running the Demo Application

```bash
streamlit run app.py
```

This launches a local web interface where two questions can be entered and compared. The classical model runs by default; the Siamese LSTM is available behind an opt-in checkbox, since it downloads and loads a larger language model on first use.

### Deploying to Streamlit Community Cloud

The app can be deployed for free at [share.streamlit.io](https://share.streamlit.io):

1. Push this repository to GitHub (including the `models/` folder — the trained artifacts are small enough to commit).
2. On Streamlit Community Cloud, create a new app pointing at this repository, branch `main`, and main file path `app.py`.
3. No additional configuration is required. Community Cloud will install `requirements.txt` (the first dependency file it finds in the repository root), which does **not** include TensorFlow — see below.

**Why TensorFlow is deliberately excluded from the deployed environment:** Streamlit Community Cloud's Python version is controlled by the platform (via its "Advanced settings" dropdown at deploy time) and can change over time; at various points it has defaulted to a Python version newer than the ones TensorFlow currently publishes wheels for, which makes `pip install tensorflow-cpu` fail outright and breaks the entire deployment, not just the deep learning feature. Rather than depend on pinning a Python version that the platform may or may not honor, `requirements.txt` only contains the dependencies for the classical (XGBoost) pipeline, which are lightweight and always installable. `app.py` detects at runtime whether TensorFlow is available: if it isn't, the classical model is used and the deep learning checkbox is automatically disabled with an explanatory message, instead of the app failing to build.

To also enable the Siamese LSTM on your own deployment, add the contents of `requirements-deep.txt` to `requirements.txt` before pushing, and set the app's Python version to 3.11 or 3.12 in "Advanced settings" when deploying (TensorFlow is not yet available for newer Python versions).

## Evaluation Methodology

Every model is evaluated on a held-out test split (20 percent, stratified by label) using accuracy, precision, recall, F1 score, ROC-AUC, and log loss rather than accuracy alone, since the dataset is imbalanced (approximately 63 percent non-duplicate versus 37 percent duplicate). The results below come from `train.py` run against the full 404,348-row dataset and are recorded in `models/classical_results.json` and `models/deep_results.json`.

**Classical models** (feature engineering described above, evaluated on the test split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log Loss |
|---|---|---|---|---|---|---|
| XGBoost *(selected)* | 0.767 | 0.678 | 0.700 | 0.689 | 0.849 | 0.450 |
| Random Forest | 0.748 | 0.613 | 0.860 | 0.716 | 0.847 | 0.472 |
| Gradient Boosting | 0.759 | 0.671 | 0.681 | 0.676 | 0.841 | 0.463 |
| Logistic Regression | 0.711 | 0.582 | 0.774 | 0.664 | 0.797 | 0.540 |

XGBoost is saved as the deployed classical model. It does not have the highest F1 of the four (Random Forest does), but Random Forest's unbounded-depth variant serializes to a multi-gigabyte file that is impractical to distribute; a depth-capped Random Forest fits within a reasonable file size but trades away most of the accuracy advantage, so XGBoost is used instead as the best accuracy-to-size tradeoff. `train.py` enforces a maximum model file size and automatically falls back to the next-best model by F1 if the top candidate exceeds it.

**Siamese LSTM** (test split, full dataset):

| Metric | Value |
|---|---|
| Accuracy | 0.840 |
| ROC-AUC | 0.919 |
| Loss (binary cross-entropy) | 0.358 |

The deep learning model outperforms every classical model by a clear margin, which is consistent with expectations: a sequence model that processes questions in order can capture compositional and contextual meaning (negation, word order, multi-word phrases) that a bag-of-features representation cannot.

**A known limitation:** both models rely on general-purpose GloVe embeddings trained on Wikipedia and news text, so neither reliably recognizes domain-specific paraphrases outside that vocabulary — for example, "How can one get rid of gynecomastia?" and "How can I get rid of man boobs?" are a true duplicate pair in the dataset, but both models score them as unlikely duplicates, since the medical term and its colloquial equivalent are not close together in general-purpose embedding space. This is a genuine property of the embeddings used, not a training bug; it would require domain-adapted or contextual embeddings (e.g. a transformer fine-tuned on this data) to fix.

## Testing

Unit tests for the feature engineering logic are provided under `tests/` and can be run with:

```bash
pytest tests/
```

## Technology Stack

- **Language**: Python
- **Data handling**: pandas, NumPy
- **Classical machine learning**: scikit-learn, XGBoost
- **Natural language processing**: NLTK, gensim (GloVe embeddings), RapidFuzz
- **Deep learning**: TensorFlow / Keras (optional, see `requirements-deep.txt`)
- **Application interface**: Streamlit
- **Testing**: pytest

