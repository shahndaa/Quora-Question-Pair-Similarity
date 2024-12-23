"""
Siamese LSTM deep-learning model for the Quora Question Pair Similarity
project.

Architecture:
  question1 --> [shared Embedding (GloVe init) -> shared BiLSTM] --\
                                                                     >-- |diff|, multiply --> Dense -> Dense(1, sigmoid)
  question2 --> [shared Embedding (GloVe init) -> shared BiLSTM] --/

Sharing weights between the two towers (a true "Siamese" network) means the
model learns ONE way of encoding a question, then compares the two encodings
-- this is both more parameter-efficient and more appropriate for a
symmetric similarity task than concatenating two separately-encoded questions.
"""
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@keras.saving.register_keras_serializable(package="quora_qp", name="AbsDifference")
class AbsDifference(layers.Layer):
    """Element-wise |a - b|. A real Layer subclass (not a raw Python lambda)
    so the saved .keras model can be reloaded safely without disabling
    Keras's Lambda-deserialization safety check."""

    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)


def build_tokenizer(texts, num_words: int = 50_000) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded(tokenizer: Tokenizer, texts, max_len: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def build_siamese_lstm(
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    embedding_matrix: np.ndarray | None = None,
    lstm_units: int = 64,
    trainable_embeddings: bool = False,
) -> tf.keras.Model:
    input_q1 = layers.Input(shape=(max_len,), name="question1")
    input_q2 = layers.Input(shape=(max_len,), name="question2")

    embedding_kwargs = dict(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True,
        trainable=trainable_embeddings,
    )
    if embedding_matrix is not None:
        embedding_kwargs["weights"] = [embedding_matrix]

    shared_embedding = layers.Embedding(**embedding_kwargs, name="shared_embedding")
    shared_lstm = layers.Bidirectional(layers.LSTM(lstm_units), name="shared_bilstm")

    encoded_q1 = shared_lstm(shared_embedding(input_q1))
    encoded_q2 = shared_lstm(shared_embedding(input_q2))

    diff = AbsDifference(name="abs_diff")([encoded_q1, encoded_q2])
    mult = layers.Multiply(name="elementwise_product")([encoded_q1, encoded_q2])
    merged = layers.Concatenate(name="merge")([diff, mult])

    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid", name="is_duplicate")(x)

    model = models.Model(inputs=[input_q1, input_q2], outputs=output, name="siamese_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_siamese_lstm(
    model: tf.keras.Model,
    q1_train, q2_train, y_train,
    q1_val, q2_val, y_val,
    epochs: int = 15,
    batch_size: int = 256,
):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=3, restore_best_weights=True
    )
    history = model.fit(
        {"question1": q1_train, "question2": q2_train},
        y_train,
        validation_data=({"question1": q1_val, "question2": q2_val}, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )
    return history
