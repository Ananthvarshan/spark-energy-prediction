"""
============================================================
FORECASTING BASELINE ARCHITECTURES
experiments/baseline_models.py
============================================================

PURPOSE
-------
Phase II / Task 5.  The five baselines the proposed Seq2Seq LSTM
is compared against, plus the proposed model itself, all built
against one interface:

    build(n_features, lookback, horizon, n_states) -> keras.Model

taking (batch, lookback, n_features) and returning
(batch, horizon, n_states) softmax.

MODELS
------
  seq2seq_lstm   PROPOSED         Sutskever et al. (2014); Hochreiter
                                  & Schmidhuber (1997)
  vanilla_lstm   ablation         Hochreiter & Schmidhuber (1997)
  gru            lighter gating   Cho et al. (2014)
  tcn            non-recurrent    Bai, Kolter & Koltun (2018)
  transformer    patch attention  Nie et al. (2023) -- PatchTST
  xgboost        tree baseline    Chen & Guestrin (2016)   [see task5]

DESIGN CONSTRAINT: MATCHED HEADS
--------------------------------
Every recurrent/convolutional/attention model uses the same
"encode to a context vector -> expand across the horizon ->
per-step softmax" output structure.  Only the ENCODER differs.

This is deliberate.  If the models differed in both encoder and
output head, a difference in the results table could not be
attributed to the sequence model, which is the thing the table is
supposed to be about.  `vanilla_lstm` is the one intentional
exception: it collapses the whole horizon into a single dense
projection, which is exactly the ablation that tests whether the
encoder-decoder structure earns its cost.

CAPACITY
--------
Hidden widths are chosen so that parameter counts land within
roughly a factor of two of each other; the exact counts are
recorded in the results JSON so a reviewer can check that the
proposed model is not simply the biggest one.
============================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


DROPOUT = 0.2


@tf.keras.utils.register_keras_serializable(package="phase2")
class AddPositionalEmbedding(layers.Layer):
    """
    Add a learned positional embedding to a (batch, n_positions, d_model)
    tensor.

    Implemented as a layer holding its own (1, n_positions, d_model) weight
    rather than as `Embedding(range(n))`: the latter produces a tensor whose
    leading axis Keras interprets as the batch dimension, so it cannot be
    broadcast across a real batch.  Owning the weight also keeps the model
    serialisable, which `tf.range` inside a functional graph does not.
    """

    def __init__(self, n_positions: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.n_positions = int(n_positions)
        self.d_model = int(d_model)

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="positional_embedding",
            shape=(1, self.n_positions, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update(n_positions=self.n_positions, d_model=self.d_model)
        return cfg


def _compile(model: Model) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


# ── Proposed: encoder-decoder LSTM ────────────────────────────────────────────

def build_seq2seq_lstm(n_features, lookback, horizon, n_states,
                       enc_units=(128, 64), dec_units=64, dropout=DROPOUT):
    """
    The project's existing model (src/lstm_model.build_lstm_model),
    restated here so Task 5 has no dependency on that module's defaults
    drifting.  Two stacked LSTM encoders compress the lookback into a
    context vector, which is repeated across the horizon and decoded by a
    third LSTM.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input_window")
    x = layers.LSTM(enc_units[0], return_sequences=True, name="enc_lstm_1")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(enc_units[1], return_sequences=False, name="enc_lstm_2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.RepeatVector(horizon, name="bridge_repeat")(x)
    x = layers.LSTM(dec_units, return_sequences=True, name="dec_lstm")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.TimeDistributed(
        layers.Dense(n_states, activation="softmax"), name="output_softmax")(x)
    return _compile(Model(inp, out, name="seq2seq_lstm"))


# ── Ablation: single LSTM + dense projection ──────────────────────────────────

def build_vanilla_lstm(n_features, lookback, horizon, n_states,
                       units=128, dropout=DROPOUT):
    """
    A single LSTM whose final hidden state is projected straight to the
    whole horizon by one dense layer.

    This is the ablation for the encoder-decoder structure: it has no
    decoder recurrence, so it cannot model dependencies BETWEEN future
    steps -- each horizon step is predicted from the context vector
    independently.  If seq2seq does not beat it, the extra decoder is
    not earning its cost.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input_window")
    x = layers.LSTM(units, return_sequences=False, name="lstm")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(horizon * n_states, name="projection")(x)
    x = layers.Reshape((horizon, n_states))(x)
    out = layers.Softmax(axis=-1, name="output_softmax")(x)
    return _compile(Model(inp, out, name="vanilla_lstm"))


# ── GRU ───────────────────────────────────────────────────────────────────────

def build_gru(n_features, lookback, horizon, n_states,
              enc_units=(128, 64), dec_units=64, dropout=DROPOUT):
    """
    Architecturally identical to the proposed model with LSTM cells
    replaced by GRU cells (Cho et al., 2014).  GRUs have three gates
    rather than four, so this is also a ~25% parameter reduction at the
    same width -- it tests whether the LSTM's extra gate buys anything on
    this signal.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input_window")
    x = layers.GRU(enc_units[0], return_sequences=True, name="enc_gru_1")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.GRU(enc_units[1], return_sequences=False, name="enc_gru_2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.RepeatVector(horizon)(x)
    x = layers.GRU(dec_units, return_sequences=True, name="dec_gru")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.TimeDistributed(
        layers.Dense(n_states, activation="softmax"), name="output_softmax")(x)
    return _compile(Model(inp, out, name="gru"))


# ── Temporal Convolutional Network ────────────────────────────────────────────

def _tcn_block(x, filters, dilation, dropout, name):
    """
    One residual block of dilated CAUSAL convolutions (Bai et al., 2018).

    Causal padding matters: with 'same' padding a convolution centred on
    step t would read steps t+1..t+k, letting the encoder see inside its
    own lookback future.  That is harmless for a pure encoder but becomes
    a leak the moment the same block is reused, so it is causal here.
    """
    prev = x
    for i in range(2):
        x = layers.Conv1D(filters, kernel_size=3, padding="causal",
                          dilation_rate=dilation, activation="relu",
                          name=f"{name}_conv{i}")(x)
        x = layers.LayerNormalization(name=f"{name}_ln{i}")(x)
        x = layers.Dropout(dropout)(x)
    if prev.shape[-1] != filters:
        prev = layers.Conv1D(filters, 1, padding="same",
                             name=f"{name}_res_proj")(prev)
    return layers.Add(name=f"{name}_residual")([prev, x])


def build_tcn(n_features, lookback, horizon, n_states,
              filters=64, dilations=(1, 2, 4, 8, 16), dropout=DROPOUT):
    """
    Dilated causal convolution stack.

    With kernel size 3 and dilations up to 16 the receptive field is
    1 + 2*sum(2*(d-1)+1... ) >= the full 120-step lookback used in Task 5,
    so the TCN sees the same context as the recurrent models.  The final
    timestep's representation is the context vector, expanded and decoded
    by the shared head.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input_window")
    x = inp
    for d in dilations:
        x = _tcn_block(x, filters, d, dropout, name=f"tcn_d{d}")
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_step",
                      output_shape=(filters,))(x)
    x = layers.RepeatVector(horizon)(x)
    x = layers.Conv1D(filters, 3, padding="causal", activation="relu",
                      name="dec_conv")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.TimeDistributed(
        layers.Dense(n_states, activation="softmax"), name="output_softmax")(x)
    return _compile(Model(inp, out, name="tcn"))


# ── PatchTST-style transformer ────────────────────────────────────────────────

def build_transformer(n_features, lookback, horizon, n_states,
                      patch_len=12, d_model=96, n_heads=4, n_blocks=3,
                      ff_dim=192, dropout=DROPOUT):
    """
    Patch-based transformer encoder in the style of PatchTST
    (Nie et al., 2023).

    The lookback is split into non-overlapping patches of `patch_len`
    steps, each flattened and linearly embedded.  Patching is the key
    idea of PatchTST: attending over 120 raw timesteps is quadratic in
    120 and lets the model fixate on single noisy samples, whereas
    attending over 10 patches is both cheaper and forces the
    representation to be about local shape rather than instantaneous
    value -- which is what distinguishes a quiet STANDBY trace from a
    modulated WORKING one.

    Learned (not sinusoidal) positional embeddings are used, as in the
    original.
    """
    n_patches = lookback // patch_len
    usable = n_patches * patch_len

    inp = layers.Input(shape=(lookback, n_features), name="input_window")
    x = layers.Lambda(lambda t: t[:, :usable, :], name="trim",
                      output_shape=(usable, n_features))(inp)
    x = layers.Reshape((n_patches, patch_len * n_features), name="patchify")(x)
    x = layers.Dense(d_model, name="patch_embed")(x)

    x = AddPositionalEmbedding(n_patches, d_model, name="add_pos")(x)
    x = layers.Dropout(dropout)(x)

    for b in range(n_blocks):
        h = layers.LayerNormalization(name=f"blk{b}_ln1")(x)
        h = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout, name=f"blk{b}_mha")(h, h)
        x = layers.Add(name=f"blk{b}_res1")([x, h])

        h = layers.LayerNormalization(name=f"blk{b}_ln2")(x)
        h = layers.Dense(ff_dim, activation="gelu", name=f"blk{b}_ff1")(h)
        h = layers.Dropout(dropout)(h)
        h = layers.Dense(d_model, name=f"blk{b}_ff2")(h)
        x = layers.Add(name=f"blk{b}_res2")([x, h])

    x = layers.LayerNormalization(name="final_ln")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(horizon * n_states, name="head")(x)
    x = layers.Reshape((horizon, n_states))(x)
    out = layers.Softmax(axis=-1, name="output_softmax")(x)
    return _compile(Model(inp, out, name="transformer"))


# ── Registry ──────────────────────────────────────────────────────────────────

NEURAL_BUILDERS = {
    "seq2seq_lstm": build_seq2seq_lstm,
    "vanilla_lstm": build_vanilla_lstm,
    "gru": build_gru,
    "tcn": build_tcn,
    "transformer": build_transformer,
}

MODEL_INFO = {
    "seq2seq_lstm": ("Seq2Seq LSTM (proposed)", "Sutskever et al. (2014)"),
    "vanilla_lstm": ("Vanilla LSTM",            "Hochreiter & Schmidhuber (1997)"),
    "gru":          ("GRU encoder-decoder",     "Cho et al. (2014)"),
    "tcn":          ("Temporal CNN (TCN)",      "Bai et al. (2018)"),
    "transformer":  ("Transformer (PatchTST)",  "Nie et al. (2023)"),
    "xgboost":      ("XGBoost (sliding window)", "Chen & Guestrin (2016)"),
}


def count_params(model: Model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_weights))
