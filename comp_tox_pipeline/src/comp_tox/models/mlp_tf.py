"""TensorFlow/Keras MLP baseline on Morgan fingerprints.

Same scaffold split, same Platt-on-validation calibration, same eval path
as the sklearn baselines and the PyTorch GNN. This head exists to exercise the
third framework (PyTorch GNN, JAX/Flax parity port, TF/Keras here),
not to claim a better model class.

`_KerasMlp` is an sklearn-style adapter storing only the dense-layer spec
and numpy weights, so the model bundle stays joblib-picklable. Keras
models themselves are not safely picklable.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class _KerasMlp(ClassifierMixin, BaseEstimator):
    """predict_proba adapter around a Keras MLP; stores weights, not the graph.

    BaseEstimator/ClassifierMixin give it the sklearn contract (tags,
    get_params/clone) that FrozenEstimator + CalibratedClassifierCV need.
    """

    def __init__(self, n_in: int, arch: list, weights: list):
        self.n_in = n_in
        self.arch = arch  # [(units, activation_name), ...]
        self.weights = weights  # list of np.ndarray, get_weights() order
        self.classes_ = np.array([0, 1])
        self._model = None

    def __getstate__(self):
        # the live Keras graph is not picklable; weights + spec are
        return {
            "n_in": self.n_in,
            "arch": self.arch,
            "weights": self.weights,
        }

    def __setstate__(self, state):
        self.__init__(state["n_in"], state["arch"], state["weights"])

    @classmethod
    def from_model(cls, model, n_in: int) -> "_KerasMlp":
        import tensorflow as tf

        dense = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dense)
        ]
        arch = [(int(l.units), l.activation.__name__) for l in dense]
        return cls(n_in, arch, model.get_weights())

    def _build(self):
        import tensorflow as tf

        x_in = tf.keras.Input(shape=(self.n_in,))
        x = x_in
        for units, act in self.arch:
            x = tf.keras.layers.Dense(units, activation=act)(x)
        model = tf.keras.Model(x_in, x)
        model.set_weights(self.weights)
        return model

    def predict_proba(self, X):
        if self._model is None:
            self._model = self._build()
        if sparse.issparse(X):
            X = X.toarray()
        p = self._model.predict(np.asarray(X, dtype=np.float32), verbose=0)
        p = p.reshape(-1)
        return np.column_stack([1.0 - p, p])

    def fit(self, X=None, y=None):
        # weights are baked in at construction (train_mlp_tf); sklearn's
        # check_is_fitted only requires the method to exist
        return self

    def predict(self, X):
        # needed so FrozenEstimator satisfies cross_val_predict's contract
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_mlp_tf(X, y, seed: int, epochs: int = 30, batch_size: int = 128):
    """Train the Keras MLP on the train split; returns a _KerasMlp.

    Determinism: `set_random_seed` covers weight init, dropout, and shuffle
    order. TF ops are deterministic at this scale on a given build, but
    unlike the lbfgs baseline this is seed-level reproducibility, not
    bit-level.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Balanced class weights: the endpoint is ~4-9% actives; unweighted BCE
    # would collapse toward the inactive majority.
    n, n_pos = len(y), float(y.sum())
    if n_pos == 0 or n_pos == n:
        raise ValueError(
            f"train split is single-class ({int(n_pos)} of {n} active); "
            "balanced class weights are undefined"
        )
    w = {0.0: n / (2 * (n - n_pos)), 1.0: n / (2 * n_pos)}

    # Shuffle once up front: keras validation_split takes the *tail* of the
    # array, which is only a random subset if the rows were shuffled.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auroc")],
    )
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        class_weight=w,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auroc",
                mode="max",
                patience=5,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    return _KerasMlp.from_model(model, X.shape[1])
