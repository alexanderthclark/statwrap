"""Utilities for inspecting trained scikit-learn multi-layer perceptrons.

The :class:`MLPInspector` class exposes the internal representations of a fitted
``sklearn.neural_network.MLPClassifier`` (optionally wrapped inside a
``Pipeline``). It provides convenient access to pre-activations (``Z``) and
activations (``A``) for each layer, tools to visualise per-layer embeddings via
PCA/t-SNE, neuron-centric plots (activation histograms/heatmaps and incoming
weights), utilities for discovering the samples that excite a neuron the most,
and basic manual differentiation routines that enable saliency maps and simple
activation maximisation in the input space.  All computations are carried out in
NumPy without relying on automatic differentiation frameworks, which keeps the
module lightweight and easy to understand.  The focus is on fully-connected
``MLPClassifier`` networks trained with scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


try:  # Optional dependency used for interactive plots
    import plotly.express as px

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PLOTLY = False


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ArrayLike = np.ndarray


def _ensure_2d(X: ArrayLike) -> ArrayLike:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def _activation_forward(name: str, Z: ArrayLike) -> ArrayLike:
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-Z))
    if name == "tanh":
        return np.tanh(Z)
    if name == "relu":
        return np.maximum(0.0, Z)
    if name == "identity":
        return Z
    if name == "softmax":
        Z = np.asarray(Z)
        shift = Z - Z.max(axis=1, keepdims=True)
        exp = np.exp(shift)
        return exp / exp.sum(axis=1, keepdims=True)
    raise ValueError(f"Unsupported activation '{name}'.")


def _activation_backward(name: str, Z: ArrayLike, A: ArrayLike) -> ArrayLike:
    if name == "logistic":
        return A * (1.0 - A)
    if name == "tanh":
        return 1.0 - A**2
    if name == "relu":
        return (Z > 0).astype(Z.dtype)
    if name == "identity":
        return np.ones_like(Z)
    if name == "softmax":
        raise ValueError("Softmax derivative requires full Jacobian; handle separately.")
    raise ValueError(f"Unsupported activation '{name}'.")


def _maybe_square_shape(n_features: int) -> tuple[int, int] | None:
    side = int(np.sqrt(n_features))
    if side * side == n_features:
        return side, side
    return None


def _grid_size(n_items: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(n_items)))
    rows = int(np.ceil(n_items / cols))
    return rows, cols


def _tv_loss_and_grad(x: ArrayLike, eps: float = 1e-6) -> tuple[float, ArrayLike]:
    """Anisotropic total variation for flattened inputs that form a square."""

    flat = np.asarray(x, dtype=float)
    shape = _maybe_square_shape(flat.size)
    if shape is None:
        return 0.0, np.zeros_like(flat)

    img = flat.reshape(shape)
    grad = np.zeros_like(img)

    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)

    loss = np.sum(np.sqrt(dx**2 + eps)) + np.sum(np.sqrt(dy**2 + eps))

    # Horizontal gradients
    gx = dx / np.sqrt(dx**2 + eps)
    grad[:, :-1] += gx
    grad[:, 1:] -= gx

    # Vertical gradients
    gy = dy / np.sqrt(dy**2 + eps)
    grad[:-1, :] += gy
    grad[1:, :] -= gy

    return float(loss), grad.ravel()


@dataclass
class _TransformInfo:
    transformer: BaseEstimator
    forward: Callable[[ArrayLike], ArrayLike]
    backward: Callable[[ArrayLike], ArrayLike] | None


class MLPInspector:
    """Introspect fitted :class:`~sklearn.neural_network.MLPClassifier` objects.

    Parameters
    ----------
    model:
        Fitted :class:`MLPClassifier` or a :class:`~sklearn.pipeline.Pipeline`
        whose final step is an ``MLPClassifier``.
    preprocess:
        Optional callable ``X -> X_pre`` applied before any pipeline transforms
        (useful if the inspector should operate on already-normalised data).
    """

    def __init__(self, model: BaseEstimator, preprocess: Callable[[ArrayLike], ArrayLike] | None = None):
        if isinstance(model, Pipeline):
            if not model.steps:
                raise ValueError("Pipeline must contain at least one step.")
            if not isinstance(model[-1], MLPClassifier):
                raise TypeError("Pipeline must terminate with an MLPClassifier.")
            self._pipeline: Pipeline | None = model
            self._estimator: MLPClassifier = model[-1]
        elif isinstance(model, MLPClassifier):
            self._pipeline = None
            self._estimator = model
        else:
            raise TypeError("model must be an MLPClassifier or Pipeline ending in one.")

        if not hasattr(self._estimator, "coefs_"):
            raise ValueError("The provided MLPClassifier instance is not fitted (missing 'coefs_').")

        self.preprocess = preprocess
        self._transforms: list[_TransformInfo] = []

        if self._pipeline is not None:
            for name, transformer in self._pipeline.steps[:-1]:
                if not hasattr(transformer, "transform"):
                    raise TypeError(f"Pipeline step '{name}' does not implement transform().")
                backward = self._infer_backward(transformer)
                self._transforms.append(
                    _TransformInfo(transformer=transformer, forward=transformer.transform, backward=backward)
                )

    # ------------------------------------------------------------------
    # Properties mirroring the estimator's attributes
    @property
    def hidden_layer_sizes(self) -> tuple[int, ...]:
        """Hidden layer sizes of the wrapped MLP."""

        hl = self._estimator.hidden_layer_sizes
        if isinstance(hl, int):
            return (hl,)
        return tuple(hl)

    @property
    def n_layers_(self) -> int:
        return len(self._estimator.coefs_)

    @property
    def coefs_(self) -> List[ArrayLike]:
        return list(self._estimator.coefs_)

    @property
    def intercepts_(self) -> List[ArrayLike]:
        return list(self._estimator.intercepts_)

    @property
    def activation_name(self) -> str:
        return self._estimator.activation

    @property
    def out_activation_name(self) -> str:
        return self._estimator.out_activation_

    # ------------------------------------------------------------------
    def _apply_forward_transforms(self, X: ArrayLike) -> ArrayLike:
        X_net = _ensure_2d(X)
        if self.preprocess is not None:
            X_net = _ensure_2d(self.preprocess(X_net))
        for info in self._transforms:
            X_net = _ensure_2d(info.forward(X_net))
        return X_net

    def _apply_backward_transforms(self, grad: ArrayLike) -> ArrayLike:
        grad_in = np.asarray(grad)
        for info in reversed(self._transforms):
            if info.backward is None:
                raise RuntimeError(
                    f"Cannot backpropagate through transformer {info.transformer.__class__.__name__}."
                )
            grad_in = _ensure_2d(info.backward(grad_in))
        if self.preprocess is not None:
            # Assume preprocess is a simple callable without Jacobian; best effort identity.
            # Users requiring precise gradients should incorporate preprocessing inside the pipeline.
            pass
        return grad_in

    @staticmethod
    def _infer_backward(transformer: BaseEstimator) -> Callable[[ArrayLike], ArrayLike] | None:
        if isinstance(transformer, StandardScaler):
            scale = getattr(transformer, "scale_", None)
            if scale is None:
                return None

            def backward(g: ArrayLike) -> ArrayLike:
                return g / scale

            return backward

        if isinstance(transformer, MinMaxScaler):
            scale = getattr(transformer, "scale_", None)
            if scale is None:
                return None

            def backward(g: ArrayLike) -> ArrayLike:
                return g * scale

            return backward

        return None

    # ------------------------------------------------------------------
    def forward(self, X: ArrayLike, return_linear: bool = True) -> dict:
        """Propagate inputs through the network.

        Parameters
        ----------
        X:
            Samples to propagate. They should be in the same representation as
            expected by the estimator (the inspector applies any pipeline
            transforms automatically).
        return_linear:
            Unused, retained for API completeness to match the specification.

        Returns
        -------
        dict
            ``{'Z': [...], 'A': [...], 'y_pred': ..., 'proba': ...}``
        """

        X_net = self._apply_forward_transforms(X)
        W = self.coefs_
        b = self.intercepts_
        hidden_activation = self.activation_name
        output_activation = self.out_activation_name

        Zs: List[ArrayLike] = []
        As: List[ArrayLike] = [X_net]
        current = X_net

        for i, (w, bias) in enumerate(zip(W, b)):
            Z = current @ w + bias
            Zs.append(Z)
            if i == len(W) - 1:
                A = _activation_forward(output_activation, Z)
            else:
                A = _activation_forward(hidden_activation, Z)
            As.append(A)
            current = A

        predictor = self._pipeline if self._pipeline is not None else self._estimator
        y_pred = predictor.predict(X)
        proba = None
        if hasattr(predictor, "predict_proba"):
            try:
                proba = predictor.predict_proba(X)
            except Exception:
                proba = None

        return {"Z": Zs, "A": As, "y_pred": y_pred, "proba": proba}

    # ------------------------------------------------------------------
    def get_layer_activations(self, X: ArrayLike, layer_index: int, kind: str = "A") -> ArrayLike:
        """Return activations or pre-activations for a specific layer.

        Parameters
        ----------
        X:
            Input samples.
        layer_index:
            ``0`` corresponds to the network input, ``1..L-1`` to hidden layers,
            and ``L`` to the output layer.
        kind:
            ``"A"`` for activations (default) or ``"Z"`` for pre-activations.
        """

        forward_pass = self.forward(X)
        if layer_index < 0:
            layer_index = len(forward_pass["A"]) - 1 + layer_index + 1
        total_layers = len(forward_pass["A"]) - 1
        if not (0 <= layer_index <= total_layers):
            raise IndexError(f"layer_index must be in [0, {total_layers}], got {layer_index}.")
        if kind not in {"A", "Z"}:
            raise ValueError("kind must be 'A' or 'Z'.")
        if kind == "A":
            return forward_pass["A"][layer_index]
        if layer_index == 0:
            raise ValueError("layer_index 0 (input) has no pre-activations.")
        return forward_pass["Z"][layer_index - 1]

    # ------------------------------------------------------------------
    def plot_layer_projection(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        layer_index: int = -1,
        method: str = "pca",
        annotate_centroids: bool = True,
        interactive: bool = False,
        random_state: int = 0,
    ):
        """Plot a 2D projection of layer activations via PCA or t-SNE."""

        activations = self.get_layer_activations(X, layer_index, kind="A")
        if method.lower() == "pca":
            projector = PCA(n_components=2, random_state=random_state)
            emb = projector.fit_transform(activations)
            explained = projector.explained_variance_ratio_.sum()
            title = f"Layer {layer_index} PCA (var={explained:.2%})"
        elif method.lower() in {"tsne", "t-sne"}:
            projector = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                perplexity=30,
                random_state=random_state,
            )
            emb = projector.fit_transform(activations)
            title = f"Layer {layer_index} t-SNE"
        else:
            raise ValueError("method must be 'pca' or 'tsne'.")

        if interactive:
            if not _HAS_PLOTLY:
                raise RuntimeError("plotly is required for interactive plots but is not installed.")
            fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=y if y is not None else None)
            fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
            return fig

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=35, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        if y is not None:
            legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
            ax.add_artist(legend)
            if annotate_centroids:
                for cls in np.unique(y):
                    mask = y == cls
                    centroid = emb[mask].mean(axis=0)
                    ax.annotate(str(cls), xy=centroid, xytext=(5, 5), textcoords="offset points", fontsize=10)
        return fig

    def plot_activation_hist(self, X: ArrayLike, layer_index: int, bins: int = 50):
        """Plot histograms of neuron activations for a given layer."""

        activations = self.get_layer_activations(X, layer_index)
        n_neurons = activations.shape[1]
        rows, cols = _grid_size(n_neurons)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        flat_axes = axes.ravel()
        for idx in range(len(flat_axes)):
            ax = flat_axes[idx]
            if idx < n_neurons:
                ax.hist(activations[:, idx], bins=bins, color="steelblue", alpha=0.8)
                ax.set_title(f"Layer {layer_index} Neuron {idx}")
                ax.set_xlabel("Activation")
            else:
                ax.axis("off")
        fig.tight_layout()
        return fig, axes

    def plot_activation_heatmap(self, X: ArrayLike, layer_index: int, max_samples: int = 256):
        """Heatmap of activations (samples × neurons)."""

        activations = self.get_layer_activations(X, layer_index)
        if activations.shape[0] > max_samples:
            activations = activations[:max_samples]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(activations, aspect="auto", cmap="viridis")
        ax.set_title(f"Layer {layer_index} Activation Heatmap")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Sample")
        fig.colorbar(im, ax=ax, label="Activation")
        fig.tight_layout()
        return fig, ax

    def plot_weight_heatmap(self, layer_index: int):
        """Heatmap of the weight matrix feeding into ``layer_index``."""

        if layer_index <= 0 or layer_index > len(self.coefs_):
            raise IndexError("layer_index must refer to a hidden or output layer (>=1).")
        weights = self.coefs_[layer_index - 1]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(weights, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(weights)), vmax=np.max(np.abs(weights)))
        ax.set_title(f"Weights into Layer {layer_index}")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Input feature")
        fig.colorbar(im, ax=ax, label="Weight")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    def top_k_examples(self, X: ArrayLike, neuron: tuple[int, int], k: int = 12) -> np.ndarray:
        """Indices of the ``k`` samples with the largest activation for ``neuron``."""

        layer, index = neuron
        activations = self.get_layer_activations(X, layer)
        if not (0 <= index < activations.shape[1]):
            raise IndexError(f"Neuron index {index} is out of bounds for layer {layer} with size {activations.shape[1]}.")
        top_idx = np.argsort(-activations[:, index])[:k]
        return top_idx

    def neuron_stats(self, X: ArrayLike, layer_index: int, y: np.ndarray | None = None) -> dict:
        """Compute simple statistics (mean/std/sparsity) per neuron."""

        activations = self.get_layer_activations(X, layer_index)
        mean = activations.mean(axis=0)
        std = activations.std(axis=0)
        sparsity = np.mean(np.abs(activations) < 1e-6, axis=0)
        stats = {"mean": mean, "std": std, "sparsity": sparsity}
        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y_encoded = y.astype(float)
                y_encoded = (y_encoded - y_encoded.mean()) / (y_encoded.std() + 1e-9)
                corr = activations.T @ y_encoded / (activations.shape[0] - 1)
                stats["corr_y"] = corr
            else:
                raise ValueError("y must be a 1D array-like of labels.")
        return stats

    # ------------------------------------------------------------------
    def _backpropagate(
        self,
        X: ArrayLike,
        forward_pass: dict | None = None,
        target: dict | None = None,
    ) -> ArrayLike:
        if forward_pass is None:
            forward_pass = self.forward(X)
        Zs = forward_pass["Z"]
        As = forward_pass["A"]
        n_layers = len(Zs)
        W = self.coefs_
        hidden_activation = self.activation_name
        output_activation = self.out_activation_name

        grad_A = [np.zeros_like(A) for A in As]
        grad_Z = [np.zeros_like(Z) for Z in Zs]

        if target is None:
            raise ValueError("Target specification is required for gradient computation.")

        if target.get("type") == "logit":
            class_index = int(target.get("class_index", 0))
            grad_Z[-1][..., class_index] = 1.0
        elif target.get("type") == "neuron":
            layer = int(target.get("layer"))
            index = int(target.get("index"))
            if layer < 0:
                layer = len(As) - 1 + layer + 1
            if not (0 <= layer < len(As)):
                raise ValueError("Target layer is out of bounds.")
            grad_A[layer][..., index] = 1.0
        else:
            raise ValueError("Unsupported target specification.")

        for layer in reversed(range(1, n_layers + 1)):
            activation = output_activation if layer == n_layers else hidden_activation
            if activation == "softmax":
                grad_from_activation = np.zeros_like(grad_A[layer])
                for i, (a_vec, g_vec) in enumerate(zip(As[layer], grad_A[layer])):
                    if not np.any(g_vec):
                        continue
                    jac = np.diag(a_vec) - np.outer(a_vec, a_vec)
                    grad_from_activation[i] = jac @ g_vec
            else:
                grad_from_activation = grad_A[layer] * _activation_backward(activation, Zs[layer - 1], As[layer])

            if np.any(grad_from_activation):
                if np.any(grad_Z[layer - 1]):
                    grad_Z[layer - 1] += grad_from_activation
                else:
                    grad_Z[layer - 1] = grad_from_activation

            grad_A[layer - 1] += grad_Z[layer - 1] @ W[layer - 1].T

        return grad_A[0]

    def input_gradient(self, X: ArrayLike, target: dict) -> ArrayLike:
        """Gradient of a class logit or neuron activation with respect to ``X``."""

        X = _ensure_2d(X)
        forward_pass = self.forward(X)
        grad = self._backpropagate(X, forward_pass=forward_pass, target=target)
        grad_input = self._apply_backward_transforms(grad)
        return grad_input.reshape(X.shape)

    # ------------------------------------------------------------------
    def activation_maximize(
        self,
        target: dict,
        x0: ArrayLike,
        steps: int = 300,
        lr: float = 0.1,
        l2: float = 1e-3,
        tv: float = 0.0,
        clip: tuple[float, float] | None = None,
        callback: Callable[[int, ArrayLike, float], None] | None = None,
    ) -> ArrayLike:
        """Gradient ascent in input space to maximise a neuron/class logit."""

        x = _ensure_2d(x0).astype(float)
        if x.shape[0] != 1:
            raise ValueError("activation_maximize currently supports a single initial point x0.")

        for step in range(steps):
            forward_pass = self.forward(x)
            if target.get("type") == "logit":
                class_index = int(target.get("class_index", 0))
                value = forward_pass["Z"][-1][0, class_index]
            elif target.get("type") == "neuron":
                layer = int(target.get("layer"))
                index = int(target.get("index"))
                if layer < 0:
                    layer = len(forward_pass["A"]) - 1 + layer + 1
                value = forward_pass["A"][layer][0, index]
            else:
                raise ValueError("Unsupported target specification.")

            grad = self.input_gradient(x, target)
            penalty = 0.0
            grad_penalty = np.zeros_like(grad)

            if l2 > 0:
                penalty += 0.5 * l2 * np.sum(x**2)
                grad_penalty += l2 * x

            if tv > 0:
                tv_loss, tv_grad = _tv_loss_and_grad(x.ravel())
                penalty += tv * tv_loss
                grad_penalty += tv * tv_grad.reshape(x.shape)

            x = x + lr * (grad - grad_penalty)

            if clip is not None:
                x = np.clip(x, clip[0], clip[1])

            if callback is not None:
                callback(step, x.copy(), float(value - penalty))

        return x.reshape(x0.shape)

