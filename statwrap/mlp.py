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
from typing import Callable

import numpy as np

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

# Module-level constants
SPARSITY_THRESHOLD = 1e-6  # Threshold for considering activations as zero
STD_EPSILON = 1e-9  # Small constant to prevent division by zero in std calculations


def _ensure_2d(X: ArrayLike) -> ArrayLike:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


# Activation function implementations
def _logistic_forward(Z: ArrayLike) -> ArrayLike:
    return 1.0 / (1.0 + np.exp(-Z))


def _tanh_forward(Z: ArrayLike) -> ArrayLike:
    return np.tanh(Z)


def _relu_forward(Z: ArrayLike) -> ArrayLike:
    return np.maximum(0.0, Z)


def _identity_forward(Z: ArrayLike) -> ArrayLike:
    return Z


def _softmax_forward(Z: ArrayLike) -> ArrayLike:
    Z = np.asarray(Z)
    shift = Z - Z.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / exp.sum(axis=1, keepdims=True)


def _logistic_backward(Z: ArrayLike, A: ArrayLike) -> ArrayLike:
    return A * (1.0 - A)


def _tanh_backward(Z: ArrayLike, A: ArrayLike) -> ArrayLike:
    return 1.0 - A**2


def _relu_backward(Z: ArrayLike, A: ArrayLike) -> ArrayLike:
    return (Z > 0).astype(Z.dtype)


def _identity_backward(Z: ArrayLike, A: ArrayLike) -> ArrayLike:
    return np.ones_like(Z)


# Activation function dispatch dictionaries
_ACTIVATION_FORWARD = {
    "logistic": _logistic_forward,
    "tanh": _tanh_forward,
    "relu": _relu_forward,
    "identity": _identity_forward,
    "softmax": _softmax_forward,
}

_ACTIVATION_BACKWARD = {
    "logistic": _logistic_backward,
    "tanh": _tanh_backward,
    "relu": _relu_backward,
    "identity": _identity_backward,
}


def _activation_forward(name: str, Z: ArrayLike) -> ArrayLike:
    """Apply forward activation function."""
    if name not in _ACTIVATION_FORWARD:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Supported activations: {', '.join(_ACTIVATION_FORWARD.keys())}"
        )
    return _ACTIVATION_FORWARD[name](Z)


def _activation_backward(name: str, Z: ArrayLike, A: ArrayLike) -> ArrayLike:
    """Apply backward activation derivative (element-wise)."""
    if name == "softmax":
        raise ValueError("Softmax derivative requires full Jacobian; handle separately.")
    if name not in _ACTIVATION_BACKWARD:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Supported activations: {', '.join(_ACTIVATION_BACKWARD.keys())}"
        )
    return _ACTIVATION_BACKWARD[name](Z, A)


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
                raise TypeError(
                    f"Pipeline must terminate with an MLPClassifier, got {type(model[-1]).__name__}."
                )
            self._pipeline: Pipeline | None = model
            self._estimator: MLPClassifier = model[-1]
        elif isinstance(model, MLPClassifier):
            self._pipeline = None
            self._estimator = model
        else:
            raise TypeError(
                f"model must be an MLPClassifier or Pipeline ending in one, got {type(model).__name__}."
            )

        if not hasattr(self._estimator, "coefs_"):
            raise ValueError(
                "The provided MLPClassifier instance is not fitted. "
                "Call .fit(X, y) on the model before creating an MLPInspector."
            )

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
    # Helper methods
    def _normalize_layer_index(self, layer_index: int, n_total_layers: int) -> int:
        """Convert negative layer indices to positive indices.

        Parameters
        ----------
        layer_index:
            Layer index (can be negative for counting from end)
        n_total_layers:
            Total number of layers including input (len(forward_pass["A"]))

        Returns
        -------
        int
            Normalized positive layer index

        Examples
        --------
        >>> # For a network with 3 layers (input, hidden, output), n_total_layers=3
        >>> inspector._normalize_layer_index(-1, 3)  # Last layer
        2
        >>> inspector._normalize_layer_index(-2, 3)  # Second-to-last layer
        1
        >>> inspector._normalize_layer_index(1, 3)  # Already positive
        1
        """
        if layer_index < 0:
            return n_total_layers + layer_index
        return layer_index

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
    def coefs_(self) -> list[ArrayLike]:
        """Weight matrices for each layer."""
        return self._estimator.coefs_

    @property
    def intercepts_(self) -> list[ArrayLike]:
        """Bias vectors for each layer."""
        return self._estimator.intercepts_

    @property
    def activation_name(self) -> str:
        return self._estimator.activation

    @property
    def out_activation_name(self) -> str:
        return self._estimator.out_activation_

    @property
    def n_classes_(self) -> int:
        """Number of output classes."""
        return self._estimator.n_outputs_

    @property
    def input_dim_(self) -> int:
        """Number of input features."""
        return self._estimator.n_features_in_

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
    def forward(self, X: ArrayLike) -> dict:
        """Propagate inputs through the network.

        Parameters
        ----------
        X:
            Samples to propagate. They should be in the same representation as
            expected by the estimator (the inspector applies any pipeline
            transforms automatically).

        Returns
        -------
        dict
            Dictionary with keys:
            - ``'Z'``: list of pre-activation arrays for each layer
            - ``'A'``: list of activation arrays (including input as first element)
            - ``'y_pred'``: predicted class labels
            - ``'proba'``: class probabilities (if available)
        """

        X_net = self._apply_forward_transforms(X)
        W = self.coefs_
        b = self.intercepts_
        hidden_activation = self.activation_name
        output_activation = self.out_activation_name

        Zs: list[ArrayLike] = []
        As: list[ArrayLike] = [X_net]
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
    def get_layer_activations(self, X: ArrayLike, layer_index: int, pre_activation: bool = False) -> ArrayLike:
        """Return activations or pre-activations for a specific layer.

        Parameters
        ----------
        X : array-like
            Input samples.
        layer_index : int
            Layer index where ``0`` corresponds to the network input,
            ``1..L-1`` to hidden layers, and ``L`` to the output layer.
            Negative indices count from the end (e.g., ``-1`` for output layer).
        pre_activation : bool, default=False
            If False (default), returns activations after applying the layer's
            activation function (e.g., after ReLU, tanh). If True, returns
            pre-activations (linear combinations before activation function).

        Returns
        -------
        ndarray
            Array of shape ``(n_samples, n_neurons)`` containing the requested
            activations or pre-activations for the specified layer.

        Raises
        ------
        IndexError
            If ``layer_index`` is out of bounds for the network.
        ValueError
            If requesting pre-activations for the input layer (layer 0),
            which has no pre-activations.
        """

        forward_pass = self.forward(X)
        n_total_layers = len(forward_pass["A"])
        layer_index = self._normalize_layer_index(layer_index, n_total_layers)

        if not (0 <= layer_index < n_total_layers):
            raise IndexError(
                f"layer_index must be in [0, {n_total_layers - 1}] or negative for counting from end, "
                f"got {layer_index}."
            )

        if not pre_activation:
            return forward_pass["A"][layer_index]

        # Pre-activation requested
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
        random_state: int = 0,
        pre_activation: bool = False,
    ):
        """Plot a 2D projection of layer activations via PCA or t-SNE.

        Visualizes high-dimensional layer activations in 2D using dimensionality
        reduction. Useful for understanding how the network separates different
        classes in its learned representation space.

        Parameters
        ----------
        X : array-like
            Input samples to visualize.
        y : array-like, optional
            Labels for coloring points by class. If None, all points are
            colored the same.
        layer_index : int, default=-1
            Layer to visualize. Use ``-1`` for the output layer (default),
            or specify a layer index (e.g., ``1`` for first hidden layer).
        method : {'pca', 'tsne', 't-sne'}, default='pca'
            Dimensionality reduction method. ``'pca'`` for Principal Component
            Analysis, ``'tsne'`` or ``'t-sne'`` for t-SNE.
        annotate_centroids : bool, default=True
            If True and ``y`` is provided, annotates class centroids with
            their labels.
        random_state : int, default=0
            Random seed for reproducibility of the projection.
        pre_activation : bool, default=False
            If False (default), visualizes post-activation values. If True,
            visualizes pre-activation values (linear combinations before
            activation function).

        Returns
        -------
        Figure
            Matplotlib figure containing the 2D projection scatter plot.

        Raises
        ------
        ValueError
            If ``method`` is not one of the supported dimensionality
            reduction methods.
        """

        activations = self.get_layer_activations(X, layer_index, pre_activation=pre_activation)
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
            raise ValueError(
                f"method must be 'pca' or 'tsne', got '{method}'. "
                "Available methods: 'pca', 'tsne', 't-sne'."
            )

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

    def plot_activation_hist(self, X: ArrayLike, layer_index: int, bins: int = 50, pre_activation: bool = False):
        """Plot histograms of neuron activations for a given layer.

        Parameters
        ----------
        X : array-like
            Input samples to analyze.
        layer_index : int
            Layer to visualize.
        bins : int, default=50
            Number of histogram bins.
        pre_activation : bool, default=False
            If False (default), plots post-activation values. If True,
            plots pre-activation values (linear combinations before
            activation function).

        Returns
        -------
        Figure
            Matplotlib figure containing the histograms.
        """

        activations = self.get_layer_activations(X, layer_index, pre_activation=pre_activation)
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
        return fig

    def plot_activation_heatmap(self, X: ArrayLike, layer_index: int, max_samples: int = 256, pre_activation: bool = False):
        """Heatmap of activations (samples × neurons).

        Parameters
        ----------
        X : array-like
            Input samples to analyze.
        layer_index : int
            Layer to visualize.
        max_samples : int, default=256
            Maximum number of samples to include in heatmap. If X has more
            samples, only the first max_samples are shown.
        pre_activation : bool, default=False
            If False (default), plots post-activation values. If True,
            plots pre-activation values (linear combinations before
            activation function).

        Returns
        -------
        Figure
            Matplotlib figure containing the heatmap.
        """

        activations = self.get_layer_activations(X, layer_index, pre_activation=pre_activation)
        if activations.shape[0] > max_samples:
            activations = activations[:max_samples]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(activations, aspect="auto", cmap="viridis")
        ax.set_title(f"Layer {layer_index} Activation Heatmap")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Sample")
        fig.colorbar(im, ax=ax, label="Activation")
        fig.tight_layout()
        return fig

    def plot_weight_heatmap(self, layer_index: int):
        """Heatmap of the weight matrix feeding into ``layer_index``.

        Returns
        -------
        Figure
            Matplotlib figure containing the weight heatmap.
        """

        if layer_index <= 0 or layer_index > len(self.coefs_):
            raise IndexError(
                f"layer_index must refer to a hidden or output layer (>=1), got {layer_index}. "
                f"Valid range: [1, {len(self.coefs_)}]."
            )
        weights = self.coefs_[layer_index - 1]
        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = np.max(np.abs(weights))
        im = ax.imshow(weights, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Weights into Layer {layer_index}")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Input feature")
        fig.colorbar(im, ax=ax, label="Weight")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def top_k_examples(self, X: ArrayLike, neuron: tuple[int, int], k: int = 12, pre_activation: bool = False) -> np.ndarray:
        """Find samples that maximally activate a specific neuron.

        Returns the indices of the ``k`` samples from ``X`` that produce the
        highest activation values for the specified neuron. Useful for
        understanding what input patterns a neuron responds to most strongly.

        Parameters
        ----------
        X : array-like
            Input samples to analyze.
        neuron : tuple of (int, int)
            Neuron specification as ``(layer_index, neuron_index)``, where
            ``layer_index`` is the layer number and ``neuron_index`` is the
            neuron's index within that layer.
        k : int, default=12
            Number of top-activating examples to return.
        pre_activation : bool, default=False
            If False (default), ranks by post-activation values. If True,
            ranks by pre-activation values (linear combinations before
            activation function).

        Returns
        -------
        ndarray
            Array of shape ``(k,)`` containing indices into ``X`` of the
            samples with the highest activations for the specified neuron,
            sorted in descending order by activation value.

        Raises
        ------
        IndexError
            If ``neuron_index`` is out of bounds for the specified layer.
        """

        layer, index = neuron
        activations = self.get_layer_activations(X, layer, pre_activation=pre_activation)
        if not (0 <= index < activations.shape[1]):
            raise IndexError(f"Neuron index {index} is out of bounds for layer {layer} with size {activations.shape[1]}.")
        top_idx = np.argsort(-activations[:, index])[:k]
        return top_idx

    def neuron_stats(self, X: ArrayLike, layer_index: int, y: np.ndarray | None = None, pre_activation: bool = False) -> dict:
        """Compute activation statistics for each neuron in a layer.

        Analyzes neuron behavior by computing summary statistics of their
        activations across the provided samples. Optionally computes
        correlation with target labels.

        Parameters
        ----------
        X : array-like
            Input samples to compute statistics over.
        layer_index : int
            Layer to analyze.
        y : array-like, optional
            Target labels. If provided, computes correlation between each
            neuron's activations and the (standardized) labels.
        pre_activation : bool, default=False
            If False (default), computes statistics on post-activation values.
            If True, computes statistics on pre-activation values (linear
            combinations before activation function).

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'mean'`` : ndarray of shape ``(n_neurons,)``
                Mean activation value for each neuron across samples.
            - ``'std'`` : ndarray of shape ``(n_neurons,)``
                Standard deviation of activations for each neuron.
            - ``'sparsity'`` : ndarray of shape ``(n_neurons,)``
                Fraction of near-zero activations (< 1e-6) for each neuron.
            - ``'corr_y'`` : ndarray of shape ``(n_neurons,)``  (only if y provided)
                Correlation between each neuron's activations and the labels.

        Raises
        ------
        ValueError
            If ``y`` is provided but is not a 1D array of labels.
        """

        activations = self.get_layer_activations(X, layer_index, pre_activation=pre_activation)
        mean = activations.mean(axis=0)
        std = activations.std(axis=0)
        sparsity = np.mean(np.abs(activations) < SPARSITY_THRESHOLD, axis=0)
        stats = {"mean": mean, "std": std, "sparsity": sparsity}
        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y_encoded = y.astype(float)
                y_encoded = (y_encoded - y_encoded.mean()) / (y_encoded.std() + STD_EPSILON)
                corr = activations.T @ y_encoded / (activations.shape[0] - 1)
                stats["corr_y"] = corr
            else:
                raise ValueError(
                    f"y must be a 1D array-like of labels, got shape {y.shape}. "
                    "Multi-label classification is not supported."
                )
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
            layer = self._normalize_layer_index(layer, len(As))
            if not (0 <= layer < len(As)):
                raise ValueError(
                    f"Target layer must be in [0, {len(As) - 1}] or negative for counting from end, "
                    f"got {layer}."
                )
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
        """Compute gradient of a target quantity with respect to inputs.

        Uses backpropagation to compute how changes in the input affect either
        a class logit (for saliency maps) or a specific neuron's activation.
        Useful for understanding what input features matter most for predictions.

        Parameters
        ----------
        X : array-like
            Input samples. Can be a single sample (1D) or batch (2D).
        target : dict
            Target specification dictionary with the following structure:

            For class logits (saliency maps):
                ``{'type': 'logit', 'class_index': int}``
            For specific neuron activations:
                ``{'type': 'neuron', 'layer': int, 'index': int}``

        Returns
        -------
        ndarray
            Gradient array with the same shape as ``X``. Each element indicates
            how much that input feature contributes to the target quantity.
            Larger absolute values indicate more important features.

        Examples
        --------
        >>> # Compute saliency for class 0 prediction
        >>> grad = inspector.input_gradient(X[0], {'type': 'logit', 'class_index': 0})
        >>> # Compute gradient for neuron (1, 5)
        >>> grad = inspector.input_gradient(X[0], {'type': 'neuron', 'layer': 1, 'index': 5})
        """

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
        """Generate an input that maximally activates a target neuron or class.

        Performs gradient ascent in input space to find an input pattern that
        produces the highest activation for a specified neuron or class logit.
        Useful for visualizing what features a neuron or classifier has learned.

        Parameters
        ----------
        target : dict
            Target specification dictionary:

            For class logits:
                ``{'type': 'logit', 'class_index': int}``
            For specific neurons:
                ``{'type': 'neuron', 'layer': int, 'index': int}``
        x0 : array-like
            Initial input to start optimization from. Must be a 1D array or
            2D array with shape ``(1, n_features)``.
        steps : int, default=300
            Number of gradient ascent iterations.
        lr : float, default=0.1
            Learning rate for gradient ascent.
        l2 : float, default=1e-3
            L2 regularization strength to prevent unbounded values.
        tv : float, default=0.0
            Total variation regularization strength. Only applies if input
            can be reshaped to a square image. Encourages smooth inputs.
        clip : tuple of (float, float), optional
            If provided, clips input values to ``[clip[0], clip[1]]`` after
            each step to keep values in valid range.
        callback : callable, optional
            Function called after each step as ``callback(step, x, value)``
            where ``step`` is the iteration number, ``x`` is the current input,
            and ``value`` is the target activation value.

        Returns
        -------
        ndarray
            Optimized input array with the same shape as ``x0``, designed to
            maximally activate the target.

        Raises
        ------
        ValueError
            If ``x0`` contains more than one sample or if ``target`` type is
            not supported.

        Examples
        --------
        >>> # Generate input that maximizes class 1 logit
        >>> x_opt = inspector.activation_maximize(
        ...     target={'type': 'logit', 'class_index': 1},
        ...     x0=np.zeros(n_features),
        ...     steps=200,
        ...     lr=0.1,
        ...     clip=(0, 1)
        ... )
        """

        x = _ensure_2d(x0).astype(float)
        if x.shape[0] != 1:
            raise ValueError(
                f"activation_maximize currently supports a single initial point x0, "
                f"got {x.shape[0]} samples. Pass a 1D array or 2D array with shape (1, n_features)."
            )

        for step in range(steps):
            forward_pass = self.forward(x)
            if target.get("type") == "logit":
                class_index = int(target.get("class_index", 0))
                value = forward_pass["Z"][-1][0, class_index]
            elif target.get("type") == "neuron":
                layer = int(target.get("layer"))
                index = int(target.get("index"))
                layer = self._normalize_layer_index(layer, len(forward_pass["A"]))
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

