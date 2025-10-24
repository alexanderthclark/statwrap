"""Demonstration script for :mod:`mlp_reps` using the digits dataset."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlp_reps import MLPInspector


def _maybe_square(n_features: int) -> tuple[int, int] | None:
    side = int(math.sqrt(n_features))
    if side * side == n_features:
        return side, side
    return None


def _plot_image_grid(X: np.ndarray, indices: np.ndarray, title: str, cmap: str = "gray") -> plt.Figure:
    images = X[indices]
    shape = _maybe_square(images.shape[1])
    rows = int(math.ceil(math.sqrt(len(indices))))
    cols = int(math.ceil(len(indices) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel()
    for idx, ax in enumerate(flat_axes):
        if idx < len(indices):
            img = images[idx]
            if shape:
                ax.imshow(img.reshape(shape), cmap=cmap)
            else:
                ax.plot(img)
            ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _plot_saliency(image: np.ndarray, gradient: np.ndarray, title: str) -> plt.Figure:
    shape = _maybe_square(image.size)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    if shape:
        axes[0].imshow(image.reshape(shape), cmap="gray")
        axes[1].imshow(gradient.reshape(shape), cmap="RdBu", vmin=-np.max(np.abs(gradient)), vmax=np.max(np.abs(gradient)))
    else:
        axes[0].plot(image)
        axes[1].plot(gradient)
    axes[0].set_title("Input")
    axes[1].set_title("Saliency")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    max_iter=300,
                    random_state=0,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    inspector = MLPInspector(pipeline)

    forward = inspector.forward(X_test)
    assert len(forward["A"]) == inspector.n_layers_ + 1
    assert len(forward["Z"]) == inspector.n_layers_
    assert forward["A"][0].shape == (X_test.shape[0], X_test.shape[1])

    hidden_acts = inspector.get_layer_activations(X_test, 1)
    assert hidden_acts.shape == (X_test.shape[0], inspector.hidden_layer_sizes[0])

    fig_proj_pca = inspector.plot_layer_projection(X_test, y_test, layer_index=-1, method="pca")
    fig_proj_tsne = inspector.plot_layer_projection(X_test, y_test, layer_index=1, method="tsne", random_state=0)

    fig_weights, _ = inspector.plot_weight_heatmap(layer_index=1)
    fig_hist, _ = inspector.plot_activation_hist(X_test, layer_index=1)
    fig_heatmap, _ = inspector.plot_activation_heatmap(X_test, layer_index=1)

    neuron = (1, 5)
    top_indices = inspector.top_k_examples(X_test, neuron=neuron, k=16)
    assert np.all((0 <= top_indices) & (top_indices < len(X_test)))
    fig_topk = _plot_image_grid(X_test, top_indices, title=f"Top activations for neuron {neuron}")

    stats = inspector.neuron_stats(X_test, layer_index=1, y=y_test)
    assert {"mean", "std", "sparsity"}.issubset(stats.keys())

    sample_index = top_indices[0]
    target_class = int(y_test[sample_index])
    grad = inspector.input_gradient(X_test[sample_index : sample_index + 1], {"type": "logit", "class_index": target_class})
    assert grad.shape == (1, X_test.shape[1])
    assert np.all(np.isfinite(grad))
    fig_saliency = _plot_saliency(X_test[sample_index], grad[0], title=f"Saliency for class {target_class}")

    x0 = np.random.normal(0.0, 1.0, size=X_test.shape[1])

    def _callback(step: int, sample: np.ndarray, value: float) -> None:
        if step % 50 == 0:
            print(f"step={step:03d} value={value:.3f}")

    optimized = inspector.activation_maximize(
        target={"type": "logit", "class_index": target_class},
        x0=x0,
        steps=150,
        lr=0.2,
        l2=1e-2,
        tv=1e-2,
        clip=(0.0, 16.0),
        callback=_callback,
    )

    initial_value = inspector.forward(x0.reshape(1, -1))["Z"][-1][0, target_class]
    final_value = inspector.forward(optimized.reshape(1, -1))["Z"][-1][0, target_class]
    assert final_value > initial_value, "Activation maximisation did not improve the target logit."

    fig_activation = _plot_saliency(optimized, optimized - x0, title="Activation maximisation result")

    print("Forward pass A layers:", len(forward["A"]))
    print("Projection figures:", fig_proj_pca, fig_proj_tsne)
    print("Top-k indices:", top_indices[:5])
    print("Initial vs. final target logit:", float(initial_value), float(final_value))

    plt.show()


if __name__ == "__main__":
    main()
