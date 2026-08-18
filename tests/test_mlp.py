import unittest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statwrap.mlp import MLPInspector


class TestMLPInspector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.binary_X, cls.binary_y = make_classification(
            n_samples=60,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
        cls.binary_model = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="tanh",
            solver="lbfgs",
            max_iter=500,
            random_state=0,
        ).fit(cls.binary_X, cls.binary_y)
        cls.binary_inspector = MLPInspector(cls.binary_model)

        cls.multi_X, cls.multi_y = make_classification(
            n_samples=75,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=1,
        )
        cls.pipeline = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(5,),
                activation="tanh",
                solver="lbfgs",
                max_iter=500,
                random_state=1,
            ),
        ).fit(cls.multi_X, cls.multi_y)
        cls.pipeline_inspector = MLPInspector(cls.pipeline)

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _class_logit(inspector, x, class_index):
        logits = inspector.forward(np.asarray(x).reshape(1, -1))["Z"][-1][0]
        if logits.size == 1:
            sign = -1.0 if class_index == 0 else 1.0
            return sign * logits[0]
        return logits[class_index]

    @classmethod
    def _finite_difference_gradient(cls, inspector, x, class_index, epsilon=1e-6):
        x = np.asarray(x, dtype=float)
        gradient = np.empty_like(x)
        for index in range(x.size):
            upper = x.copy()
            lower = x.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            gradient[index] = (
                cls._class_logit(inspector, upper, class_index)
                - cls._class_logit(inspector, lower, class_index)
            ) / (2 * epsilon)
        return gradient

    def test_metadata_matches_wrapped_binary_estimator(self):
        inspector = self.binary_inspector
        model = self.binary_model

        self.assertEqual(inspector.n_layers_, model.n_layers_)
        self.assertEqual(inspector.n_weight_layers_, len(model.coefs_))
        self.assertEqual(inspector.n_classes_, len(model.classes_))

    def test_binary_forward_matches_predictor(self):
        result = self.binary_inspector.forward(self.binary_X)
        positive_probability = result["A"][-1][:, 0]

        np.testing.assert_allclose(
            positive_probability,
            self.binary_model.predict_proba(self.binary_X)[:, 1],
        )
        np.testing.assert_array_equal(
            result["y_pred"],
            self.binary_model.predict(self.binary_X),
        )

    def test_pipeline_forward_matches_predictor(self):
        result = self.pipeline_inspector.forward(self.multi_X)

        np.testing.assert_allclose(
            result["A"][-1],
            self.pipeline.predict_proba(self.multi_X),
        )
        np.testing.assert_array_equal(
            result["y_pred"],
            self.pipeline.predict(self.multi_X),
        )

    def test_custom_preprocess_is_used_for_predictions(self):
        transformed_X = self.binary_X * 2
        model = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="tanh",
            solver="lbfgs",
            max_iter=500,
            random_state=2,
        ).fit(transformed_X, self.binary_y)
        inspector = MLPInspector(model, preprocess=lambda values: values * 2)
        result = inspector.forward(self.binary_X)

        np.testing.assert_allclose(
            result["A"][-1][:, 0],
            result["proba"][:, 1],
        )
        np.testing.assert_array_equal(
            result["y_pred"],
            model.predict(transformed_X),
        )

        with self.assertRaisesRegex(RuntimeError, "custom preprocess"):
            inspector.input_gradient(
                self.binary_X[0],
                {"type": "logit", "class_index": 1},
            )

    def test_binary_class_gradients_match_finite_differences(self):
        x = self.binary_X[0]

        for class_index in (0, 1):
            with self.subTest(class_index=class_index):
                actual = self.binary_inspector.input_gradient(
                    x,
                    {"type": "logit", "class_index": class_index},
                )
                expected = self._finite_difference_gradient(
                    self.binary_inspector,
                    x,
                    class_index,
                )
                self.assertEqual(actual.shape, x.shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

        class_zero = self.binary_inspector.input_gradient(
            x,
            {"type": "logit", "class_index": 0},
        )
        class_one = self.binary_inspector.input_gradient(
            x,
            {"type": "logit", "class_index": 1},
        )
        np.testing.assert_allclose(class_zero, -class_one)

    def test_pipeline_gradient_matches_finite_differences(self):
        x = self.multi_X[0]
        actual = self.pipeline_inspector.input_gradient(
            x,
            {"type": "logit", "class_index": 2},
        )
        expected = self._finite_difference_gradient(
            self.pipeline_inspector,
            x,
            2,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_invalid_class_index_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "class_index"):
            self.binary_inspector.input_gradient(
                self.binary_X[0],
                {"type": "logit", "class_index": 2},
            )

    def test_neuron_stats_reports_one_vs_rest_class_correlations(self):
        activations = self.pipeline_inspector.get_layer_activations(
            self.multi_X,
            layer_index=1,
        )
        stats = self.pipeline_inspector.neuron_stats(
            self.multi_X,
            layer_index=1,
            y=self.multi_y,
        )

        np.testing.assert_array_equal(stats["classes"], np.unique(self.multi_y))
        self.assertEqual(
            stats["class_correlation"].shape,
            (len(np.unique(self.multi_y)), activations.shape[1]),
        )
        self.assertTrue(np.all(np.abs(stats["class_correlation"]) <= 1))

        first_class = stats["classes"][0]
        expected = np.corrcoef(
            activations[:, 0],
            (self.multi_y == first_class).astype(float),
        )[0, 1]
        self.assertAlmostEqual(stats["class_correlation"][0, 0], expected)

    def test_neuron_stats_validates_label_length(self):
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            self.binary_inspector.neuron_stats(
                self.binary_X,
                layer_index=1,
                y=self.binary_y[:-1],
            )

    def test_projection_handles_binary_output_and_list_labels(self):
        figure = self.binary_inspector.plot_layer_projection(
            self.binary_X,
            self.binary_y.tolist(),
        )
        offsets = figure.axes[0].collections[0].get_offsets()

        self.assertEqual(offsets.shape, (len(self.binary_X), 2))

    def test_tsne_selects_a_valid_perplexity_for_small_samples(self):
        figure = self.binary_inspector.plot_layer_projection(
            self.binary_X[:12],
            self.binary_y[:12],
            layer_index=1,
            method="tsne",
            random_state=0,
        )
        offsets = figure.axes[0].collections[0].get_offsets()

        self.assertEqual(offsets.shape, (12, 2))

    def test_plotting_helpers_and_top_examples_smoke(self):
        figures = (
            self.binary_inspector.plot_activation_hist(
                self.binary_X,
                layer_index=1,
                bins=10,
            ),
            self.binary_inspector.plot_activation_heatmap(
                self.binary_X,
                layer_index=1,
                max_samples=10,
            ),
            self.binary_inspector.plot_weight_heatmap(layer_index=1),
        )
        top_indices = self.binary_inspector.top_k_examples(
            self.binary_X,
            neuron=(1, 0),
            k=5,
        )

        self.assertTrue(all(isinstance(figure, plt.Figure) for figure in figures))
        self.assertEqual(top_indices.shape, (5,))
        self.assertTrue(np.all((0 <= top_indices) & (top_indices < len(self.binary_X))))

    def test_activation_maximize_preserves_list_input_shape(self):
        result = self.binary_inspector.activation_maximize(
            target={"type": "logit", "class_index": 1},
            x0=self.binary_X[0].tolist(),
            steps=2,
            lr=0.01,
        )

        self.assertEqual(result.shape, self.binary_X[0].shape)


if __name__ == "__main__":
    unittest.main()
