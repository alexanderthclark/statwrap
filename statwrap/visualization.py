import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

class LossSurfacePlotter:
    """Visualize loss surfaces for two-coefficient regression models"""

    def __init__(self, model, X, y, loss_range=2.0, grid_size=50, refit=False):
        self.model = model
        self.loss_range = loss_range
        self.grid_size = grid_size

        # Ensure we have exactly 2 features
        if X.shape[1] != 2:
            raise ValueError("Loss surface visualization requires exactly 2 features")

        # Demean X and y to make intercept ~zero
        self.X_centered = X - X.mean(axis=0)
        self.y_centered = y - y.mean()

        # Fit the model on centered data
        if not hasattr(model, 'coef_'):
            model.fit(self.X_centered, self.y_centered)
        elif refit:
            # Re-fit on centered data even if already fitted
            model.fit(self.X_centered, self.y_centered)

    def plot(self, plot_type='contour', ax=None):
        """Plot the loss surface over two slope coefficients"""
        # Get optimal coefficients (intercept should be ~0 after centering)
        w1_opt, w2_opt = self.model.coef_

        # Create coefficient grid
        w1_range = np.linspace(w1_opt - self.loss_range, w1_opt + self.loss_range, self.grid_size)
        w2_range = np.linspace(w2_opt - self.loss_range, w2_opt + self.loss_range, self.grid_size)
        W1, W2 = np.meshgrid(w1_range, w2_range)

        # Calculate loss surface (MSE)
        def mse_loss(w1, w2):
            y_pred = w1 * self.X_centered[:, 0] + w2 * self.X_centered[:, 1]
            return np.mean((self.y_centered - y_pred) ** 2)

        Z = np.array([[mse_loss(w1, w2) for w1 in w1_range] for w2 in w2_range])

        # Plot
        if plot_type == '3d':
            from mpl_toolkits.mplot3d import Axes3D
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(W1, W2, Z, alpha=0.7, cmap='viridis')
            optimal_loss = mse_loss(w1_opt, w2_opt)
            ax.scatter(w1_opt, w2_opt, optimal_loss, color='red', s=100)
            ax.set_zlabel('MSE Loss')
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contour(W1, W2, Z, levels=20)
            ax.clabel(contour, inline=True, fontsize=8)
            ax.scatter(w1_opt, w2_opt, color='red', s=100, marker='*',
                       label=f'Optimal: ({w1_opt:.2f}, {w2_opt:.2f})')
            ax.legend()

        ax.set_xlabel('Weight 1 (Feature 1 Coefficient)')
        ax.set_ylabel('Weight 2 (Feature 2 Coefficient)')
        ax.set_title(f'{type(self.model).__name__} Loss Surface')

        return ax

    def compare_models(self, other_models, labels=None, ax=None):
        """Compare multiple models on the same loss surface"""
        all_models = [self.model] + list(other_models)
        if labels is None:
            labels = [type(m).__name__ for m in all_models]

        # Fit all models on centered data
        for model in other_models:
            model.fit(self.X_centered, self.y_centered)

        # Get coefficient ranges from all models
        all_coefs = np.array([m.coef_ for m in all_models])
        w1_min, w1_max = all_coefs[:, 0].min() - self.loss_range, all_coefs[:, 0].max() + self.loss_range
        w2_min, w2_max = all_coefs[:, 1].min() - self.loss_range, all_coefs[:, 1].max() + self.loss_range

        w1_range = np.linspace(w1_min, w1_max, self.grid_size)
        w2_range = np.linspace(w2_min, w2_max, self.grid_size)
        W1, W2 = np.meshgrid(w1_range, w2_range)

        # Calculate loss surface
        def mse_loss(w1, w2):
            y_pred = w1 * self.X_centered[:, 0] + w2 * self.X_centered[:, 1]
            return np.mean((self.y_centered - y_pred) ** 2)

        Z = np.array([[mse_loss(w1, w2) for w1 in w1_range] for w2 in w2_range])

        # Plot with all model solutions
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contour(W1, W2, Z, levels=20, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model, label) in enumerate(zip(all_models, labels)):
            w1, w2 = model.coef_
            ax.scatter(w1, w2, color=colors[i % len(colors)], s=100,
                       label=f'{label}: ({w1:.2f}, {w2:.2f})')

        ax.set_xlabel('Weight 1 (Feature 1 Coefficient)')
        ax.set_ylabel('Weight 2 (Feature 2 Coefficient)')
        ax.set_title('Model Comparison on Loss Surface')
        ax.legend()

        return ax


    def plot_ridge_path(self, alphas=None, show_loss_surface=True, ax=None):
        """Show Ridge regularization path on loss surface"""
        if alphas is None:
            alphas = np.logspace(-3, 2, 20)  # 0.001 to 100

        ridge_models = []
        coefficients = []

        # Fit Ridge for different alpha values
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(self.X_centered, self.y_centered)
            ridge_models.append(ridge)
            coefficients.append(ridge.coef_)

        coefficients = np.array(coefficients)

        if show_loss_surface:
            # Plot on loss surface
            self._plot_path_on_surface(coefficients, alphas, "Ridge Regularization Path")

        # Plot coefficient paths vs alpha
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(alphas, coefficients[:, 0], 'b-o', label='Weight 1', markersize=4)
        ax.semilogx(alphas, coefficients[:, 1], 'r-s', label='Weight 2', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Regularization Strength (α)')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Ridge Coefficient Path')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_lasso_path(self, alphas=None, show_loss_surface=True, ax=None):
        """Show Lasso regularization path on loss surface"""
        if alphas is None:
            alphas = np.logspace(-3, 1, 20)  # 0.001 to 10

        lasso_models = []
        coefficients = []

        # Fit Lasso for different alpha values
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=2000)
            lasso.fit(self.X_centered, self.y_centered)
            lasso_models.append(lasso)
            coefficients.append(lasso.coef_)

        coefficients = np.array(coefficients)

        if show_loss_surface:
            # Plot on loss surface
            self._plot_path_on_surface(coefficients, alphas, "Lasso Regularization Path")

        # Plot coefficient paths vs alpha
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(alphas, coefficients[:, 0], 'b-o', label='Weight 1', markersize=4)
        ax.semilogx(alphas, coefficients[:, 1], 'r-s', label='Weight 2', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Regularization Strength (α)')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Lasso Coefficient Path')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def _plot_path_on_surface(self, coefficients, alphas, title, ax=None):
        """Helper method to plot regularization path on loss surface"""
        # Calculate loss surface
        w1_min, w1_max = coefficients[:, 0].min() - 0.5, coefficients[:, 0].max() + 0.5
        w2_min, w2_max = coefficients[:, 1].min() - 0.5, coefficients[:, 1].max() + 0.5

        w1_range = np.linspace(w1_min, w1_max, self.grid_size)
        w2_range = np.linspace(w2_min, w2_max, self.grid_size)
        W1, W2 = np.meshgrid(w1_range, w2_range)

        def mse_loss(w1, w2):
            y_pred = w1 * self.X_centered[:, 0] + w2 * self.X_centered[:, 1]
            return np.mean((self.y_centered - y_pred) ** 2)

        Z = np.array([[mse_loss(w1, w2) for w1 in w1_range] for w2 in w2_range])

        # Plot loss surface with path
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contour(W1, W2, Z, levels=15, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        # Plot the regularization path
        ax.plot(coefficients[:, 0], coefficients[:, 1], 'ro-',
                linewidth=2, markersize=6, alpha=0.8, label='Regularization Path')

        # Mark OLS solution (alpha=0, should be first point)
        ax.scatter(coefficients[0, 0], coefficients[0, 1],
                   color='green', s=100, marker='*', label='OLS Solution')

        # Mark high regularization endpoint
        ax.scatter(coefficients[-1, 0], coefficients[-1, 1],
                   color='red', s=100, marker='s', label=f'α={alphas[-1]:.3f}')

        ax.set_xlabel('Weight 1 (Feature 1 Coefficient)')
        ax.set_ylabel('Weight 2 (Feature 2 Coefficient)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax