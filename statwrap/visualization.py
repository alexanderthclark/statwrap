import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import clone


class LossSurface:
    """
    Visualize (unregularized) MSE loss surfaces for two-coefficient linear models.

    Key idea: the surface plotted is the cross-section of the MSE where the intercept
    has been optimized out (i.e., intercept = ȳ - wᵀ x̄). This is equivalent to
    computing MSE on centered data (X - x̄, y - ȳ), regardless of whether the
    original estimator used an intercept. If the provided model is already fitted,
    we *inherit its fit_intercept semantics and coefficients* for the overlay point.

    Parameters
    ----------
    model : sklearn estimator (LinearRegression, Ridge, Lasso, etc.)
        A fitted or unfitted estimator with exactly 2 coefficients.
        If unfitted, a cloned copy will be fit on (X, y) to obtain the base point.
        If already fitted, we do not refit and we reuse its coefficients/intercept.
    X : array-like, shape (n_samples, 2)
    y : array-like, shape (n_samples,)
    loss_range : float
        Half-width of the coefficient window around the base point.
    grid_size : int
        Resolution of the (w1, w2) grid.

    Notes
    -----
    - The contour/surface is always the *unregularized* MSE with the intercept
      optimized out (i.e., centered-data MSE). Points/paths from Ridge/Lasso are
      overlaid on this surface.
    - No mutation of the passed-in model: if refitting is needed, a clone is used.
    """

    # ------------------------- initialization -------------------------

    def __init__(self, model, X, y, loss_range=3.0, grid_size=50):
        # Coerce inputs
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be of shape (n_samples, 2)")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples: {X.shape[0]} vs {y.shape[0]}")

        self.loss_range = float(loss_range)
        self.grid_size = int(grid_size)

        # Store centered data for surface calculations (intercept optimized out)
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        self.X_centered = X - self.x_mean_
        self.y_centered = y - self.y_mean_

        # Keep original (uncentered) too, for fitting models in their native semantics
        self.X = X
        self.y = y

        # Determine whether model is already fitted
        self.original_model = model
        self.model = None  # fitted base model used for overlay/labels
        self.prefit_ = hasattr(model, "coef_")

        if self.prefit_:
            coef = np.asarray(model.coef_)
            if coef.ndim != 1 or coef.size != 2:
                raise ValueError(f"Prefit model must have exactly 2 coefficients; got {coef.shape}")
            # Reuse the fitted model directly (no mutation)
            self.model = model
        else:
            # Fit a clone on (X, y) with its own intercept semantics preserved
            self.model = clone(model)
            self.model.fit(self.X, self.y)
            coef = np.asarray(self.model.coef_)
            if coef.ndim != 1 or coef.size != 2:
                raise ValueError(f"Model must yield exactly 2 coefficients after fit; got {coef.shape}")

        # Base point (overlay star and grid center)
        self.w_opt_ = np.asarray(self.model.coef_, dtype=float).reshape(2)
        # In case caller wants it:
        self.intercept_opt_ = float(getattr(self.model, "intercept_", 0.0))

        # Cache fit_intercept preference (used for paths); inherit if available
        self.fit_intercept_pref_ = self._get_fit_intercept_pref(self.model)

    # ------------------------- utilities -------------------------

    @staticmethod
    def _get_fit_intercept_pref(est):
        """
        Extract fit_intercept preference from estimator, default True.

        Parameters
        ----------
        est : sklearn estimator
            Fitted or unfitted estimator.

        Returns
        -------
        bool
            The fit_intercept preference, defaulting to True.
        """
        try:
            params = est.get_params(deep=False)
            if "fit_intercept" in params:
                return bool(params["fit_intercept"])
        except Exception:
            pass
        return bool(getattr(est, "fit_intercept", True))

    @staticmethod
    def _coef2(est):
        """
        Extract and validate 2-coefficient array from fitted estimator.

        Parameters
        ----------
        est : sklearn estimator
            Fitted estimator with coef_ attribute.

        Returns
        -------
        ndarray, shape (2,)
            The coefficient array.

        Raises
        ------
        ValueError
            If estimator doesn't have exactly 2 coefficients.
        """
        c = np.asarray(getattr(est, "coef_", None))
        if c is None or c.ndim != 1 or c.size != 2:
            raise ValueError(f"Estimator must have coef_ with shape (2,), got {None if c is None else c.shape}")
        return c

    def _ensure_alphas(self, alphas, default):
        """
        Return a validated, ascending 1D float array of positive alphas.

        Parameters
        ----------
        alphas : array-like or None
            Candidate alpha values. If None, uses `default`.
        default : array-like
            Default alpha values to use when `alphas` is None.

        Returns
        -------
        ndarray
            1-D, strictly positive, finite, ascending array of alphas.

        Raises
        ------
        ValueError
            If alphas are not 1-D positive finite numbers.
        """
        if alphas is None:
            alphas = np.asarray(default, dtype=float)
        else:
            alphas = np.asarray(alphas, dtype=float)
        if alphas.ndim != 1 or not np.all(np.isfinite(alphas)) or np.any(alphas <= 0):
            raise ValueError("alphas must be a 1-D array of positive, finite values")
        if np.any(np.diff(alphas) < 0):
            alphas = np.sort(alphas)
        return alphas

    def evaluate_loss(self, coefficients, intercept=None):
        """
        Evaluate MSE loss at specific coefficient and intercept values.

        Parameters
        ----------
        coefficients : array-like, shape (2,)
            The two coefficient values to evaluate.
        intercept : float, optional
            The intercept value. If None, uses the model's intercept.

        Returns
        -------
        float
            The mean squared error at the given coefficients and intercept.

        Raises
        ------
        ValueError
            If coefficients is not a 2-element array.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size != 2:
            raise ValueError(f"coefficients must be a 2-element array, got shape {coefficients.shape}")

        if intercept is None:
            intercept = self.intercept_opt_

        # Compute predictions: y_pred = X @ coef + intercept
        y_pred = self.X @ coefficients + intercept

        # Compute MSE: mean((y - y_pred)^2)
        mse = np.mean((self.y - y_pred) ** 2)

        return float(mse)

    def _mse_grid(self, w1_range, w2_range):
        """
        Compute vectorized MSE grid with intercept optimized out.

        Parameters
        ----------
        w1_range : array-like
            Values for first coefficient.
        w2_range : array-like
            Values for second coefficient.

        Returns
        -------
        tuple of ndarray
            (W1, W2, Z) where W1, W2 are meshgrids and Z is the MSE surface.
        """
        W1, W2 = np.meshgrid(w1_range, w2_range)
        W = np.stack([W1.ravel(), W2.ravel()], axis=0)            # (2, M)
        Yhat = self.X_centered @ W                                # (n, M)
        R = self.y_centered[:, None] - Yhat                       # (n, M)
        Z = np.mean(R * R, axis=0).reshape(W1.shape)              # (g, g)
        return W1, W2, Z

    # ------------------------- core plots -------------------------

    def plot(self, plot_type='contour', ax=None, square=True, grid_size=None, loss_range=None):
        """
        Plot MSE surface with base model solution overlay.

        Parameters
        ----------
        plot_type : {'contour', '3d'}, optional
            Type of plot to create. Default is 'contour'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        square : bool, optional
            If True and plot_type is 'contour', set equal aspect for square axes.
        grid_size : int, optional
            Resolution of the coefficient grid to use for this call. Defaults to the
            value provided at initialization.
        loss_range : float, optional
            Half-width of the coefficient window around the base point. Defaults to the
            value provided at initialization.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        w1_opt, w2_opt = self.w_opt_

        if grid_size is None:
            grid_size = self.grid_size
        else:
            grid_size = int(grid_size)
            if grid_size < 2:
                raise ValueError("grid_size must be an integer >= 2")

        if loss_range is None:
            loss_range = self.loss_range
        else:
            loss_range = float(loss_range)
            if loss_range <= 0:
                raise ValueError("loss_range must be a positive number")

        w1_range = np.linspace(w1_opt - loss_range, w1_opt + loss_range, grid_size)
        w2_range = np.linspace(w2_opt - loss_range, w2_opt + loss_range, grid_size)
        W1, W2, Z = self._mse_grid(w1_range, w2_range)

        if plot_type == '3d':
            # (Axes3D import is not required on modern Matplotlib; projection='3d' suffices)
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(W1, W2, Z, alpha=0.7, cmap='viridis')
            # Loss at base coefficients on the same cross-section
            base_loss = np.mean((self.y_centered - (w1_opt*self.X_centered[:, 0] + w2_opt*self.X_centered[:, 1]))**2)
            ax.scatter(w1_opt, w2_opt, base_loss, color='red', s=100)
            ax.set_zlabel('MSE (intercept optimized)')
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            cs = ax.contour(W1, W2, Z, levels=20)
            try:
                ax.clabel(cs, inline=True, fontsize=8)
            except Exception:
                pass
            ax.scatter(w1_opt, w2_opt, color='red', s=100, marker='*',
                       label=f'Base: ({w1_opt:.2f}, {w2_opt:.2f})')
            ax.legend()

        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_title(f'{type(self.model).__name__} on Unregularized MSE Surface (intercept optimized)')
        if square and plot_type != '3d':
            ax.set_aspect('equal', adjustable='box')
        return ax

    def compare_models(self, other_models, labels=None, ax=None):
        """
        Compare multiple model solutions on the same MSE surface.

        Parameters
        ----------
        other_models : list of sklearn estimators
            Additional models to fit and compare.
        labels : list of str, optional
            Labels for the models. If None, uses class names.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        # Tolerate None
        other_models = [] if other_models is None else list(other_models)

        # Base (already fitted or cloned+fitted)
        fitted = [self.model]

        # Clone and fit others on *original* (X, y) to respect their own intercept semantics
        for m in other_models:
            mm = clone(m)
            mm.fit(self.X, self.y)
            _ = self._coef2(mm)
            fitted.append(mm)

        # Labels
        if labels is None:
            labels = [type(m).__name__ for m in fitted]
        if len(labels) != len(fitted):
            raise ValueError("labels length must equal number of models (base + others)")

        # Build grid bounds from all coefs
        all_coefs = np.vstack([self._coef2(m) for m in fitted])
        w1_min = all_coefs[:, 0].min() - self.loss_range
        w1_max = all_coefs[:, 0].max() + self.loss_range
        w2_min = all_coefs[:, 1].min() - self.loss_range
        w2_max = all_coefs[:, 1].max() + self.loss_range

        w1_range = np.linspace(w1_min, w1_max, self.grid_size)
        w2_range = np.linspace(w2_min, w2_max, self.grid_size)
        W1, W2, Z = self._mse_grid(w1_range, w2_range)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        cs = ax.contour(W1, W2, Z, levels=20, alpha=0.6)
        try:
            ax.clabel(cs, inline=True, fontsize=8)
        except Exception:
            pass

        for m, lab in zip(fitted, labels):
            w = self._coef2(m)
            ax.scatter(w[0], w[1], s=100, label=f'{lab}: ({w[0]:.2f}, {w[1]:.2f})')

        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_title('Model Solutions on Unregularized MSE Surface (intercept optimized)')
        ax.legend()
        return ax

    # ------------------------- regularization paths -------------------------

    def plot_ridge_path_on_surface(self, alphas=None, ax=None):
        """Overlay the Ridge regularization path on the unregularized MSE surface."""
        alphas = self._ensure_alphas(alphas, np.logspace(-3, 2, 20))
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        fit_intercept = self.fit_intercept_pref_

        # OLS baseline
        ols = LinearRegression(fit_intercept=fit_intercept)
        ols.fit(self.X, self.y)
        coefs = [self._coef2(ols)]

        rid = Ridge(alpha=alphas[0], fit_intercept=fit_intercept)
        rid.fit(self.X, self.y)
        path = [self._coef2(rid)]
        for a in alphas[1:]:
            rid.set_params(alpha=a)
            rid.fit(self.X, self.y)
            path.append(self._coef2(rid))

        coefficients = np.vstack([coefs[0], np.vstack(path)])
        self._plot_path_on_surface(coefficients,
                                   alphas=np.r_[0.0, alphas],
                                   title="Ridge Regularization Path",
                                   ax=ax)
        return ax

    def plot_ridge_coef_path(self, alphas=None, ax=None):
        """
        Plot Ridge coefficients vs α (semilog x), including the OLS point at α≈0.

        Parameters
        ----------
        alphas : array-like, optional
            Regularization strengths. If None, uses default range.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        alphas = self._ensure_alphas(alphas, np.logspace(-3, 2, 20))

        fit_intercept = self.fit_intercept_pref_
        # OLS baseline
        ols = LinearRegression(fit_intercept=fit_intercept)
        ols.fit(self.X, self.y)
        start = [self._coef2(ols)]

        rid = Ridge(alpha=alphas[0], fit_intercept=fit_intercept)
        rid.fit(self.X, self.y)
        path = [self._coef2(rid)]
        for a in alphas[1:]:
            rid.set_params(alpha=a)
            rid.fit(self.X, self.y)
            path.append(self._coef2(rid))
        coefficients = np.vstack([start[0], np.vstack(path)])

        # Plot vs alpha (semilog x); place OLS at small epsilon
        eps = float(alphas[0]) * 0.1
        x_alphas = np.r_[eps, alphas]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(x_alphas, coefficients[:, 0], 'o-', label='Weight 1', markersize=4)
        ax.semilogx(x_alphas, coefficients[:, 1], 's-', label='Weight 2', markersize=4)
        ax.axvline(eps, linestyle=':', alpha=0.3)  # indicates OLS location
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('α')
        ax.set_ylabel('Coefficient')
        ax.set_title('Ridge Coefficient Path (includes OLS at α≈0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def _lasso_path_coeffs(self, alphas):
        """
        Compute Lasso path coefficients including the OLS baseline (α=0) as the first row.
        Respects this instance's fit_intercept preference. Uses a single warm-started
        Lasso instance for stability along the path.

        Parameters
        ----------
        alphas : array-like of positive floats

        Returns
        -------
        np.ndarray, shape (len(alphas)+1, 2)
            Rows are [OLS, Lasso(α1), Lasso(α2), ...]
        """
        alphas = self._ensure_alphas(alphas, np.logspace(-3, 1, 20))

        fit_intercept = self.fit_intercept_pref_

        # OLS baseline (α=0)
        ols = LinearRegression(fit_intercept=fit_intercept)
        ols.fit(self.X, self.y)
        coeffs = [self._coef2(ols)]

        las = Lasso(alpha=alphas[0], fit_intercept=fit_intercept, max_iter=10000)
        las.fit(self.X, self.y)
        coeffs.append(self._coef2(las))
        for a in alphas[1:]:
            las.set_params(alpha=a)
            las.fit(self.X, self.y)
            coeffs.append(self._coef2(las))

        return np.vstack(coeffs)

    def plot_lasso_path_on_surface(self, alphas=None, ax=None):
        """
        Overlay the Lasso regularization path on the unregularized MSE surface
        (intercept optimized). Includes the OLS point (α=0) at the start.
        """
        alphas = self._ensure_alphas(alphas, np.logspace(-3, 1, 20))
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        coefficients = self._lasso_path_coeffs(alphas)
        self._plot_path_on_surface(
            coefficients=coefficients,
            alphas=np.r_[0.0, alphas],
            title="Lasso Regularization Path",
            ax=ax
        )
        return ax

    def plot_lasso_coef_path(self, alphas=None, ax=None):
        """
        Plot Lasso coefficients vs α (semilog x), including the OLS point at α≈0.
        """
        alphas = self._ensure_alphas(alphas, np.logspace(-3, 1, 20))

        coefficients = self._lasso_path_coeffs(alphas)

        # Place OLS at a small positive epsilon so it appears on a log x-axis
        eps = float(alphas[0]) * 0.1
        x_alphas = np.r_[eps, alphas]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(x_alphas, coefficients[:, 0], 'o-', label='Weight 1', markersize=4)
        ax.semilogx(x_alphas, coefficients[:, 1], 's-', label='Weight 2', markersize=4)
        ax.axvline(eps, linestyle=':', alpha=0.3)  # indicates OLS location
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('α')
        ax.set_ylabel('Coefficient')
        ax.set_title('Lasso Coefficient Path (includes OLS at α≈0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------- helper for path overlay -------------------------

    def _plot_path_on_surface(self, coefficients, alphas, title, ax=None):
        """
        Overlay coefficient path on MSE contour plot.

        Parameters
        ----------
        coefficients : array-like, shape (n_points, 2)
            Coefficient values along the regularization path.
        alphas : array-like, shape (n_points,)
            Regularization parameter values.
        title : str
            Title for the plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        w1_min = coefficients[:, 0].min() - 0.5
        w1_max = coefficients[:, 0].max() + 0.5
        w2_min = coefficients[:, 1].min() - 0.5
        w2_max = coefficients[:, 1].max() + 0.5

        w1_range = np.linspace(w1_min, w1_max, self.grid_size)
        w2_range = np.linspace(w2_min, w2_max, self.grid_size)
        W1, W2, Z = self._mse_grid(w1_range, w2_range)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        cs = ax.contour(W1, W2, Z, levels=15, alpha=0.6)
        try:
            ax.clabel(cs, inline=True, fontsize=8)
        except Exception:
            pass

        ax.plot(coefficients[:, 0], coefficients[:, 1], 'r-', marker='.', linewidth=2, alpha=0.85, label='Path')
        ax.scatter(coefficients[0, 0], coefficients[0, 1], color='green', s=100, marker='*',
                   label=f'Start (α={alphas[0]:.3g})')
        ax.scatter(coefficients[-1, 0], coefficients[-1, 1], color='red', s=100, marker='s',
                   label=f'End (α={alphas[-1]:.3g})')

        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_title(f'{title} on Unregularized MSE Surface (intercept optimized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
