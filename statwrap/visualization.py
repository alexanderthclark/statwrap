"""Matplotlib tools for teaching two-coefficient least squares and regularization."""

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} number (finite)")
    return value


def _grid_size(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("grid_size must be an integer >= 2")
    try:
        integer = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("grid_size must be an integer >= 2") from exc
    if integer != value or integer < 2:
        raise ValueError("grid_size must be an integer >= 2")
    return integer


def _plot_ball(radius, ax, penalty, kwargs):
    radius = _positive(radius, "radius", allow_zero=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    if penalty == "l1":
        points = radius * np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    else:
        theta = np.linspace(0, 2 * np.pi, 401)
        points = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    style = {
        "color": "#444444",
        "linewidth": 2,
        "zorder": 3,
        "label": f"{penalty.upper()} boundary (radius={radius:.3g})",
    }
    if radius == 0:
        style["marker"] = "o"
    # Let Matplotlib aliases (c/lw/ls) override our defaults without conflicts.
    for alias, canonical in (("c", "color"), ("lw", "linewidth")):
        if alias in kwargs:
            style.pop(canonical, None)
    style.update(kwargs)
    ax.plot(points[:, 0], points[:, 1], **style)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_l1_ball(radius, ax=None, **kwargs):
    """Draw the boundary ``|w1| + |w2| = radius`` and return the axes.

    Parameters
    ----------
    radius : float
        Nonnegative L1 norm bound, in coefficient units (not alpha).
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when omitted.
    **kwargs
        Matplotlib line styling, such as ``color``, ``lw``, and ``label``.
    """
    return _plot_ball(radius, ax, "l1", kwargs)


def plot_l2_ball(radius, ax=None, **kwargs):
    """Draw the boundary ``sqrt(w1**2 + w2**2) = radius`` and return the axes.

    Parameters
    ----------
    radius : float
        Nonnegative Euclidean radius (not its square, and not alpha).
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when omitted.
    **kwargs
        Matplotlib line styling, such as ``color``, ``lw``, and ``label``.
    """
    return _plot_ball(radius, ax, "l2", kwargs)


@dataclass
class RegularizationPath:
    """Numerical results shared by the coefficient and contour plots.

    Attributes
    ----------
    kind : str
        ``'ridge'`` or ``'lasso'``.
    alphas : ndarray, shape (n_points,)
        Ascending scikit-learn alpha values, with zero for the optional OLS row.
    coefficients : ndarray, shape (n_points, 2)
        Fitted coefficients at each alpha.
    intercepts : ndarray, shape (n_points,)
        Fitted intercepts (zero for models without an intercept).
    mse : ndarray, shape (n_points,)
        Unpenalized training mean squared errors.
    """

    kind: str
    alphas: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    mse: np.ndarray


class LossSurface:
    """Teach least-squares geometry with two coefficients.

    Parameters
    ----------
    model : sklearn estimator
        Fitted or unfitted single-output linear estimator. Unfitted estimators
        are cloned before fitting; supplied estimators are never modified.
    X : array-like, shape (n_samples, 2)
        Two predictors. Scaling is the caller's choice; no standardization is
        performed. DataFrame column names become axis labels by default.
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Response values.
    coefficient_range : float, default=3
        Half-width around the base model's coefficients, in coefficient units.
    grid_size : int, default=50
        Number of grid points on each axis, at least two.
    feature_names : sequence of two strings, optional
        Custom coefficient labels.
    loss_range : float, optional
        Deprecated alias for ``coefficient_range`` used by older notebooks.

    Notes
    -----
    Contours show **unregularized MSE**. For models with ``fit_intercept=True``,
    the intercept is optimized separately at every point, equivalently using
    centered X and y. With ``fit_intercept=False``, the intercept is fixed at
    zero, so surfaces and paths use the same objective. This corrects the older
    behavior that always centered the surface even for no-intercept models.

    The base marker need not be the OLS minimum (for example, when model is
    Ridge). ``w_ols_`` and ``minimum_loss_`` describe the unregularized minimum.
    A two-by-two Gram matrix evaluates grids without an n_samples-by-grid
    prediction array, including when the design is rank deficient.
    """

    def __init__(
        self,
        model,
        X,
        y,
        coefficient_range=3.0,
        grid_size=50,
        *,
        feature_names=None,
        loss_range=None,
    ):
        if feature_names is None:
            feature_names = getattr(X, "columns", ("Weight 1", "Weight 2"))
        if isinstance(feature_names, str) or len(feature_names) != 2:
            raise ValueError("feature_names must contain exactly two labels")
        self.feature_names = tuple(str(name) for name in feature_names)
        X = np.array(X, dtype=float, copy=True)
        y = np.array(y, dtype=float, copy=True)
        if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
            raise ValueError(
                "X must be of shape (n_samples, 2) with at least one sample"
            )
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError("y must have shape (n_samples,) or (n_samples, 1)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite numbers")
        self.coefficient_range = _positive(coefficient_range, "coefficient_range")
        if loss_range is not None:
            if coefficient_range != 3.0:
                raise ValueError("Use only one of coefficient_range and loss_range")
            self.coefficient_range = self._resolve_coefficient_range(None, loss_range)
        self.grid_size = _grid_size(grid_size)
        self.X, self.y = X, y
        self.x_mean_, self.y_mean_ = X.mean(axis=0), y.mean()
        self.X_centered, self.y_centered = X - self.x_mean_, y - self.y_mean_
        self.original_model = model
        self.prefit_ = hasattr(model, "coef_")
        self.model = model if self.prefit_ else clone(model).fit(X.copy(), y.copy())
        self.w_opt_ = self._coef2(self.model).copy()
        intercept = np.asarray(getattr(self.model, "intercept_", 0.0))
        if intercept.size != 1 or not np.all(np.isfinite(intercept)):
            raise ValueError("Model must have one finite intercept")
        self.intercept_opt_ = float(intercept.reshape(-1)[0])
        self.fit_intercept_pref_ = self._get_fit_intercept_pref(self.model)
        self._X_loss = self.X_centered if self.fit_intercept_pref_ else self.X
        self._y_loss = self.y_centered if self.fit_intercept_pref_ else self.y
        self.ols_model_ = LinearRegression(fit_intercept=self.fit_intercept_pref_).fit(
            X, y
        )
        self.w_ols_ = self._coef2(self.ols_model_).copy()
        self.gram_ = self._X_loss.T @ self._X_loss / len(y)
        residual = self._y_loss - self._X_loss @ self.w_ols_
        self.minimum_loss_ = float(np.mean(residual**2))

    @staticmethod
    def _get_fit_intercept_pref(est):
        params = est.get_params(deep=False)
        return bool(params.get("fit_intercept", getattr(est, "fit_intercept", True)))

    @staticmethod
    def _coef2(est):
        coef = np.asarray(getattr(est, "coef_", None), dtype=float)
        if coef.shape == (1, 2):
            coef = coef[0]
        if coef.shape != (2,) or not np.all(np.isfinite(coef)):
            raise ValueError(
                f"Estimator must have two finite coefficients; got {coef.shape}"
            )
        return coef

    @staticmethod
    def _kind(kind):
        if kind not in ("ridge", "lasso"):
            raise ValueError("kind must be 'ridge' or 'lasso'")
        return kind

    def _ensure_alphas(self, alphas, default):
        alphas = np.asarray(default if alphas is None else alphas, dtype=float)
        if (
            alphas.ndim != 1
            or alphas.size == 0
            or not np.all(np.isfinite(alphas))
            or np.any(alphas <= 0)
        ):
            raise ValueError(
                "alphas must be a nonempty 1-D array of positive, finite values"
            )
        return np.unique(alphas)

    def _resolve_coefficient_range(self, coefficient_range, loss_range=None):
        if loss_range is not None:
            if coefficient_range is not None:
                raise ValueError("Use only one of coefficient_range and loss_range")
            warnings.warn(
                "loss_range is now called coefficient_range (in coefficient units)",
                DeprecationWarning,
                stacklevel=3,
            )
            coefficient_range = loss_range
        return _positive(
            self.coefficient_range if coefficient_range is None else coefficient_range,
            "coefficient_range",
        )

    @staticmethod
    def _limits(limits):
        limits = np.asarray(limits, dtype=float)
        if (
            limits.shape != (2,)
            or not np.all(np.isfinite(limits))
            or limits[0] >= limits[1]
        ):
            raise ValueError(
                "Axis limits must be two finite values in increasing order"
            )
        return tuple(limits)

    def evaluate_loss(self, coefficients, intercept=None):
        """Return MSE for two coefficients and a **fixed** intercept.

        ``intercept=None`` retains the base model's fitted intercept for backward
        compatibility. Use :meth:`surface_loss` to evaluate the plotted surface,
        whose intercept is reoptimized at each point when fitting an intercept.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (2,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be a finite 2-element array")
        intercept = self.intercept_opt_ if intercept is None else float(intercept)
        if not np.isfinite(intercept):
            raise ValueError("intercept must be finite")
        return float(np.mean((self.y - self.X @ coefficients - intercept) ** 2))

    def surface_loss(self, coefficients, relative=False):
        """Evaluate the plotted MSE at one or many coefficient pairs.

        Parameters
        ----------
        coefficients : array-like, shape (..., 2)
            One pair or an array of pairs.
        relative : bool, default=False
            Return excess MSE above the OLS minimum. This isolates contour
            geometry from the vertical offset due to residual noise.

        Returns
        -------
        float or ndarray
            MSE values with the final coefficient dimension removed.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        if (
            coefficients.ndim < 1
            or coefficients.shape[-1] != 2
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("coefficients must be finite with shape (..., 2)")
        delta = coefficients - self.w_ols_
        excess = np.maximum(np.einsum("...i,ij,...j->...", delta, self.gram_, delta), 0)
        loss = excess if relative else excess + self.minimum_loss_
        return float(loss) if loss.ndim == 0 else loss

    def _mse_grid(self, w1_range, w2_range, relative=False):
        W1, W2 = np.meshgrid(w1_range, w2_range)
        Z = self.surface_loss(np.stack((W1, W2), axis=-1), relative=relative)
        return W1, W2, Z

    def _axis_labels(self, ax):
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])

    @property
    def _loss_label(self):
        return "intercept optimized" if self.fit_intercept_pref_ else "intercept = 0"

    def plot(
        self,
        plot_type="contour",
        ax=None,
        square=True,
        grid_size=None,
        coefficient_range=None,
        *,
        loss_range=None,
        relative_loss=False,
        levels=20,
        label_contours=True,
        cmap="viridis",
        title=None,
        include_origin=False,
        xlim=None,
        ylim=None,
    ):
        """Plot the unregularized MSE and base model point; return the axes.

        Parameters
        ----------
        plot_type : {'contour', '3d'}
            Contours or a three-dimensional surface.
        ax : matplotlib.axes.Axes, optional
            Existing axes, including a 3D projection for ``plot_type='3d'``.
        square : bool, default=True
            Use equal coefficient units on the two contour axes.
        grid_size : int, optional
            Override grid resolution for this plot.
        coefficient_range : float, optional
            Half-width around the base coefficients.
        loss_range : float, optional
            Deprecated alias for ``coefficient_range``.
        relative_loss : bool, default=False
            Plot excess MSE above the OLS minimum. Use the same explicit levels
            and axis limits across datasets to compare their geometry.
        levels : int or array-like, default=20
            Matplotlib contour levels (ignored for 3D plots).
        label_contours : bool, default=True
            Label contour lines with their loss values.
        cmap : str, default='viridis'
            Matplotlib colormap.
        title : str, optional
            Custom plot title.
        include_origin : bool, default=False
            Expand the default window to include zero with some padding.
        xlim, ylim : pair of floats, optional
            Explicit coefficient bounds, overriding the default window.
        """
        if plot_type not in ("contour", "3d"):
            raise ValueError("plot_type must be 'contour' or '3d'")
        size = self.grid_size if grid_size is None else _grid_size(grid_size)
        width = self._resolve_coefficient_range(coefficient_range, loss_range)
        bounds = [(w - width, w + width) for w in self.w_opt_]
        if include_origin:
            bounds = [
                (min(low, -0.05 * width), max(high, 0.05 * width))
                for low, high in bounds
            ]
        xlim = self._limits(bounds[0] if xlim is None else xlim)
        ylim = self._limits(bounds[1] if ylim is None else ylim)
        W1, W2, Z = self._mse_grid(
            np.linspace(*xlim, size), np.linspace(*ylim, size), relative=relative_loss
        )
        if ax is None:
            _, ax = plt.subplots(
                figsize=(8, 6),
                subplot_kw={"projection": "3d"} if plot_type == "3d" else {},
            )
        elif (getattr(ax, "name", None) == "3d") != (plot_type == "3d"):
            raise ValueError("ax projection must match plot_type")
        loss_name = "Excess MSE above OLS" if relative_loss else "MSE"
        if plot_type == "3d":
            ax.plot_surface(W1, W2, Z, alpha=0.7, cmap=cmap)
            ax.scatter(
                *self.w_opt_,
                self.surface_loss(self.w_opt_, relative_loss),
                color="#D55E00",
                s=100,
            )
            ax.set_zlabel(loss_name)
        else:
            contours = ax.contour(W1, W2, Z, levels=levels, cmap=cmap, alpha=0.7)
            if label_contours:
                ax.clabel(contours, inline=True, fontsize=8)
            base_name = type(self.model).__name__
            if isinstance(self.model, LinearRegression):
                base_name = "OLS"
                if np.linalg.matrix_rank(self._X_loss) < 2:
                    base_name += " (one solution)"
            base_artist = ax.scatter(
                *self.w_opt_,
                color="#D55E00",
                s=110,
                marker="*",
                zorder=5,
                label=f"{base_name}: " f"({self.w_opt_[0]:.2f}, {self.w_opt_[1]:.2f})",
            )
            if square:
                ax.set_aspect("equal", adjustable="box")
            ax.legend()
            # Only skip contours when composing onto this same surface instance.
            ax._statwrap_loss_surface = self
            ax._statwrap_relative_loss = relative_loss
            ax._statwrap_base_artist = base_artist
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self._axis_labels(ax)
        ax.set_title(
            title if title is not None else f"{loss_name} ({self._loss_label})"
        )
        return ax

    def compare_models(self, other_models, labels=None, ax=None):
        """Fit clones of other models and compare their points on one surface.

        Labels, when supplied, include the base model followed by the others.
        All models must share the base model's ``fit_intercept`` convention.
        Returns the Matplotlib axes.
        """
        fitted = [self.model]
        for model in ([] if other_models is None else other_models):
            if self._get_fit_intercept_pref(model) != self.fit_intercept_pref_:
                raise ValueError(
                    "Compared models must use the same fit_intercept setting"
                )
            fitted.append(clone(model).fit(self.X.copy(), self.y.copy()))
        labels = [type(m).__name__ for m in fitted] if labels is None else list(labels)
        if len(labels) != len(fitted):
            raise ValueError(
                "labels length must equal number of models (base + others)"
            )
        coefficients = np.vstack([self._coef2(m) for m in fitted])
        width = self.coefficient_range
        ax = self.plot(
            ax=ax,
            xlim=(coefficients[:, 0].min() - width, coefficients[:, 0].max() + width),
            ylim=(coefficients[:, 1].min() - width, coefficients[:, 1].max() + width),
            title="Model solutions on unregularized MSE",
        )
        ax.collections[-1].set_label(labels[0])
        for coef, label in zip(coefficients[1:], labels[1:]):
            ax.scatter(*coef, s=70, label=label, zorder=5)
        ax.legend()
        return ax

    def _default_alphas(self, kind):
        if kind == "lasso":
            scale = float(np.max(np.abs(self._X_loss.T @ self._y_loss)) / len(self.y))
            return np.geomspace((scale or 1) * 1e-4, (scale or 1) * 1.05, 50)
        scale = float(np.linalg.eigvalsh(self.gram_)[-1] * len(self.y))
        return np.geomspace((scale or 1) * 1e-4, (scale or 1) * 1e4, 50)

    def _fit_regularized(self, kind, alpha):
        if alpha == 0:
            return self.ols_model_
        if kind == "ridge":
            estimator = Ridge(
                alpha=alpha, fit_intercept=self.fit_intercept_pref_, solver="svd"
            )
        else:
            estimator = Lasso(
                alpha=alpha,
                fit_intercept=self.fit_intercept_pref_,
                max_iter=100000,
                tol=1e-8,
            )
        return estimator.fit(self.X, self.y)

    def regularization_path(self, kind="ridge", alphas=None, *, include_ols=True):
        """Compute coefficients, intercepts, and MSE without creating a figure.

        Parameters
        ----------
        kind : {'ridge', 'lasso'}, default='ridge'
            Model family, preserving the base model's intercept convention.
        alphas : array-like, optional
            Nonempty positive finite values, sorted and deduplicated. Defaults
            adapt to the data, from near OLS to strong shrinkage. Alpha uses
            scikit-learn's convention: Ridge minimizes RSS + alpha * ||w||2**2;
            Lasso minimizes MSE/2 + alpha * ||w||1. Equal numerical alphas are
            not equivalent penalties.
        include_ols : bool, default=True
            Prepend a minimum-norm OLS solution at alpha=0. With collinear X,
            this need not equal the limiting Lasso solution.

        Returns
        -------
        RegularizationPath
            Numerical values used by both styles of path plot.
        """
        kind = self._kind(kind)
        alphas = self._ensure_alphas(alphas, self._default_alphas(kind))
        if include_ols:
            alphas = np.r_[0.0, alphas]
        fitted = [self._fit_regularized(kind, alpha) for alpha in alphas]
        coefficients = np.vstack([self._coef2(m) for m in fitted])
        intercepts = np.array([m.intercept_ for m in fitted], dtype=float)
        mse = np.array(
            [self.evaluate_loss(c, b) for c, b in zip(coefficients, intercepts)]
        )
        return RegularizationPath(kind, alphas, coefficients, intercepts, mse)

    def _lasso_path_coeffs(self, alphas):
        return self.regularization_path("lasso", alphas).coefficients

    def _has_surface(self, ax):
        return (
            ax is not None
            and getattr(ax, "_statwrap_loss_surface", None) is self
            and getattr(ax, "_statwrap_base_artist", None) in ax.collections
        )

    def _prepare_overlay(self, ax, draw_surface=None, **surface_kwargs):
        if (
            getattr(ax, "name", None) == "3d"
            or surface_kwargs.get("plot_type", "contour") != "contour"
        ):
            raise ValueError("Overlays require two-dimensional axes")
        if draw_surface is None:
            draw_surface = not self._has_surface(ax)
        if draw_surface:
            return self.plot(ax=ax, **surface_kwargs)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        # Explicit range/limit requests still take effect on an existing surface.
        if (
            surface_kwargs.get("coefficient_range") is not None
            or surface_kwargs.get("loss_range") is not None
        ):
            width = self._resolve_coefficient_range(
                surface_kwargs.get("coefficient_range"),
                surface_kwargs.get("loss_range"),
            )
            ax.set_xlim(self.w_opt_[0] - width, self.w_opt_[0] + width)
            ax.set_ylim(self.w_opt_[1] - width, self.w_opt_[1] + width)
        for name in ("xlim", "ylim"):
            if surface_kwargs.get(name) is not None:
                getattr(ax, f"set_{name}")(self._limits(surface_kwargs[name]))
        return ax

    def _plot_regularization_path(
        self, kind, alphas, ax, draw_surface, **surface_kwargs
    ):
        path = self.regularization_path(kind, alphas)
        ax = self._prepare_overlay(ax, draw_surface, **surface_kwargs)
        limits = ax.get_xlim(), ax.get_ylim()
        color = "#0072B2" if kind == "ridge" else "#D55E00"
        # A nonunique OLS point need not lie on Lasso's limit as alpha -> 0.
        start = 1 if kind == "lasso" and np.linalg.matrix_rank(self._X_loss) < 2 else 0
        ax.plot(
            *path.coefficients[start:].T,
            color=color,
            marker=".",
            linewidth=2,
            markersize=4,
            label=f"{kind.title()} path (increasing α)",
            zorder=4,
        )
        if not (
            self._has_surface(ax)
            and isinstance(self.model, LinearRegression)
            and np.allclose(self.w_opt_, self.w_ols_)
        ):
            ax.scatter(
                *path.coefficients[0],
                marker="*",
                color="#009E73",
                s=100,
                label="OLS (α=0)" if start == 0 else "OLS (one solution, α=0)",
                zorder=5,
            )
        ax.scatter(
            *path.coefficients[-1],
            marker="s",
            color=color,
            s=35,
            label=f"End (α={path.alphas[-1]:.3g})",
            zorder=5,
        )
        if not ax.get_autoscalex_on():
            ax.set_xlim(limits[0])
        if not ax.get_autoscaley_on():
            ax.set_ylim(limits[1])
        ax.set_aspect("equal", adjustable="box")
        self._axis_labels(ax)
        ax.legend()
        return ax

    def plot_ridge_path_on_surface(
        self,
        alphas=None,
        ax=None,
        coefficient_range=None,
        *,
        loss_range=None,
        draw_surface=None,
        **surface_kwargs,
    ):
        """Overlay a ridge path; return axes.

        Reusing axes from this instance does not redraw contours or reset limits.
        ``draw_surface=True`` explicitly redraws; ``False`` adds only the path.
        Additional keywords are forwarded to :meth:`plot` when drawing a surface.
        """
        return self._plot_regularization_path(
            "ridge",
            alphas,
            ax,
            draw_surface,
            coefficient_range=coefficient_range,
            loss_range=loss_range,
            **surface_kwargs,
        )

    def plot_lasso_path_on_surface(
        self,
        alphas=None,
        ax=None,
        coefficient_range=None,
        *,
        loss_range=None,
        draw_surface=None,
        **surface_kwargs,
    ):
        """Overlay a lasso path; return axes.

        Accepts the same options as :meth:`plot_ridge_path_on_surface`. Under
        exact collinearity, the minimum-norm OLS marker is not connected to the
        positive-alpha Lasso path because the OLS solution is not unique.
        """
        return self._plot_regularization_path(
            "lasso",
            alphas,
            ax,
            draw_surface,
            coefficient_range=coefficient_range,
            loss_range=loss_range,
            **surface_kwargs,
        )

    def _plot_coef_path(self, kind, alphas, ax):
        path = self.regularization_path(kind, alphas)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        start = 1 if kind == "lasso" and np.linalg.matrix_rank(self._X_loss) < 2 else 0
        for i, name in enumerate(self.feature_names):
            (line,) = ax.plot(
                path.alphas[start:],
                path.coefficients[start:, i],
                label=name,
                marker=".",
                markersize=4,
            )
            if start:
                ax.scatter(
                    0, path.coefficients[0, i], marker="*", color=line.get_color()
                )
        # A linear segment near zero puts OLS at its actual alpha, not an epsilon.
        ax.set_xscale("symlog", linthresh=float(path.alphas[1]))
        ax.axhline(0, color="k", linestyle="--", alpha=0.25)
        ax.axvline(0, color="k", linestyle=":", alpha=0.25)
        ax.set_xlabel("α (linear near 0, logarithmic above first positive α)")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"{kind.title()} coefficients; OLS at α=0")
        ax.legend()
        ax.grid(True, alpha=0.2)
        return ax

    def plot_ridge_coef_path(self, alphas=None, ax=None):
        """Plot ridge coefficients against alpha, with OLS at exactly zero."""
        return self._plot_coef_path("ridge", alphas, ax)

    def plot_lasso_coef_path(self, alphas=None, ax=None):
        """Plot lasso coefficients against alpha, with OLS at exactly zero."""
        return self._plot_coef_path("lasso", alphas, ax)

    def plot_constraint(self, radius, penalty="l1", ax=None, **kwargs):
        """Add an L1 diamond or L2 circle to a surface; return axes.

        ``radius`` is a norm bound, not alpha. ``penalty`` is ``'l1'`` or
        ``'l2'``. Line styling is passed through to Matplotlib. Omitted axes
        create a new loss surface; existing axes and limits are preserved.
        """
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        _positive(radius, "radius", allow_zero=True)
        ax = self._prepare_overlay(ax)
        _plot_ball(radius, ax, penalty, kwargs)
        ax.legend()
        return ax

    def plot_regularization(
        self,
        kind="lasso",
        alpha=1.0,
        ax=None,
        *,
        show_path=True,
        show_loss_contour=True,
        alphas=None,
        **surface_kwargs,
    ):
        """Draw a fitted solution, its matching constraint, and an optional path.

        Parameters
        ----------
        kind : {'ridge', 'lasso'}, default='lasso'
            Model family using scikit-learn's alpha convention.
        alpha : float, default=1
            Nonnegative penalty strength. Zero uses OLS.
        ax : matplotlib.axes.Axes, optional
            Existing surface. Its limits and contours are preserved.
        show_path : bool, default=True
            Show the regularization path as context.
        show_loss_contour : bool, default=True
            Highlight the exact MSE contour through the selected solution. This
            shows where the norm boundary touches a loss contour, independently
            of the background grid's resolution.
        alphas : array-like, optional
            Positive alpha values for the path; the selected positive alpha is
            also included. Defaults adapt to the data.
        **surface_kwargs
            Options for :meth:`plot`. On a new surface, automatic bounds include
            the entire norm boundary unless explicit bounds/range are supplied.

        Returns
        -------
        matplotlib.axes.Axes
            The composed plot. The boundary radius is ||w||1 for Lasso and
            ||w||2 for Ridge at the selected alpha, so it passes through the
            marked solution. A zero solution is shown as a point at the origin.
        """
        kind = self._kind(kind)
        alpha = _positive(alpha, "alpha", allow_zero=True)
        fitted = self._fit_regularized(kind, alpha)
        coef = self._coef2(fitted)
        penalty = "l1" if kind == "lasso" else "l2"
        radius = float(np.linalg.norm(coef, ord=1 if penalty == "l1" else 2))
        options = dict(surface_kwargs)
        new_surface = not self._has_surface(ax)
        if (
            new_surface
            and options.get("coefficient_range") is None
            and options.get("loss_range") is None
        ):
            padding = max(0.1 * radius, 0.2)
            for i, name in enumerate(("xlim", "ylim")):
                if options.get(name) is None:
                    options[name] = (
                        min(-radius, self.w_opt_[i] - self.coefficient_range) - padding,
                        max(radius, self.w_opt_[i] + self.coefficient_range) + padding,
                    )
        ax = self._prepare_overlay(ax, **options)
        if show_path:
            values = self._ensure_alphas(alphas, self._default_alphas(kind))
            if alpha > 0:
                values = np.unique(np.r_[values, alpha])
            self._plot_regularization_path(kind, values, ax, False)
        if show_loss_contour:
            self._plot_solution_contour(coef, ax)
        self.plot_constraint(radius, penalty, ax=ax, linestyle="--")
        ax.scatter(
            *coef,
            s=90,
            color="#CC79A7",
            edgecolors="black",
            zorder=7,
            label=f"{kind.title()} (α={alpha:.3g})",
        )
        ax.set_title(
            options.get("title")
            or f"{kind.title()}: α={alpha:.3g}, "
            f"{penalty.upper()} radius={radius:.3g}"
        )
        ax.legend(loc="upper left", fontsize="small")
        return ax

    def _plot_solution_contour(self, coefficients, ax):
        """Draw an exact quadratic level set, including parallel lines at rank 1."""
        excess = self.surface_loss(coefficients, relative=True)
        values, vectors = np.linalg.eigh(self.gram_)
        tolerance = np.finfo(float).eps * float(values[-1]) * 2
        active = values > tolerance
        if excess == 0 or not np.any(active):
            return
        style = {
            "color": "#CC79A7",
            "linewidth": 1.8,
            "zorder": 3,
            "label": "MSE at selected fit",
        }
        if np.all(active):
            theta = np.linspace(0, 2 * np.pi, 721)
            circle = np.array([np.cos(theta), np.sin(theta)])
            points = self.w_ols_[:, None] + vectors @ (
                np.sqrt(excess / values)[:, None] * circle
            )
            ax.plot(*points, **style)
        else:
            # Rank one: the level set is two lines along the flat direction.
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            corners = np.array([[x, y] for x in xlim for y in ylim])
            span = np.max(np.linalg.norm(corners - self.w_ols_, axis=1)) * 2
            tangent = vectors[:, 0, None] * np.array([-span, span])
            normal = np.sqrt(excess / values[-1]) * vectors[:, -1]
            for sign in (-1, 1):
                points = (self.w_ols_ + sign * normal)[:, None] + tangent
                ax.plot(*points, **style)
                style["label"] = "_nolegend_"

    def plot_eigenvectors(
        self, ax=None, length=1.0, *, annotate=True, **surface_kwargs
    ):
        """Overlay principal directions of the MSE contours; return axes.

        Eigenvectors of X-centered.T @ X-centered / n (or X.T @ X / n without
        an intercept) are drawn from the **OLS minimum**, even if the base model
        is regularized. Both arrows have the supplied positive length in
        coefficient units. Labels give Gram eigenvalues; MSE curvature is twice
        those eigenvalues. Zero eigenvalues identify flat directions.

        ``annotate=False`` hides the labels. Extra keywords go to :meth:`plot`
        when creating the surface. Access numerical directions through
        ``np.linalg.eigh(surface.gram_)``.
        """
        length = _positive(length, "length")
        ax = self._prepare_overlay(ax, **surface_kwargs)
        values, vectors = np.linalg.eigh(self.gram_)
        for i, color in enumerate(("#0072B2", "#D55E00")):
            end = self.w_ols_ + length * vectors[:, i]
            ax.annotate(
                "",
                xy=end,
                xytext=self.w_ols_,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
                zorder=6,
            )
            if annotate:
                ax.annotate(
                    f"λ{i + 1}={max(values[i], 0):.3g}",
                    xy=end,
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=color,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
        ax.set_aspect("equal", adjustable="box")
        return ax
