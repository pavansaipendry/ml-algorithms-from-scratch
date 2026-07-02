"""LinearRegressionPlus -- an upgraded, from-scratch linear regression.

Upgrades over ordinary least squares (OLS):

  1. Regularization   : L2 (ridge), L1 (lasso), or elastic net
  2. Robust loss      : Huber loss, resistant to outliers
  3. Two solvers      : exact closed-form normal equation, and
                        mini-batch gradient descent with early stopping
  4. Feature pipeline : built-in standardization + polynomial expansion
  5. Self-tuning      : alpha='auto' / degree='auto' picks the best value
                        by k-fold cross-validation
  6. Uncertainty      : standard errors, p-values, confidence intervals on
                        coefficients, and prediction intervals -- classical
                        t-statistics for exact OLS, bootstrap otherwise
  7. Diagnostics      : loss history, CV results, R^2, adjusted R^2, RMSE, MAE

Only dependencies: NumPy and the standard library.
"""

import math

import numpy as np

_PENALTIES = (None, "l2", "l1", "elasticnet")
_LOSSES = ("squared", "huber")
_SOLVERS = ("auto", "closed_form", "gradient_descent")


# --------------------------------------------------------------------------
# Student's t distribution from scratch (so classical inference needs no scipy)
# --------------------------------------------------------------------------

def _betacf(a, b, x, max_iter=200, eps=3e-12):
    """Continued fraction for the incomplete beta (Numerical Recipes, Lentz)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t_abs, dof):
    """P(|T| > t) for T ~ Student's t with `dof` degrees of freedom."""
    return _betainc(dof / 2.0, 0.5, dof / (dof + t_abs * t_abs))


def _t_crit(level, dof):
    """Two-sided critical value: t such that P(|T| > t) = 1 - level."""
    target = 1.0 - level
    lo, hi = 0.0, 10.0
    while _t_two_sided_p(hi, dof) > target and hi < 1e6:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _t_two_sided_p(mid, dof) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------


class LinearRegressionPlus:
    """Linear regression, upgraded.

    Parameters
    ----------
    solver : 'auto' | 'closed_form' | 'gradient_descent'
        'auto' uses the exact closed-form solution when one exists
        (squared loss, no L1 term) and gradient descent otherwise.
    penalty : None | 'l2' | 'l1' | 'elasticnet'
        Regularization type. The bias/intercept is never penalized.
    alpha : float | 'auto'
        Regularization strength. 'auto' picks the best value from a log
        grid by k-fold cross-validation at fit time (requires a penalty).
    l1_ratio : float
        Elastic net mix in [0, 1] (1.0 = pure L1, 0.0 = pure L2).
    loss : 'squared' | 'huber'
        Huber is quadratic for small residuals, linear for large ones,
        so outliers pull on the fit far less.
    huber_delta : float
        Residual size where Huber switches from quadratic to linear.
    degree : int | 'auto'
        Polynomial degree for feature expansion (1 = plain linear). Each
        feature x becomes x, x^2, ..., x^degree. 'auto' picks the degree
        by k-fold cross-validation at fit time.
    standardize : bool
        Z-score the (expanded) features before fitting. Coefficients are
        converted back to the original feature scale after fitting, so
        ``coef_`` / ``intercept_`` are always in user units.
    learning_rate, max_iter, batch_size, tol, patience
        Gradient descent knobs. ``batch_size=None`` means full batch.
        Training stops early once the epoch loss has not improved by
        more than ``tol`` for ``patience`` consecutive epochs.
    cv : int
        Number of folds used when tuning alpha/degree.
    n_bootstrap : int
        Number of bootstrap refits used for inference when the classical
        formulas do not apply (any penalized, Huber, or GD fit).
    random_state : int | None
        Seed for mini-batch shuffling, CV splits, and the bootstrap.

    Attributes
    ----------
    coef_ : ndarray of shape (n_expanded_features,)
    intercept_ : float
    history_ : list of per-epoch loss values (gradient descent only)
    alpha_, degree_ : values chosen by cross-validation (if tuned)
    cv_results_ : list of {'alpha', 'degree', 'cv_mse'} (if tuned)
    """

    _ALPHA_GRID = tuple(np.logspace(-4, 2, 13))
    _DEGREE_GRID = (1, 2, 3, 4)

    def __init__(self, solver="auto", penalty=None, alpha=0.0, l1_ratio=0.5,
                 loss="squared", huber_delta=1.0, degree=1, standardize=True,
                 learning_rate=0.05, max_iter=2000, batch_size=None,
                 tol=1e-8, patience=25, cv=5, n_bootstrap=200,
                 random_state=None):
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}")
        if penalty not in _PENALTIES:
            raise ValueError(f"penalty must be one of {_PENALTIES}")
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        if degree != "auto" and degree < 1:
            raise ValueError("degree must be >= 1 or 'auto'")
        if alpha != "auto" and alpha < 0.0:
            raise ValueError("alpha must be >= 0 or 'auto'")
        if alpha == "auto" and penalty is None:
            raise ValueError("alpha='auto' needs a penalty; "
                             "set penalty='l2', 'l1' or 'elasticnet'")

        self.solver = solver
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.huber_delta = huber_delta
        self.degree = degree
        self.standardize = standardize
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.cv = cv
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        X_raw = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(X_raw) != len(y):
            raise ValueError("X and y have different numbers of samples")

        if self.alpha == "auto" or self.degree == "auto":
            self._tune(X_raw, y)

        X_exp = self._expand(X_raw)
        X = X_exp
        if self.standardize:
            self._mu = X.mean(axis=0)
            self._sigma = X.std(axis=0)
            self._sigma[self._sigma == 0.0] = 1.0  # constant column: leave as-is
            X = (X - self._mu) / self._sigma
        Xb = np.column_stack([np.ones(len(X)), X])  # bias column first

        solver = self.solver
        if solver == "auto":
            exact_ok = self.loss == "squared" and self._l1() == 0.0
            solver = "closed_form" if exact_ok else "gradient_descent"
        if solver == "closed_form" and (self.loss != "squared" or self._l1() != 0.0):
            raise ValueError("closed_form needs squared loss and no L1 term; "
                             "use solver='gradient_descent'")

        self.history_ = []
        if solver == "closed_form":
            w = self._solve_closed_form(Xb, y)
        else:
            w = self._solve_gradient_descent(Xb, y)

        self._store_readable_coefs(w)

        # bookkeeping for inference (uncertainty estimates)
        self._X_raw = X_raw
        self._y = y
        self._Xb_orig = np.column_stack([np.ones(len(X_exp)), X_exp])
        self._n_features_raw = X_raw.shape[1]
        # classical OLS formulas require an unpenalized squared-loss exact fit
        self._classical_ok = solver == "closed_form" and self._l2() == 0.0
        self._inference_cache = None
        return self

    def _solve_closed_form(self, Xb, y):
        # Objective: (1/2n)||Xw - y||^2 + (l2/2)||w||^2
        # => (X'X + n*l2*I) w = X'y, with the bias left unpenalized.
        if self._l2() > 0.0:
            reg = self._l2() * len(y) * np.eye(Xb.shape[1])
            reg[0, 0] = 0.0
            return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        # Pseudo-inverse: stable even when X'X is singular (collinear features).
        return np.linalg.pinv(Xb) @ y

    def _solve_gradient_descent(self, Xb, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = Xb.shape
        batch = self.batch_size or n_samples
        l1 = self._l1()

        w = np.zeros(n_features)
        best_loss, best_w, epochs_since_best = np.inf, w.copy(), 0

        for _ in range(self.max_iter):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch):
                idx = order[start:start + batch]
                w -= self.learning_rate * self._gradient(Xb[idx], y[idx], w)
                if l1 > 0.0:
                    # Proximal step for the non-differentiable L1 term:
                    # soft-thresholding pushes small weights exactly to zero.
                    shrink = self.learning_rate * l1
                    w[1:] = np.sign(w[1:]) * np.maximum(np.abs(w[1:]) - shrink, 0.0)

            loss = self._loss(Xb, y, w)
            self.history_.append(loss)
            if loss < best_loss - self.tol:
                best_loss, best_w, epochs_since_best = loss, w.copy(), 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= self.patience:
                    break
        return best_w

    def _gradient(self, Xb, y, w):
        r = Xb @ w - y
        if self.loss == "huber":
            r = np.clip(r, -self.huber_delta, self.huber_delta)
        grad = Xb.T @ r / len(y)
        if self._l2() > 0.0:
            grad[1:] += self._l2() * w[1:]
        return grad

    def _loss(self, Xb, y, w):
        r = Xb @ w - y
        if self.loss == "huber":
            d = self.huber_delta
            a = np.abs(r)
            data = np.where(a <= d, 0.5 * r ** 2, d * (a - 0.5 * d)).mean()
        else:
            data = 0.5 * (r ** 2).mean()
        return (data
                + self._l1() * np.abs(w[1:]).sum()
                + 0.5 * self._l2() * (w[1:] ** 2).sum())

    # -------------------------------------------------- self-tuning (CV)

    def _tune(self, X_raw, y):
        alphas = self._ALPHA_GRID if self.alpha == "auto" else (self.alpha,)
        degrees = self._DEGREE_GRID if self.degree == "auto" else (self.degree,)

        n = len(y)
        k = min(self.cv, n)
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n)
        folds = np.array_split(order, k)

        results = []
        for d in degrees:
            for a in alphas:
                fold_mse = []
                for i in range(k):
                    val = folds[i]
                    train = np.concatenate([folds[j] for j in range(k) if j != i])
                    m = self._clone(alpha=a, degree=d)
                    m.fit(X_raw[train], y[train])
                    err = y[val] - m.predict(X_raw[val])
                    fold_mse.append((err ** 2).mean())
                se = (float(np.std(fold_mse, ddof=1) / np.sqrt(k))
                      if k > 1 else 0.0)
                results.append({"alpha": a, "degree": d,
                                "cv_mse": float(np.mean(fold_mse)),
                                "cv_se": se})

        # One-standard-error rule (as in glmnet): among all models whose CV
        # error is within one SE of the best, pick the simplest -- lowest
        # degree first, then strongest regularization. This avoids choosing
        # a more complex model over a simpler one for a statistically
        # meaningless improvement.
        best = min(results, key=lambda r: r["cv_mse"])
        threshold = best["cv_mse"] + best["cv_se"]
        candidates = [r for r in results if r["cv_mse"] <= threshold]
        chosen = min(candidates, key=lambda r: (r["degree"], -r["alpha"]))

        self.alpha, self.degree = chosen["alpha"], chosen["degree"]
        self.alpha_, self.degree_ = chosen["alpha"], chosen["degree"]
        self.cv_results_ = results

    def _clone(self, **overrides):
        params = dict(solver=self.solver, penalty=self.penalty, alpha=self.alpha,
                      l1_ratio=self.l1_ratio, loss=self.loss,
                      huber_delta=self.huber_delta, degree=self.degree,
                      standardize=self.standardize,
                      learning_rate=self.learning_rate, max_iter=self.max_iter,
                      batch_size=self.batch_size, tol=self.tol,
                      patience=self.patience, cv=self.cv,
                      n_bootstrap=self.n_bootstrap,
                      random_state=self.random_state)
        params.update(overrides)
        return LinearRegressionPlus(**params)

    # -------------------------------------------------------------- predict

    def predict(self, X):
        X = self._expand(self._as_2d(X))
        return X @ self.coef_ + self.intercept_

    def predict_interval(self, X, level=0.95):
        """Point predictions with a prediction interval for each one.

        Returns ``(y_pred, lower, upper)``. Classical t-interval for exact
        OLS fits; bootstrap-plus-resampled-residuals otherwise.
        """
        stats = self._inference()
        X_exp = self._expand(self._as_2d(X))
        y_pred = X_exp @ self.coef_ + self.intercept_
        Xb0 = np.column_stack([np.ones(len(X_exp)), X_exp])

        if stats["method"] == "classical":
            leverage = np.sum((Xb0 @ stats["xtx_inv"]) * Xb0, axis=1)
            half = (_t_crit(level, stats["dof"])
                    * np.sqrt(stats["sigma2"] * (1.0 + leverage)))
            return y_pred, y_pred - half, y_pred + half

        # Bootstrap: model uncertainty from refit coefficients,
        # noise from resampled training residuals.
        boot = stats["boot_coefs"]                      # (B, p+1)
        rng = np.random.default_rng(self.random_state)
        preds = Xb0 @ boot.T                            # (m, B)
        preds = preds + rng.choice(stats["residuals"], size=preds.shape)
        q = 100.0 * (1.0 - level) / 2.0
        lower = np.percentile(preds, q, axis=1)
        upper = np.percentile(preds, 100.0 - q, axis=1)
        return y_pred, lower, upper

    def score(self, X, y):
        """Coefficient of determination R^2."""
        return self.evaluate(X, y)["r2"]

    def evaluate(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        residual = y - self.predict(X)
        ss_res = (residual ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        n, p = len(y), len(self.coef_)
        r2 = 1.0 - ss_res / ss_tot
        return {
            "r2": r2,
            "adjusted_r2": 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1),
            "mse": (residual ** 2).mean(),
            "rmse": float(np.sqrt((residual ** 2).mean())),
            "mae": np.abs(residual).mean(),
        }

    # ------------------------------------------------ inference / summary

    def coef_stats(self, level=0.95):
        """Per-coefficient uncertainty: stderr, test statistic, p-value, CI.

        Classical t-statistics for exact OLS fits; bootstrap otherwise
        (bootstrap p-values are sign-crossing approximations, floored
        at 1/n_bootstrap).

        Note: for penalized fits the intervals describe the *regularized*
        estimator, which is deliberately biased toward zero -- under strong
        shrinkage they will center on the shrunken coefficient, not the
        unpenalized truth.
        """
        stats = self._inference()
        w = np.r_[self.intercept_, self.coef_]

        if stats["method"] == "classical":
            se = stats["se"]
            t = w / se
            p = np.array([_t_two_sided_p(abs(ti), stats["dof"]) for ti in t])
            crit = _t_crit(level, stats["dof"])
            lower, upper = w - crit * se, w + crit * se
        else:
            boot = stats["boot_coefs"]
            se = boot.std(axis=0, ddof=1)
            t = np.divide(w, se, out=np.zeros_like(w), where=se > 0)
            q = 100.0 * (1.0 - level) / 2.0
            lower = np.percentile(boot, q, axis=0)
            upper = np.percentile(boot, 100.0 - q, axis=0)
            frac_pos = (boot > 0).mean(axis=0)
            p = np.maximum(2.0 * np.minimum(frac_pos, 1.0 - frac_pos),
                           1.0 / len(boot))

        return {"term": ["intercept"] + self._feature_names(),
                "coef": w, "stderr": se, "stat": t, "p_value": p,
                "ci_lower": lower, "ci_upper": upper,
                "method": stats["method"]}

    def summary(self, level=0.95):
        """statsmodels-style text summary of the fitted model."""
        s = self.coef_stats(level)
        ev = self.evaluate(self._X_raw, self._y)
        pct = f"{level:.0%}"
        lines = [
            f"LinearRegressionPlus summary  ({s['method']} inference, {pct} CI)",
            "=" * 74,
            (f"n={len(self._y)}  features={len(self.coef_)}  "
             f"R^2={ev['r2']:.4f}  adj R^2={ev['adjusted_r2']:.4f}  "
             f"RMSE={ev['rmse']:.4f}"),
        ]
        if hasattr(self, "alpha_"):
            lines.append(f"tuned by {self.cv}-fold CV: "
                         f"alpha={self.alpha_:g}, degree={self.degree_}")
        lines += [
            "-" * 74,
            (f"{'term':<14}{'coef':>10}{'stderr':>10}{'stat':>9}"
             f"{'p-value':>12}{'ci_low':>10}{'ci_high':>10}"),
        ]
        for i, name in enumerate(s["term"]):
            lines.append(f"{name:<14}{s['coef'][i]:>10.4f}{s['stderr'][i]:>10.4f}"
                         f"{s['stat'][i]:>9.2f}{s['p_value'][i]:>12.4g}"
                         f"{s['ci_lower'][i]:>10.4f}{s['ci_upper'][i]:>10.4f}")
        lines.append("=" * 74)
        return "\n".join(lines)

    def _inference(self):
        if self._inference_cache is not None:
            return self._inference_cache
        if self._classical_ok:
            self._inference_cache = self._classical_inference()
        else:
            self._inference_cache = self._bootstrap_inference()
        return self._inference_cache

    def _classical_inference(self):
        Xb, y = self._Xb_orig, self._y
        n, k = Xb.shape
        dof = n - k
        if dof <= 0:
            raise ValueError("not enough samples for classical inference "
                             f"(n={n}, parameters={k})")
        residuals = y - Xb @ np.r_[self.intercept_, self.coef_]
        sigma2 = float(residuals @ residuals) / dof
        xtx_inv = np.linalg.pinv(Xb.T @ Xb)
        se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0.0, None))
        return {"method": "classical", "se": se, "dof": dof,
                "sigma2": sigma2, "xtx_inv": xtx_inv}

    def _bootstrap_inference(self):
        rng = np.random.default_rng(self.random_state)
        n = len(self._y)
        B = self.n_bootstrap
        boot = np.empty((B, len(self.coef_) + 1))
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            m = self._clone()  # alpha/degree already resolved to numbers
            m.fit(self._X_raw[idx], self._y[idx])
            boot[b, 0] = m.intercept_
            boot[b, 1:] = m.coef_
        residuals = self._y - self.predict(self._X_raw)
        return {"method": "bootstrap", "boot_coefs": boot,
                "residuals": residuals}

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _expand(self, X):
        if self.degree < 2:
            return X
        return np.hstack([X ** d for d in range(1, self.degree + 1)])

    def _feature_names(self):
        base = [f"x{j + 1}" for j in range(self._n_features_raw)]
        if self.degree < 2:
            return base
        return [name if d == 1 else f"{name}^{d}"
                for d in range(1, self.degree + 1) for name in base]

    def _l1(self):
        if self.penalty == "l1":
            return self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * self.l1_ratio
        return 0.0

    def _l2(self):
        if self.penalty == "l2":
            return self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * (1.0 - self.l1_ratio)
        return 0.0

    def _store_readable_coefs(self, w):
        # Undo standardization so coef_/intercept_ are in original units.
        if self.standardize:
            self.coef_ = w[1:] / self._sigma
            self.intercept_ = float(w[0] - (w[1:] * self._mu / self._sigma).sum())
        else:
            self.coef_ = w[1:].copy()
            self.intercept_ = float(w[0])
