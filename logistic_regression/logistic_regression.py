"""LogisticRegressionPlus -- an upgraded, from-scratch logistic regression.

Upgrades over textbook logistic regression:

  1. Newton's method   : converges in ~10 iterations (vs thousands for GD)
  2. Regularization    : L2 (ridge), L1 (lasso), or elastic net
  3. Imbalanced data   : class_weight='balanced' reweights rare classes
  4. Multiclass        : softmax regression built in (not one-vs-rest)
  5. Feature pipeline  : built-in standardization + polynomial expansion
  6. Self-tuning       : alpha='auto' / degree='auto' picked by stratified
                         k-fold cross-validation (one-standard-error rule)
  7. Uncertainty       : standard errors, p-values, confidence intervals on
                         coefficients -- Wald tests from the Fisher
                         information for exact fits, bootstrap otherwise
  8. Diagnostics       : accuracy, precision, recall, F1, ROC-AUC, log loss,
                         ROC curves, loss history, CV results

Only dependencies: NumPy and the standard library.
"""

import math

import numpy as np

_PENALTIES = (None, "l2", "l1", "elasticnet")
_SOLVERS = ("auto", "newton", "gradient_descent")


# --------------------------------------------------------------------------
# standard normal distribution from scratch (for Wald tests, no scipy)
# --------------------------------------------------------------------------

def _norm_sf(z):
    """P(Z > z) for a standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_crit(level):
    """Two-sided critical value: z such that P(|Z| > z) = 1 - level."""
    target = (1.0 - level) / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _norm_sf(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sigmoid(z):
    # tanh form is numerically stable for large |z|
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------


class LogisticRegressionPlus:
    """Logistic regression, upgraded. Binary or multiclass (softmax).

    Parameters
    ----------
    solver : 'auto' | 'newton' | 'gradient_descent'
        'auto' uses Newton's method (IRLS) for binary problems without an
        L1 term -- typically converging in under 15 iterations -- and
        mini-batch gradient descent otherwise (L1, multiclass).
    penalty : None | 'l2' | 'l1' | 'elasticnet'
        Regularization type. The bias/intercept is never penalized.
    alpha : float | 'auto'
        Regularization strength. 'auto' picks the best value from a log
        grid by stratified k-fold CV at fit time (requires a penalty).
    l1_ratio : float
        Elastic net mix in [0, 1] (1.0 = pure L1, 0.0 = pure L2).
    degree : int | 'auto'
        Polynomial degree for feature expansion, so the model can learn
        curved decision boundaries. 'auto' picks it by CV.
    standardize : bool
        Z-score the (expanded) features before fitting. Coefficients are
        converted back to original units after fitting.
    class_weight : None | 'balanced'
        'balanced' weights each sample by n / (n_classes * count(class)),
        so rare classes are not drowned out by common ones.
    learning_rate, max_iter, batch_size, tol, patience
        Gradient descent knobs (early stopping as in LinearRegressionPlus).
    newton_max_iter : int
        Iteration cap for Newton's method.
    cv : int
        Folds used when tuning alpha/degree (stratified by class).
    n_bootstrap : int
        Bootstrap refits used for inference when Wald theory does not
        apply (penalized or class-weighted fits).
    random_state : int | None
        Seed for shuffling, CV splits, and the bootstrap.

    Attributes
    ----------
    classes_ : sorted unique class labels
    coef_ : (n_features,) for binary, (n_classes, n_features) for multiclass
    intercept_ : float for binary, (n_classes,) for multiclass
    history_ : per-epoch loss (gradient descent only)
    alpha_, degree_, cv_results_ : set when tuned by CV
    """

    _ALPHA_GRID = tuple(np.logspace(-4, 2, 13))
    _DEGREE_GRID = (1, 2, 3)

    def __init__(self, solver="auto", penalty=None, alpha=0.0, l1_ratio=0.5,
                 degree=1, standardize=True, class_weight=None,
                 learning_rate=0.1, max_iter=1000, batch_size=None,
                 tol=1e-8, patience=25, newton_max_iter=50, cv=5,
                 n_bootstrap=200, random_state=None):
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}")
        if penalty not in _PENALTIES:
            raise ValueError(f"penalty must be one of {_PENALTIES}")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        if degree != "auto" and degree < 1:
            raise ValueError("degree must be >= 1 or 'auto'")
        if alpha != "auto" and alpha < 0.0:
            raise ValueError("alpha must be >= 0 or 'auto'")
        if alpha == "auto" and penalty is None:
            raise ValueError("alpha='auto' needs a penalty; "
                             "set penalty='l2', 'l1' or 'elasticnet'")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")

        self.solver = solver
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.degree = degree
        self.standardize = standardize
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.newton_max_iter = newton_max_iter
        self.cv = cv
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        X_raw = self._as_2d(X)
        y = np.asarray(y).ravel()
        if len(X_raw) != len(y):
            raise ValueError("X and y have different numbers of samples")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least 2 classes")

        if self.alpha == "auto" or self.degree == "auto":
            self._tune(X_raw, y)

        X_exp = self._expand(X_raw)
        X = X_exp
        if self.standardize:
            self._mu = X.mean(axis=0)
            self._sigma = X.std(axis=0)
            self._sigma[self._sigma == 0.0] = 1.0
            X = (X - self._mu) / self._sigma
        Xb = np.column_stack([np.ones(len(X)), X])

        sw = self._sample_weights(y)
        binary = len(self.classes_) == 2

        solver = self.solver
        if solver == "auto":
            solver = ("newton" if binary and self._l1() == 0.0
                      else "gradient_descent")
        if solver == "newton" and (not binary or self._l1() != 0.0):
            raise ValueError("newton handles binary problems without an L1 "
                             "term; use solver='gradient_descent'")

        self.history_ = []
        if binary:
            y01 = (y == self.classes_[1]).astype(float)
            if solver == "newton":
                w = self._solve_newton(Xb, y01, sw)
            else:
                w = self._solve_gd_binary(Xb, y01, sw)
            self._store_readable_coefs(w)
        else:
            Y = (y[:, None] == self.classes_[None, :]).astype(float)
            W = self._solve_gd_softmax(Xb, Y, sw)
            self._store_readable_coefs_multi(W)

        # bookkeeping for inference
        self._X_raw = X_raw
        self._y = y
        self._Xb_orig = np.column_stack([np.ones(len(X_exp)), X_exp])
        self._n_features_raw = X_raw.shape[1]
        # Wald theory needs an unpenalized, unweighted maximum-likelihood fit
        self._classical_ok = (binary and self._l1() == 0.0
                              and self._l2() == 0.0
                              and self.class_weight is None)
        self._inference_cache = None
        return self

    def _sample_weights(self, y):
        if self.class_weight is None:
            return np.ones(len(y))
        counts = {c: (y == c).sum() for c in self.classes_}
        k = len(self.classes_)
        return np.array([len(y) / (k * counts[c]) for c in y])

    # ------------------------------------------------------------- solvers

    def _solve_newton(self, Xb, y01, sw):
        """IRLS: Newton's method on the (weighted, L2-penalized) log loss."""
        n_features = Xb.shape[1]
        l2 = self._l2()
        reg = l2 * np.eye(n_features)
        reg[0, 0] = 0.0
        denom = sw.sum()

        w = np.zeros(n_features)
        for _ in range(self.newton_max_iter):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (sw * (p - y01)) / denom + reg @ w
            curv = sw * np.clip(p * (1.0 - p), 1e-10, None)
            H = (Xb * curv[:, None]).T @ Xb / denom + reg
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < self.tol:
                break
        return w

    def _solve_gd_binary(self, Xb, y01, sw):
        def grad(idx, w):
            p = _sigmoid(Xb[idx] @ w)
            g = Xb[idx].T @ (sw[idx] * (p - y01[idx])) / sw[idx].sum()
            if self._l2() > 0.0:
                g[1:] += self._l2() * w[1:]
            return g

        def loss(w):
            z = Xb @ w
            data = (sw * (np.logaddexp(0.0, z) - y01 * z)).sum() / sw.sum()
            return (data + self._l1() * np.abs(w[1:]).sum()
                    + 0.5 * self._l2() * (w[1:] ** 2).sum())

        return self._gd_loop(np.zeros(Xb.shape[1]), len(y01), grad, loss)

    def _solve_gd_softmax(self, Xb, Y, sw):
        n_features, k = Xb.shape[1], Y.shape[1]

        def grad(idx, W):
            P = _softmax(Xb[idx] @ W)
            G = Xb[idx].T @ ((P - Y[idx]) * sw[idx, None]) / sw[idx].sum()
            if self._l2() > 0.0:
                G[1:] += self._l2() * W[1:]
            return G

        def loss(W):
            P = _softmax(Xb @ W)
            p_true = np.clip((P * Y).sum(axis=1), 1e-12, None)
            data = -(sw * np.log(p_true)).sum() / sw.sum()
            return (data + self._l1() * np.abs(W[1:]).sum()
                    + 0.5 * self._l2() * (W[1:] ** 2).sum())

        return self._gd_loop(np.zeros((n_features, k)), len(Y), grad, loss)

    def _gd_loop(self, w, n_samples, grad, loss):
        """Mini-batch gradient descent with proximal L1 and early stopping."""
        rng = np.random.default_rng(self.random_state)
        batch = self.batch_size or n_samples
        l1 = self._l1()
        best_loss, best_w, epochs_since_best = np.inf, w.copy(), 0

        for _ in range(self.max_iter):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch):
                idx = order[start:start + batch]
                w -= self.learning_rate * grad(idx, w)
                if l1 > 0.0:
                    shrink = self.learning_rate * l1
                    w[1:] = np.sign(w[1:]) * np.maximum(np.abs(w[1:]) - shrink,
                                                        0.0)
            cur = loss(w)
            self.history_.append(cur)
            if cur < best_loss - self.tol:
                best_loss, best_w, epochs_since_best = cur, w.copy(), 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= self.patience:
                    break
        return best_w

    # -------------------------------------------------- self-tuning (CV)

    def _tune(self, X_raw, y):
        alphas = self._ALPHA_GRID if self.alpha == "auto" else (self.alpha,)
        degrees = self._DEGREE_GRID if self.degree == "auto" else (self.degree,)

        rng = np.random.default_rng(self.random_state)
        folds = self._stratified_folds(y, min(self.cv, len(y)), rng)
        k = len(folds)

        results = []
        for d in degrees:
            for a in alphas:
                fold_ll = []
                for i in range(k):
                    val = folds[i]
                    train = np.concatenate([folds[j] for j in range(k)
                                            if j != i])
                    m = self._clone(alpha=a, degree=d)
                    m.fit(X_raw[train], y[train])
                    proba = m.predict_proba(X_raw[val])
                    col = np.searchsorted(m.classes_, y[val])
                    p_true = np.clip(proba[np.arange(len(val)), col],
                                     1e-12, None)
                    fold_ll.append(-np.log(p_true).mean())
                se = (float(np.std(fold_ll, ddof=1) / np.sqrt(k))
                      if k > 1 else 0.0)
                results.append({"alpha": a, "degree": d,
                                "cv_log_loss": float(np.mean(fold_ll)),
                                "cv_se": se})

        # one-standard-error rule: simplest model within one SE of the best
        best = min(results, key=lambda r: r["cv_log_loss"])
        threshold = best["cv_log_loss"] + best["cv_se"]
        candidates = [r for r in results if r["cv_log_loss"] <= threshold]
        chosen = min(candidates, key=lambda r: (r["degree"], -r["alpha"]))

        self.alpha, self.degree = chosen["alpha"], chosen["degree"]
        self.alpha_, self.degree_ = chosen["alpha"], chosen["degree"]
        self.cv_results_ = results

    @staticmethod
    def _stratified_folds(y, k, rng):
        """k folds, each preserving the class proportions of y."""
        folds = [[] for _ in range(k)]
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            idx = idx[rng.permutation(len(idx))]
            for i, part in enumerate(np.array_split(idx, k)):
                folds[i].extend(part.tolist())
        return [np.asarray(f) for f in folds]

    def _clone(self, **overrides):
        params = dict(solver=self.solver, penalty=self.penalty,
                      alpha=self.alpha, l1_ratio=self.l1_ratio,
                      degree=self.degree, standardize=self.standardize,
                      class_weight=self.class_weight,
                      learning_rate=self.learning_rate,
                      max_iter=self.max_iter, batch_size=self.batch_size,
                      tol=self.tol, patience=self.patience,
                      newton_max_iter=self.newton_max_iter, cv=self.cv,
                      n_bootstrap=self.n_bootstrap,
                      random_state=self.random_state)
        params.update(overrides)
        return LogisticRegressionPlus(**params)

    # -------------------------------------------------------------- predict

    def predict_proba(self, X):
        X = self._expand(self._as_2d(X))
        if len(self.classes_) == 2:
            p1 = _sigmoid(X @ self.coef_ + self.intercept_)
            return np.column_stack([1.0 - p1, p1])
        return _softmax(X @ self.coef_.T + self.intercept_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        """Accuracy."""
        return float((self.predict(X) == np.asarray(y).ravel()).mean())

    def evaluate(self, X, y):
        """accuracy, precision, recall, F1, log loss (+ ROC-AUC if binary).

        Binary: precision/recall/F1 are for the positive class
        (``classes_[1]``). Multiclass: macro averages.
        """
        y = np.asarray(y).ravel()
        proba = self.predict_proba(X)
        pred = self.predict(X)

        col = np.searchsorted(self.classes_, y)
        p_true = np.clip(proba[np.arange(len(y)), col], 1e-12, None)

        precs, recs, f1s = [], [], []
        for c in self.classes_:
            tp = ((pred == c) & (y == c)).sum()
            fp = ((pred == c) & (y != c)).sum()
            fn = ((pred != c) & (y == c)).sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

        binary = len(self.classes_) == 2
        i = 1 if binary else slice(None)
        out = {
            "accuracy": float((pred == y).mean()),
            "precision": float(precs[1] if binary else np.mean(precs)),
            "recall": float(recs[1] if binary else np.mean(recs)),
            "f1": float(f1s[1] if binary else np.mean(f1s)),
            "log_loss": float(-np.log(p_true).mean()),
        }
        if binary:
            y01 = (y == self.classes_[1]).astype(float)
            out["roc_auc"] = self._roc_auc(y01, proba[:, 1])
        return out

    def roc_curve(self, X, y):
        """(fpr, tpr) points for plotting a ROC curve. Binary only."""
        if len(self.classes_) != 2:
            raise ValueError("roc_curve supports binary models only")
        scores = self.predict_proba(X)[:, 1]
        y01 = (np.asarray(y).ravel() == self.classes_[1]).astype(float)
        order = np.argsort(-scores)
        tps = np.cumsum(y01[order])
        fps = np.cumsum(1.0 - y01[order])
        tpr = np.r_[0.0, tps / max(tps[-1], 1.0)]
        fpr = np.r_[0.0, fps / max(fps[-1], 1.0)]
        return fpr, tpr

    @staticmethod
    def _roc_auc(y01, scores):
        """AUC via the rank / Mann-Whitney formulation (tie-aware)."""
        order = np.argsort(scores)
        ranks = np.empty(len(scores))
        sorted_scores = scores[order]
        rank_vals = np.arange(1, len(scores) + 1, dtype=float)
        i = 0
        while i < len(scores):
            j = i
            while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            rank_vals[i:j + 1] = 0.5 * (i + j) + 1.0
            i = j + 1
        ranks[order] = rank_vals
        n_pos = y01.sum()
        n_neg = len(y01) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[y01 == 1].sum() - n_pos * (n_pos + 1) / 2)
                     / (n_pos * n_neg))

    # ------------------------------------------------ inference / summary

    def coef_stats(self, level=0.95):
        """Per-coefficient uncertainty: stderr, z, p-value, CI. Binary only.

        Wald tests from the Fisher information for exact maximum-likelihood
        fits; bootstrap for penalized or class-weighted fits (which describe
        the regularized, deliberately biased estimator).
        """
        if len(self.classes_) != 2:
            raise ValueError("coef_stats supports binary models only")
        stats = self._inference()
        w = np.r_[self.intercept_, self.coef_]

        if stats["method"] == "wald":
            se = stats["se"]
            z = np.divide(w, se, out=np.zeros_like(w), where=se > 0)
            p = np.array([2.0 * _norm_sf(abs(zi)) for zi in z])
            crit = _norm_crit(level)
            lower, upper = w - crit * se, w + crit * se
        else:
            boot = stats["boot_coefs"]
            se = boot.std(axis=0, ddof=1)
            z = np.divide(w, se, out=np.zeros_like(w), where=se > 0)
            q = 100.0 * (1.0 - level) / 2.0
            lower = np.percentile(boot, q, axis=0)
            upper = np.percentile(boot, 100.0 - q, axis=0)
            frac_pos = (boot > 0).mean(axis=0)
            p = np.maximum(2.0 * np.minimum(frac_pos, 1.0 - frac_pos),
                           1.0 / len(boot))

        return {"term": ["intercept"] + self._feature_names(),
                "coef": w, "stderr": se, "stat": z, "p_value": p,
                "ci_lower": lower, "ci_upper": upper,
                "method": stats["method"]}

    def summary(self, level=0.95):
        """statsmodels-style text summary of the fitted model. Binary only."""
        s = self.coef_stats(level)
        ev = self.evaluate(self._X_raw, self._y)
        pct = f"{level:.0%}"
        lines = [
            f"LogisticRegressionPlus summary  ({s['method']} inference, {pct} CI)",
            "=" * 76,
            (f"n={len(self._y)}  features={len(self.coef_)}  "
             f"accuracy={ev['accuracy']:.4f}  ROC-AUC={ev['roc_auc']:.4f}  "
             f"log-loss={ev['log_loss']:.4f}"),
        ]
        if hasattr(self, "alpha_"):
            lines.append(f"tuned by {self.cv}-fold CV: "
                         f"alpha={self.alpha_:g}, degree={self.degree_}")
        lines += [
            "-" * 76,
            (f"{'term':<14}{'coef':>10}{'stderr':>10}{'stat':>9}"
             f"{'p-value':>12}{'ci_low':>10}{'ci_high':>10}"),
        ]
        for i, name in enumerate(s["term"]):
            lines.append(f"{name:<14}{s['coef'][i]:>10.4f}{s['stderr'][i]:>10.4f}"
                         f"{s['stat'][i]:>9.2f}{s['p_value'][i]:>12.4g}"
                         f"{s['ci_lower'][i]:>10.4f}{s['ci_upper'][i]:>10.4f}")
        lines.append("=" * 76)
        return "\n".join(lines)

    def _inference(self):
        if self._inference_cache is not None:
            return self._inference_cache
        if self._classical_ok:
            self._inference_cache = self._wald_inference()
        else:
            self._inference_cache = self._bootstrap_inference()
        return self._inference_cache

    def _wald_inference(self):
        # Cov(w) ~ inverse Fisher information at the MLE, in original units
        Xb = self._Xb_orig
        p = _sigmoid(Xb @ np.r_[self.intercept_, self.coef_])
        curv = np.clip(p * (1.0 - p), 1e-10, None)
        info = (Xb * curv[:, None]).T @ Xb
        cov = np.linalg.pinv(info)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return {"method": "wald", "se": se}

    def _bootstrap_inference(self):
        rng = np.random.default_rng(self.random_state)
        n = len(self._y)
        boot = np.empty((self.n_bootstrap, len(self.coef_) + 1))
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            # a resample can miss a class entirely; redraw if so
            while len(np.unique(self._y[idx])) < 2:
                idx = rng.integers(0, n, size=n)
            m = self._clone()
            m.fit(self._X_raw[idx], self._y[idx])
            boot[b, 0] = m.intercept_
            boot[b, 1:] = m.coef_
        return {"method": "bootstrap", "boot_coefs": boot}

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
        if self.standardize:
            self.coef_ = w[1:] / self._sigma
            self.intercept_ = float(w[0] - (w[1:] * self._mu / self._sigma).sum())
        else:
            self.coef_ = w[1:].copy()
            self.intercept_ = float(w[0])

    def _store_readable_coefs_multi(self, W):
        if self.standardize:
            self.coef_ = (W[1:] / self._sigma[:, None]).T
            self.intercept_ = W[0] - (W[1:] * (self._mu / self._sigma)[:, None]).sum(axis=0)
        else:
            self.coef_ = W[1:].T.copy()
            self.intercept_ = W[0].copy()
