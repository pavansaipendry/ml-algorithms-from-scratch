"""KNeighborsPlus -- an upgraded, from-scratch k-nearest neighbors.

Upgrades over textbook KNN:

  1. KD-tree          : built from scratch; each query prunes most of the
                        training set instead of scanning every point
  2. Distance weights : closer neighbors count more (weights='distance')
  3. Self-tuning      : k='auto' picked by cross-validation, preferring
                        larger (smoother) k via the one-standard-error rule
  4. Standardization  : built in -- textbook KNN silently breaks when
                        features live on different scales
  5. Both tasks       : classification (with predict_proba) and regression
  6. Metrics          : euclidean, manhattan, minkowski(p), cosine

Only dependencies: NumPy and the standard library.
"""

import heapq

import numpy as np

_TASKS = ("classification", "regression")
_WEIGHTS = ("uniform", "distance")
_METRICS = ("euclidean", "manhattan", "minkowski", "cosine")
_ALGORITHMS = ("auto", "kd_tree", "brute")


class _KDNode:
    __slots__ = ("axis", "split", "left", "right", "idx")

    def __init__(self, axis=-1, split=0.0, left=None, right=None, idx=None):
        self.axis = axis
        self.split = split
        self.left = left
        self.right = right
        self.idx = idx


class KNeighborsPlus:
    """K-nearest neighbors, upgraded. Classification or regression.

    Parameters
    ----------
    k : int | 'auto'
        Number of neighbors. 'auto' picks k from a grid by k-fold CV
        (stratified for classification), preferring the largest k within
        one standard error of the best score -- larger k means a smoother,
        simpler model.
    task : 'classification' | 'regression'
    weights : 'uniform' | 'distance'
        'distance' weights each neighbor by 1/distance, so near neighbors
        dominate far ones.
    metric : 'euclidean' | 'manhattan' | 'minkowski' | 'cosine'
    p : float
        Order for the minkowski metric.
    algorithm : 'auto' | 'kd_tree' | 'brute'
        'auto' uses the KD-tree for low-dimensional Lp metrics and brute
        force otherwise (cosine cannot be KD-tree pruned).
    leaf_size : int
        KD-tree leaf size; leaves are evaluated with vectorized NumPy.
    standardize : bool
        Z-score features using training statistics.
    cv : int
        Folds used when tuning k.
    random_state : int | None
        Seed for CV splits.

    Attributes
    ----------
    classes_ : sorted class labels (classification)
    k_, cv_results_ : set when k is tuned by CV
    examined_fraction_ : mean fraction of training points examined per
        query in the most recent KD-tree search (1.0 for brute force)
    """

    _K_GRID = (1, 3, 5, 7, 9, 11, 15, 21, 31, 51)

    def __init__(self, k=5, task="classification", weights="uniform",
                 metric="euclidean", p=2.0, algorithm="auto", leaf_size=32,
                 standardize=True, cv=5, random_state=None):
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}")
        if weights not in _WEIGHTS:
            raise ValueError(f"weights must be one of {_WEIGHTS}")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}")
        if algorithm == "kd_tree" and metric == "cosine":
            raise ValueError("cosine distance cannot be KD-tree pruned; "
                             "use algorithm='brute'")
        if k != "auto" and k < 1:
            raise ValueError("k must be >= 1 or 'auto'")

        self.k = k
        self.task = task
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.standardize = standardize
        self.cv = cv
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        X = self._as_2d(X)
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError("X and y have different numbers of samples")

        if self.standardize:
            self._mu = X.mean(axis=0)
            self._sigma = X.std(axis=0)
            self._sigma[self._sigma == 0.0] = 1.0
            X = (X - self._mu) / self._sigma

        self._Xt = X
        if self.task == "classification":
            self.classes_ = np.unique(y)
            self._y_codes = np.searchsorted(self.classes_, y)
        else:
            self._y = y.astype(float)

        if self.k == "auto":
            self._tune(y)
        if self.k > len(X):
            raise ValueError(f"k={self.k} exceeds n_samples={len(X)}")

        self._algorithm = self.algorithm
        if self._algorithm == "auto":
            self._algorithm = ("kd_tree"
                               if self.metric != "cosine" and X.shape[1] <= 20
                               else "brute")
        if self._algorithm == "kd_tree":
            self._root = self._build_tree(np.arange(len(X)))
        return self

    # ------------------------------------------------------------- KD-tree

    def _build_tree(self, idx):
        if len(idx) <= self.leaf_size:
            return _KDNode(idx=idx)
        pts = self._Xt[idx]
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        order = np.argsort(pts[:, axis], kind="stable")
        mid = len(idx) // 2
        idx_sorted = idx[order]
        # split by position, not value: both halves are always non-empty
        return _KDNode(axis=axis,
                       split=float(self._Xt[idx_sorted[mid], axis]),
                       left=self._build_tree(idx_sorted[:mid]),
                       right=self._build_tree(idx_sorted[mid:]))

    def _query_tree(self, q, k):
        heap = []  # max-heap of (-distance, index); heap[0] is the worst kept

        def visit(node):
            if node.idx is not None:
                dists = self._distances(q[None, :], self._Xt[node.idx])[0]
                self._examined += len(node.idx)
                for d, i in zip(dists, node.idx):
                    if len(heap) < k:
                        heapq.heappush(heap, (-d, i))
                    elif d < -heap[0][0]:
                        heapq.heapreplace(heap, (-d, i))
                return
            diff = q[node.axis] - node.split
            near, far = ((node.left, node.right) if diff <= 0
                         else (node.right, node.left))
            visit(near)
            # |diff| lower-bounds the distance to anything across the split
            # plane for every Lp metric, so the far side can be pruned
            if len(heap) < k or abs(diff) < -heap[0][0]:
                visit(far)

        visit(self._root)
        heap.sort(reverse=True)
        return (np.array([-h[0] for h in heap]),
                np.array([h[1] for h in heap], dtype=int))

    # ------------------------------------------------------------ distances

    def _distances(self, A, B):
        """Pairwise distances between rows of A (m) and B (n) -> (m, n)."""
        if self.metric == "euclidean":
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
            return np.sqrt(np.clip(d2, 0.0, None))
        if self.metric == "manhattan":
            return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
        if self.metric == "minkowski":
            diff = np.abs(A[:, None, :] - B[None, :, :]) ** self.p
            return diff.sum(axis=2) ** (1.0 / self.p)
        # cosine
        an = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - an @ bn.T

    def kneighbors(self, X, k=None):
        """Distances and indices of the k nearest training points.

        Returns ``(distances, indices)``, each of shape (n_queries, k),
        sorted nearest-first.
        """
        k = k or self.k
        Xq = self._transform(X)
        self._examined = 0

        if self._algorithm == "kd_tree":
            out_d = np.empty((len(Xq), k))
            out_i = np.empty((len(Xq), k), dtype=int)
            for row, q in enumerate(Xq):
                out_d[row], out_i[row] = self._query_tree(q, k)
        else:
            dists = self._distances(Xq, self._Xt)
            self._examined = dists.size
            part = np.argpartition(dists, k - 1, axis=1)[:, :k]
            rows = np.arange(len(Xq))[:, None]
            order = np.argsort(dists[rows, part], axis=1)
            out_i = part[rows, order]
            out_d = dists[rows, out_i]

        self.examined_fraction_ = self._examined / (len(Xq) * len(self._Xt))
        return out_d, out_i

    # -------------------------------------------------------------- predict

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is for classification only")
        dists, idx = self.kneighbors(X)
        w = self._neighbor_weights(dists)
        proba = np.zeros((len(idx), len(self.classes_)))
        codes = self._y_codes[idx]
        for c in range(len(self.classes_)):
            proba[:, c] = (w * (codes == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        if self.task == "classification":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        dists, idx = self.kneighbors(X)
        w = self._neighbor_weights(dists)
        return (w * self._y[idx]).sum(axis=1) / w.sum(axis=1)

    def _neighbor_weights(self, dists):
        if self.weights == "uniform":
            return np.ones_like(dists)
        return 1.0 / (dists + 1e-10)

    def score(self, X, y):
        """Accuracy (classification) or R^2 (regression)."""
        y = np.asarray(y).ravel()
        if self.task == "classification":
            return float((self.predict(X) == y).mean())
        residual = y - self.predict(X)
        return float(1.0 - (residual ** 2).sum() / ((y - y.mean()) ** 2).sum())

    def evaluate(self, X, y):
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if self.task == "regression":
            residual = y - pred
            return {
                "r2": float(1.0 - (residual ** 2).sum()
                            / ((y - y.mean()) ** 2).sum()),
                "mse": float((residual ** 2).mean()),
                "rmse": float(np.sqrt((residual ** 2).mean())),
                "mae": float(np.abs(residual).mean()),
            }
        precs, recs, f1s = [], [], []
        for c in self.classes_:
            tp = ((pred == c) & (y == c)).sum()
            fp = ((pred == c) & (y != c)).sum()
            fn = ((pred != c) & (y == c)).sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            precs.append(prec)
            recs.append(rec)
        binary = len(self.classes_) == 2
        return {
            "accuracy": float((pred == y).mean()),
            "precision": float(precs[1] if binary else np.mean(precs)),
            "recall": float(recs[1] if binary else np.mean(recs)),
            "f1": float(f1s[1] if binary else np.mean(f1s)),
        }

    # -------------------------------------------------- self-tuning (CV)

    def _tune(self, y):
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        n_folds = min(self.cv, n)
        if self.task == "classification":
            folds = self._stratified_folds(y, n_folds, rng)
        else:
            order = rng.permutation(n)
            folds = [np.asarray(f) for f in np.array_split(order, n_folds)]

        max_k = min(len(f) for f in
                    [np.concatenate([folds[j] for j in range(n_folds)
                                     if j != i]) for i in range(n_folds)])
        grid = [k for k in self._K_GRID if k <= max_k]

        results = []
        for k in grid:
            scores = []
            for i in range(n_folds):
                val = folds[i]
                train = np.concatenate([folds[j] for j in range(n_folds)
                                        if j != i])
                m = self._clone(k=k)
                m.fit(self._unstandardized(train), y[train])
                scores.append(m.score(self._unstandardized(val), y[val]))
            se = (float(np.std(scores, ddof=1) / np.sqrt(n_folds))
                  if n_folds > 1 else 0.0)
            results.append({"k": k, "cv_score": float(np.mean(scores)),
                            "cv_se": se})

        # one-SE rule, preferring LARGER k: more neighbors = smoother model
        best = max(results, key=lambda r: r["cv_score"])
        threshold = best["cv_score"] - best["cv_se"]
        candidates = [r for r in results if r["cv_score"] >= threshold]
        chosen = max(candidates, key=lambda r: r["k"])

        self.k = chosen["k"]
        self.k_ = chosen["k"]
        self.cv_results_ = results

    def _unstandardized(self, idx):
        # _tune runs after fit standardized _Xt; clones re-standardize on
        # their own training folds, so hand them original-unit data
        if self.standardize:
            return self._Xt[idx] * self._sigma + self._mu
        return self._Xt[idx]

    @staticmethod
    def _stratified_folds(y, k, rng):
        folds = [[] for _ in range(k)]
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            idx = idx[rng.permutation(len(idx))]
            for i, part in enumerate(np.array_split(idx, k)):
                folds[i].extend(part.tolist())
        return [np.asarray(f) for f in folds]

    def _clone(self, **overrides):
        params = dict(k=self.k, task=self.task, weights=self.weights,
                      metric=self.metric, p=self.p, algorithm=self.algorithm,
                      leaf_size=self.leaf_size, standardize=self.standardize,
                      cv=self.cv, random_state=self.random_state)
        params.update(overrides)
        return KNeighborsPlus(**params)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _transform(self, X):
        X = self._as_2d(X)
        if self.standardize:
            return (X - self._mu) / self._sigma
        return X
