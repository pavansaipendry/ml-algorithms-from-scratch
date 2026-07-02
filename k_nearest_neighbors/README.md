# K-Nearest Neighbors, Upgraded

A from-scratch KNN (`KNeighborsPlus`) for classification **and** regression,
with the infrastructure textbook implementations skip.

## What "upgraded" means

| Weakness of textbook KNN | Upgrade here |
|---|---|
| Scans every training point for every query | **KD-tree from scratch** — median splits on the widest dimension, max-heap search with plane-distance pruning; examines ~4% of 30k points and returns identical neighbors |
| Silently broken by feature scales | **Built-in standardization** — the demo's raw-feature accuracy is 0.45 (coin flip); standardized is 0.94 |
| User must guess k; k=1 memorizes noise | **`k='auto'`**: cross-validation over a k grid, one-standard-error rule preferring *larger* k (smoother model) |
| All k neighbors vote equally | **`weights='distance'`** — makes the model robust to an oversized k: at k=80 on a noisy sine, uniform RMSE is 0.41 vs 0.15 distance-weighted |
| Classification only | **Regression too** (`task='regression'`), plus `predict_proba` from weighted votes |
| Euclidean only | **manhattan, minkowski(p), cosine** (cosine falls back to brute force — it has no KD-tree pruning bound) |

Only dependencies: NumPy and the standard library (`heapq` for the search heap).

## How the KD-tree works

**Build** (recursive): pick the axis with the largest spread, sort the points
along it, split at the median *position* (not value — guarantees both halves
are non-empty), recurse until a leaf holds ≤ `leaf_size` points.

**Query** (k nearest for point q): walk down to q's leaf and fill a max-heap
of the k best candidates (leaves are evaluated with vectorized NumPy). On the
way back up, a subtree on the far side of a split plane can only contain a
closer point if `|q[axis] − split|` is smaller than the current k-th best
distance — for any Lp metric the per-axis gap lower-bounds the true distance —
so most subtrees are pruned without being visited. `examined_fraction_`
reports how much of the training set a query actually touched.

**Prediction.** Neighbors vote (classification) or average (regression),
optionally weighted by 1/distance. Probabilities are normalized weighted votes.

**Self-tuning.** CV (stratified for classification) over
k ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31, 51}; among all k within one standard error
of the best score, choose the **largest** — more neighbors means a smoother,
lower-variance model, the KNN analogue of "prefer the simpler model."

## Usage

```python
from k_nearest_neighbors import KNeighborsPlus

# classification, k chosen by cross-validation
model = KNeighborsPlus(k="auto").fit(X, y)
model.predict(X_new)
model.predict_proba(X_new)
print(model.k_, model.cv_results_)

# regression with distance weighting
model = KNeighborsPlus(k=15, task="regression", weights="distance").fit(X, y)

# raw neighbor lookups
distances, indices = model.kneighbors(X_new)
model.examined_fraction_   # how much of the training set the KD-tree touched
```

## Demo

```bash
python demo.py
```

Four demonstrations: (1) standardization taking accuracy from 0.45 to 0.94
when one feature is on a 1000× scale, (2) CV choosing a smooth k that beats
k=1 on held-out data, (3) distance weighting holding RMSE steady while
uniform weighting degrades 5× as k grows, and (4) the KD-tree matching brute
force exactly while examining 4% of the points. Saves `demo_plot.png`.

Verified against scikit-learn: 100% prediction agreement with
`KNeighborsClassifier`, exact (0.00e+00) match with `KNeighborsRegressor`,
and KD-tree neighbor distances identical to brute force.
