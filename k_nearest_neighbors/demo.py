"""Demo: why KNeighborsPlus beats textbook KNN.

Four stories:
  1. Standardization -- unscaled features silently break KNN
  2. Self-tuning k -- CV picks a smooth k instead of overfitting with k=1
  3. Distance weighting -- smoother regression fits near data edges
  4. KD-tree -- examines a few percent of points instead of all of them

Also sanity-checks predictions against scikit-learn, and saves a 4-panel
plot to demo_plot.png.
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from k_nearest_neighbors import KNeighborsPlus

rng = np.random.default_rng(42)


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_moons(n, noise=0.25):
    t = rng.uniform(0, np.pi, n)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (2 * n, 2))
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    order = rng.permutation(2 * n)
    return X[order], y[order]


# --- 1. standardization ------------------------------------------------------
banner("1. STANDARDIZATION: informative feature in [0,1], junk in [0,1000]")

n = 600
info = rng.uniform(0, 1, n)
y1 = (info > 0.5).astype(int)
junk = rng.uniform(0, 1000, n)
X1 = np.column_stack([info + rng.normal(0, 0.08, n), junk])
X1_train, y1_train = X1[:400], y1[:400]
X1_test, y1_test = X1[400:], y1[400:]

raw = KNeighborsPlus(k=7, standardize=False).fit(X1_train, y1_train)
std = KNeighborsPlus(k=7, standardize=True).fit(X1_train, y1_train)
print(f"test accuracy: raw features = {raw.score(X1_test, y1_test):.3f}, "
      f"standardized = {std.score(X1_test, y1_test):.3f}")
print("-> on raw features the junk column dominates every distance.")

# --- 2. self-tuning k ---------------------------------------------------------
banner("2. SELF-TUNING k: noisy two-moons, CV picks a smooth k")

X2, y2 = make_moons(400, noise=0.3)
X2_train, y2_train = X2[:600], y2[:600]
X2_test, y2_test = X2[600:], y2[600:]

overfit = KNeighborsPlus(k=1).fit(X2_train, y2_train)
tuned = KNeighborsPlus(k="auto", random_state=0).fit(X2_train, y2_train)
print(f"k=1:      train accuracy = {overfit.score(X2_train, y2_train):.3f}, "
      f"test accuracy = {overfit.score(X2_test, y2_test):.3f}")
print(f"k={tuned.k_} (CV): train accuracy = "
      f"{tuned.score(X2_train, y2_train):.3f}, "
      f"test accuracy = {tuned.score(X2_test, y2_test):.3f}")
print("-> k=1 memorizes the training set; CV chooses a smoother model.")

# --- 3. distance weighting (regression) ----------------------------------------
banner("3. DISTANCE WEIGHTING: forgives a too-large k (noisy sine wave)")

x3 = np.sort(rng.uniform(0, 4 * np.pi, 250))
y3 = np.sin(x3) + rng.normal(0, 0.25, 250)
x3_test = np.linspace(0, 4 * np.pi, 500)
y3_test = np.sin(x3_test)

print(f"{'k':>4}{'uniform RMSE':>14}{'distance RMSE':>15}")
for k in [15, 40, 80, 120]:
    u = KNeighborsPlus(k=k, task="regression", weights="uniform").fit(x3, y3)
    d = KNeighborsPlus(k=k, task="regression", weights="distance").fit(x3, y3)
    print(f"{k:>4}{u.evaluate(x3_test, y3_test)['rmse']:>14.4f}"
          f"{d.evaluate(x3_test, y3_test)['rmse']:>15.4f}")
print("-> uniform averaging flattens the curve as k grows; distance")
print("   weighting keeps following it. (At small k, uniform is fine.)")

uni = KNeighborsPlus(k=80, task="regression", weights="uniform").fit(x3, y3)
dw = KNeighborsPlus(k=80, task="regression", weights="distance").fit(x3, y3)

# --- 4. KD-tree vs brute force ---------------------------------------------------
banner("4. KD-TREE: 30,000 training points, 200 queries, 5 features")

X4 = rng.normal(size=(30_000, 5))
y4 = (X4[:, :2].sum(axis=1) > 0).astype(int)
Q = rng.normal(size=(200, 5))

brute = KNeighborsPlus(k=10, algorithm="brute").fit(X4, y4)
tree = KNeighborsPlus(k=10, algorithm="kd_tree").fit(X4, y4)

t0 = time.perf_counter()
d_b, i_b = brute.kneighbors(Q)
t_brute = time.perf_counter() - t0
t0 = time.perf_counter()
d_t, i_t = tree.kneighbors(Q)
t_tree = time.perf_counter() - t0

assert np.allclose(np.sort(d_b, axis=1), np.sort(d_t, axis=1)), \
    "KD-tree and brute force disagree!"
print(f"brute force: examined 100.0% of points, {t_brute * 1000:.0f} ms")
print(f"KD-tree:     examined {tree.examined_fraction_:.1%} of points, "
      f"{t_tree * 1000:.0f} ms")
print("-> identical neighbors, a fraction of the work.")

# --- sanity check vs scikit-learn ---------------------------------------------
try:
    from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor)

    sk_c = KNeighborsClassifier(n_neighbors=7).fit(
        (X2_train - X2_train.mean(0)) / X2_train.std(0), y2_train)
    ours_c = KNeighborsPlus(k=7).fit(X2_train, y2_train)
    X2t_std = (X2_test - X2_train.mean(0)) / X2_train.std(0)
    agree = (ours_c.predict(X2_test) == sk_c.predict(X2t_std)).mean()

    sk_r = KNeighborsRegressor(n_neighbors=80).fit(
        ((x3 - x3.mean()) / x3.std()).reshape(-1, 1), y3)
    pred_diff = np.abs(
        uni.predict(x3_test)
        - sk_r.predict(((x3_test - x3.mean()) / x3.std()).reshape(-1, 1))
    ).max()
    print(f"\nsanity vs sklearn: classifier agreement = {agree:.1%}, "
          f"regressor max pred diff = {pred_diff:.2e}")
except ImportError:
    print("\n(scikit-learn not installed -- skipping cross-check)")

# --- plot -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax1, ax2), (ax3, ax4) = axes

gx, gy = np.meshgrid(np.linspace(X2[:, 0].min() - .3, X2[:, 0].max() + .3, 250),
                     np.linspace(X2[:, 1].min() - .3, X2[:, 1].max() + .3, 250))
G = np.column_stack([gx.ravel(), gy.ravel()])
Z = tuned.predict(G).reshape(gx.shape)
ax1.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], alpha=0.2,
             colors=["tab:blue", "tab:orange"])
ax1.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], s=8, alpha=0.5, label="class 0")
ax1.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], s=8, alpha=0.5, label="class 1")
ax1.set_title(f"1. Self-tuned boundary (k={tuned.k_}) on two moons")
ax1.legend(fontsize=8)

ks = [r["k"] for r in tuned.cv_results_]
scores = [r["cv_score"] for r in tuned.cv_results_]
ses = [r["cv_se"] for r in tuned.cv_results_]
ax2.errorbar(ks, scores, yerr=ses, marker="o", ms=4, capsize=3)
ax2.axvline(tuned.k_, color="k", ls=":", lw=1, label=f"chosen k={tuned.k_}")
ax2.set_xlabel("k")
ax2.set_ylabel("cross-validation accuracy")
ax2.set_title("2. CV curve: one-SE rule picks the largest good k")
ax2.legend(fontsize=8)

ax3.scatter(x3, y3, s=10, alpha=0.4, label="noisy data")
ax3.plot(x3_test, y3_test, "k--", lw=1.2, label="true curve")
ax3.plot(x3_test, uni.predict(x3_test), lw=1.8, label="uniform weights")
ax3.plot(x3_test, dw.predict(x3_test), lw=1.8, label="distance weights")
ax3.set_title("3. KNN regression with k=80: uniform flattens, distance doesn't")
ax3.legend(fontsize=8)

bars = ax4.bar(["brute force", "KD-tree"],
               [100.0, 100 * tree.examined_fraction_],
               color=["tab:gray", "tab:green"], width=0.5)
ax4.bar_label(bars, fmt="%.1f%%")
ax4.set_ylabel("% of training points examined per query")
ax4.set_title(f"4. KD-tree work vs brute force "
              f"({t_brute * 1000:.0f} ms vs {t_tree * 1000:.0f} ms)")

fig.tight_layout()
fig.savefig("demo_plot.png", dpi=120)
print("plot saved to demo_plot.png")
