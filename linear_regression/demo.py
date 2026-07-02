"""Demo: why LinearRegressionPlus beats plain OLS.

Four stories:
  1. Outlier robustness -- Huber loss stays on the true line, OLS doesn't
  2. Inference -- p-values expose which features are real and which are noise
  3. Self-tuning -- alpha='auto', degree='auto' picked by cross-validation
  4. Prediction intervals -- predictions with honest error bars (+ coverage)

Also sanity-checks the closed-form solver against scikit-learn, and saves
a 4-panel plot to demo_plot.png.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linear_regression import LinearRegressionPlus

rng = np.random.default_rng(42)


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- 1. outlier robustness --------------------------------------------------
banner("1. OUTLIER ROBUSTNESS: true line is y = 4 + 3x, 6% labels corrupted")

n = 200
X1 = rng.uniform(-3, 3, size=(n, 1))
y1_clean = 4 + 3 * X1[:, 0] + rng.normal(0, 1, n)
y1 = y1_clean.copy()
outliers = rng.choice(n, size=12, replace=False)
y1[outliers] += rng.normal(35, 5, size=12)

ols = LinearRegressionPlus(solver="closed_form", standardize=False).fit(X1, y1)
huber = LinearRegressionPlus(loss="huber", huber_delta=1.0,
                             learning_rate=0.1, max_iter=3000,
                             random_state=0).fit(X1, y1)

print(f"{'model':<14}{'intercept':>10}{'slope':>9}{'R^2 (clean y)':>16}")
for name, m in [("plain OLS", ols), ("huber loss", huber)]:
    print(f"{name:<14}{m.intercept_:>10.3f}{m.coef_[0]:>9.3f}"
          f"{m.evaluate(X1, y1_clean)['r2']:>16.4f}")

# --- 2. inference: which features actually matter? --------------------------
banner("2. INFERENCE: y = 4 + 3*x1 - 2*x2 + 0*x3  (x3 is pure noise)")

X2 = rng.normal(size=(200, 3))
y2 = 4 + 3 * X2[:, 0] - 2 * X2[:, 1] + rng.normal(0, 1, 200)

m2 = LinearRegressionPlus().fit(X2, y2)
print(m2.summary())
print("-> x3's p-value is large: the model correctly flags it as noise.")

# --- 3. self-tuning by cross-validation -------------------------------------
banner("3. SELF-TUNING: data is quadratic, model picks alpha & degree itself")

x3 = rng.uniform(-3, 3, 200)
y3 = 1 + 2 * x3 - 0.5 * x3 ** 2 + rng.normal(0, 0.5, 200)

m3 = LinearRegressionPlus(penalty="l2", alpha="auto", degree="auto",
                          random_state=0).fit(x3, y3)


def poly_str(m):
    s = f"{m.intercept_:.3f}"
    for i, c in enumerate(m.coef_, start=1):
        s += f" {'+' if c >= 0 else '-'} {abs(c):.3f}x" + (f"^{i}" if i > 1 else "")
    return s


print(f"chosen by 5-fold CV (one-SE rule): degree={m3.degree_}, alpha={m3.alpha_:g}")
print(f"fit: {poly_str(m3)}   (true: 1 + 2x - 0.5x^2)   R^2={m3.score(x3, y3):.4f}")

# --- 4. prediction intervals -------------------------------------------------
banner("4. PREDICTION INTERVALS: predictions with honest error bars")

# classical intervals on the OLS fit from section 2
X2_test = rng.normal(size=(2000, 3))
y2_test = 4 + 3 * X2_test[:, 0] - 2 * X2_test[:, 1] + rng.normal(0, 1, 2000)
pred, lo, hi = m2.predict_interval(X2_test, level=0.95)
coverage = ((y2_test >= lo) & (y2_test <= hi)).mean()
print(f"classical 95% interval, OLS model:   coverage on 2000 fresh points"
      f" = {coverage:.1%}  (target 95%)")

# bootstrap intervals on the tuned ridge fit from section 3
x3_test = rng.uniform(-3, 3, 2000)
y3_test = 1 + 2 * x3_test - 0.5 * x3_test ** 2 + rng.normal(0, 0.5, 2000)
pred3, lo3, hi3 = m3.predict_interval(x3_test, level=0.95)
coverage3 = ((y3_test >= lo3) & (y3_test <= hi3)).mean()
print(f"bootstrap 95% interval, ridge model: coverage on 2000 fresh points"
      f" = {coverage3:.1%}  (target 95%)")

# --- sanity check vs scikit-learn --------------------------------------------
try:
    from sklearn.linear_model import LinearRegression as SkOLS

    ours = LinearRegressionPlus(solver="closed_form").fit(X1, y1_clean)
    theirs = SkOLS().fit(X1, y1_clean)
    diff = max(abs(ours.intercept_ - theirs.intercept_),
               abs(ours.coef_[0] - theirs.coef_[0]))
    print(f"\nsanity check vs sklearn on clean data: max coef diff = {diff:.2e}")
except ImportError:
    print("\n(scikit-learn not installed -- skipping cross-check)")

# --- plot ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax1, ax2), (ax3, ax4) = axes

grid = np.linspace(-3, 3, 120).reshape(-1, 1)
ax1.scatter(X1, y1, s=14, alpha=0.5, label="data (with outliers)")
ax1.plot(grid, 4 + 3 * grid, "k--", lw=1.5, label="true line")
ax1.plot(grid, ols.predict(grid), lw=2,
         label=f"OLS ({ols.intercept_:.1f} + {ols.coef_[0]:.2f}x)")
ax1.plot(grid, huber.predict(grid), lw=2,
         label=f"Huber ({huber.intercept_:.1f} + {huber.coef_[0]:.2f}x)")
ax1.set_title("1. Outliers drag OLS up; Huber stays on the true line")
ax1.legend(fontsize=8)

g = np.linspace(-3, 3, 120)
gp, glo, ghi = m3.predict_interval(g, level=0.95)
ax2.scatter(x3, y3, s=12, alpha=0.4, label="data")
ax2.fill_between(g, glo, ghi, alpha=0.25, label="95% prediction interval")
ax2.plot(g, gp, lw=2, label=f"self-tuned fit (degree={m3.degree_})")
ax2.set_title("2. Self-tuned ridge fit with 95% prediction band")
ax2.legend(fontsize=8)

for d in sorted({r["degree"] for r in m3.cv_results_}):
    pts = [(r["alpha"], r["cv_mse"]) for r in m3.cv_results_ if r["degree"] == d]
    alphas, mses = zip(*pts)
    ax3.plot(alphas, mses, marker="o", ms=3, label=f"degree {d}")
ax3.axvline(m3.alpha_, color="k", ls=":", lw=1, label=f"chosen alpha={m3.alpha_:g}")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("alpha")
ax3.set_ylabel("cross-validation MSE")
ax3.set_title("3. CV landscape: degree 2 wins, alpha chosen at the minimum")
ax3.legend(fontsize=8)

ax4.plot(huber.history_)
ax4.set_xlabel("epoch")
ax4.set_ylabel("loss")
ax4.set_title("4. Gradient descent convergence (Huber model)")

fig.tight_layout()
fig.savefig("demo_plot.png", dpi=120)
print("plot saved to demo_plot.png")
