"""Demo: why LogisticRegressionPlus beats textbook logistic regression.

Four stories:
  1. Imbalanced data -- class_weight='balanced' rescues the rare class
  2. Inference -- Wald p-values expose which features are real
  3. Self-tuning -- CV picks the polynomial degree for a curved boundary
  4. Multiclass -- softmax regression on 3 classes

Also sanity-checks Newton's method against scikit-learn, and saves a
4-panel plot to demo_plot.png.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegressionPlus

rng = np.random.default_rng(42)


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- 1. imbalanced data -------------------------------------------------------
banner("1. IMBALANCED DATA: 1900 negatives, 100 positives")

X_neg = rng.normal([0, 0], 1.0, size=(1900, 2))
X_pos = rng.normal([1.7, 1.7], 1.0, size=(100, 2))
X1 = np.vstack([X_neg, X_pos])
y1 = np.r_[np.zeros(1900), np.ones(100)]

plain = LogisticRegressionPlus().fit(X1, y1)
balanced = LogisticRegressionPlus(class_weight="balanced").fit(X1, y1)

print(f"{'model':<12}{'accuracy':>10}{'precision':>11}{'recall':>9}"
      f"{'F1':>8}{'ROC-AUC':>9}")
for name, m in [("plain", plain), ("balanced", balanced)]:
    ev = m.evaluate(X1, y1)
    print(f"{name:<12}{ev['accuracy']:>10.3f}{ev['precision']:>11.3f}"
          f"{ev['recall']:>9.3f}{ev['f1']:>8.3f}{ev['roc_auc']:>9.3f}")
print("-> plain looks great on accuracy but misses most positives;")
print("   'balanced' trades a little precision for far higher recall.")

# --- 2. inference: which features actually matter? -----------------------------
banner("2. INFERENCE: log-odds = 0.5 + 1.5*x1 - 1.0*x2 + 0*x3 (x3 is noise)")

X2 = rng.normal(size=(800, 3))
logit = 0.5 + 1.5 * X2[:, 0] - 1.0 * X2[:, 1]
y2 = (rng.uniform(size=800) < 1 / (1 + np.exp(-logit))).astype(int)

m2 = LogisticRegressionPlus().fit(X2, y2)
print(m2.summary())
print("-> x3's p-value is large: the model correctly flags it as noise.")

# --- 3. self-tuning on a curved decision boundary ------------------------------
banner("3. SELF-TUNING: circular boundary, model picks degree by CV")

X3 = rng.uniform(-3, 3, size=(600, 2))
inside = (X3 ** 2).sum(axis=1) < 2.5 ** 2
flip = rng.uniform(size=600) < 0.05  # 5% label noise
y3 = (inside ^ flip).astype(int)

linear3 = LogisticRegressionPlus(penalty="l2", alpha=0.001).fit(X3, y3)
m3 = LogisticRegressionPlus(penalty="l2", alpha="auto", degree="auto",
                            random_state=0).fit(X3, y3)
print(f"chosen by 5-fold CV (one-SE rule): degree={m3.degree_}, "
      f"alpha={m3.alpha_:g}")
print(f"accuracy: straight-line model = {linear3.score(X3, y3):.3f}, "
      f"self-tuned model = {m3.score(X3, y3):.3f}")

# --- 4. multiclass softmax ------------------------------------------------------
banner("4. MULTICLASS: 3 overlapping gaussian blobs, softmax regression")

centers = np.array([[0.0, 0.0], [3.5, 1.0], [1.0, 3.5]])
X4 = np.vstack([rng.normal(c, 1.0, size=(200, 2)) for c in centers])
y4 = np.repeat([0, 1, 2], 200)

m4 = LogisticRegressionPlus(random_state=0).fit(X4, y4)
ev4 = m4.evaluate(X4, y4)
print(f"accuracy={ev4['accuracy']:.3f}  macro-F1={ev4['f1']:.3f}  "
      f"log-loss={ev4['log_loss']:.3f}")

# --- sanity check vs scikit-learn -----------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression as SkLogit

    theirs = SkLogit(penalty=None, tol=1e-10, max_iter=10000).fit(X2, y2)
    diff = max(abs(m2.intercept_ - theirs.intercept_[0]),
               np.abs(m2.coef_ - theirs.coef_[0]).max())
    sk4 = SkLogit(penalty=None, tol=1e-10, max_iter=10000).fit(X4, y4)
    acc_diff = abs(m4.score(X4, y4) - sk4.score(X4, y4))
    print(f"\nsanity vs sklearn: binary coef max diff = {diff:.2e}, "
          f"multiclass accuracy diff = {acc_diff:.4f}")
except ImportError:
    print("\n(scikit-learn not installed -- skipping cross-check)")

# --- plot -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax1, ax2), (ax3, ax4) = axes

for name, m in [("plain", plain), ("balanced", balanced)]:
    fpr, tpr = m.roc_curve(X1, y1)
    ax1.plot(fpr, tpr, label=f"{name} (AUC={m.evaluate(X1, y1)['roc_auc']:.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax1.set_xlabel("false positive rate")
ax1.set_ylabel("true positive rate")
ax1.set_title("1. ROC curves on imbalanced data")
ax1.legend(fontsize=8)

gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
G = np.column_stack([gx.ravel(), gy.ravel()])
P = m3.predict_proba(G)[:, 1].reshape(gx.shape)
ax2.contourf(gx, gy, P, levels=[0, 0.5, 1], alpha=0.2,
             colors=["tab:blue", "tab:orange"])
ax2.contour(gx, gy, P, levels=[0.5], colors="k", linewidths=1.5)
ax2.scatter(X3[y3 == 0, 0], X3[y3 == 0, 1], s=8, alpha=0.5, label="class 0")
ax2.scatter(X3[y3 == 1, 0], X3[y3 == 1, 1], s=8, alpha=0.5, label="class 1")
ax2.set_title(f"2. Self-tuned boundary (degree={m3.degree_}) on circular data")
ax2.legend(fontsize=8)

for d in sorted({r["degree"] for r in m3.cv_results_}):
    pts = [(r["alpha"], r["cv_log_loss"])
           for r in m3.cv_results_ if r["degree"] == d]
    alphas, lls = zip(*pts)
    ax3.plot(alphas, lls, marker="o", ms=3, label=f"degree {d}")
ax3.axvline(m3.alpha_, color="k", ls=":", lw=1,
            label=f"chosen alpha={m3.alpha_:g}")
ax3.set_xscale("log")
ax3.set_xlabel("alpha")
ax3.set_ylabel("cross-validation log loss")
ax3.set_title("3. CV landscape: curved beats straight decisively")
ax3.legend(fontsize=8)

gx4, gy4 = np.meshgrid(np.linspace(-3.5, 7, 200), np.linspace(-3.5, 7, 200))
G4 = np.column_stack([gx4.ravel(), gy4.ravel()])
regions = m4.predict(G4).reshape(gx4.shape)
ax4.contourf(gx4, gy4, regions, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.15,
             colors=["tab:blue", "tab:orange", "tab:green"])
for c, color in zip([0, 1, 2], ["tab:blue", "tab:orange", "tab:green"]):
    ax4.scatter(X4[y4 == c, 0], X4[y4 == c, 1], s=8, alpha=0.5, color=color,
                label=f"class {c}")
ax4.set_title("4. Softmax multiclass decision regions")
ax4.legend(fontsize=8)

fig.tight_layout()
fig.savefig("demo_plot.png", dpi=120)
print("plot saved to demo_plot.png")
