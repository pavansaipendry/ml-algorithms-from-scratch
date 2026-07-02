# Logistic Regression, Upgraded

A from-scratch logistic regression (`LogisticRegressionPlus`) — binary and
multiclass — that fixes the real weaknesses of the textbook version.

## What "upgraded" means

| Weakness of textbook logistic regression | Upgrade here |
|---|---|
| Gradient descent needs thousands of iterations | **Newton's method (IRLS)** — converges in ~10 iterations, matches sklearn to ~1e-8 |
| Blind to class imbalance: predicts the majority class and looks accurate | **`class_weight='balanced'`** — reweights rare classes (recall 0.38 → 0.89 in the demo) |
| Binary only (or bolted-on one-vs-rest) | **Softmax (multinomial) regression** built in |
| No way to tell real features from noise | **Wald standard errors, z-stats, p-values, CIs** per coefficient (`summary()`), bootstrap fallback for penalized fits |
| User must guess hyperparameters | **Self-tuning**: `alpha='auto'` / `degree='auto'` by stratified k-fold CV with the one-standard-error rule |
| Overfits with many features | **L2 / L1 / elastic net**; L1 via proximal soft-thresholding zeroes features exactly |
| Only linear decision boundaries | **Polynomial feature expansion** — learns curved boundaries |
| Accuracy is the only metric | **Precision, recall, F1, ROC-AUC, log loss, ROC curves** — all from scratch |

Only dependencies: NumPy and the standard library (the normal distribution
for Wald tests is built from `erfc`).

## The math

**Model.** Binary: P(y=1|x) = σ(xᵀw + b), σ(z) = 1/(1+e⁻ᶻ).
Multiclass: P(y=k|x) = softmax(xᵀWₖ + bₖ).

**Objective.** minimize the (optionally class-weighted) cross-entropy:

```
(1/Σsᵢ) Σ sᵢ [ log(1 + e^{zᵢ}) − yᵢ zᵢ ]  +  λ₁‖w‖₁  +  (λ₂/2)‖w‖²
```

**Newton / IRLS.** With p = σ(Xw) and curvature matrix W = diag(p(1−p)):

```
gradient  g = Xᵀ(p − y)/n + λ₂w
Hessian   H = XᵀWX/n + λ₂I
update    w ← w − H⁻¹g          (typically < 15 iterations)
```

**Gradient descent** handles everything Newton can't: L1 terms via a proximal
soft-thresholding step, and multiclass softmax. Mini-batch with early stopping.

**Wald inference.** At the unpenalized MLE, Cov(ŵ) ≈ (XᵀWX)⁻¹ (inverse Fisher
information). SEs are the square roots of the diagonal; z-tests give p-values.
For penalized or class-weighted fits, inference falls back to the bootstrap
(intervals then describe the regularized, deliberately biased estimator).

**Self-tuning.** Stratified k-fold CV (fold class proportions match the data)
over a log grid of alphas and degrees, scored by validation log loss, with the
one-standard-error rule so a more complex model never wins on noise.

## Usage

```python
from logistic_regression import LogisticRegressionPlus

# binary with full inference
model = LogisticRegressionPlus().fit(X, y)
print(model.summary())            # coef, stderr, z, p-value, 95% CI
model.predict_proba(X_new)
model.evaluate(X_test, y_test)    # accuracy, precision, recall, F1, AUC, log loss

# imbalanced data
model = LogisticRegressionPlus(class_weight="balanced").fit(X, y)

# curved decision boundary, self-tuned
model = LogisticRegressionPlus(penalty="l2", alpha="auto", degree="auto").fit(X, y)

# multiclass just works -- pass 3+ classes and it switches to softmax
model = LogisticRegressionPlus().fit(X, y_multiclass)
```

## Demo

```bash
python demo.py
```

Four demonstrations: (1) `class_weight='balanced'` lifting rare-class recall
from 0.38 to 0.89, (2) Wald p-values correctly flagging a pure-noise feature,
(3) CV choosing a quadratic boundary for circular data (accuracy 0.57 → 0.92),
and (4) softmax on three overlapping classes. Saves `demo_plot.png`.

Verified against the reference stack: Newton coefficients match sklearn's
unpenalized fit to ~1e-8, Wald standard errors and p-values match statsmodels
`Logit` to ~1e-16, and ROC-AUC matches sklearn exactly.
