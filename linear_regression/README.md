# Linear Regression, Upgraded

A from-scratch linear regression (`LinearRegressionPlus`) that fixes the real
weaknesses of ordinary least squares (OLS) — including two things even
scikit-learn's `LinearRegression` won't give you: **uncertainty estimates**
and **self-tuning**.

## What "upgraded" means

| Weakness of plain OLS | Upgrade here |
|---|---|
| Says "ŷ = 12.3" with no idea how sure it is | **Prediction intervals** (`predict_interval`) — classical t-intervals for exact OLS, bootstrap otherwise |
| No way to tell real features from noise | **Standard errors, t-stats, p-values, CIs** per coefficient (`summary()`, `coef_stats()`), statsmodels-style |
| User must guess hyperparameters | **Self-tuning**: `alpha='auto'` / `degree='auto'` picked by k-fold CV with the one-standard-error rule (as in glmnet) |
| Overfits when features are many or correlated | **L2 (ridge)** shrinks weights; closed-form or gradient descent |
| Keeps every feature, even useless ones | **L1 (lasso)** / **elastic net** push weights exactly to zero (proximal soft-thresholding) |
| One outlier can wreck the whole fit | **Huber loss** — quadratic for small residuals, linear for large ones |
| Only fits straight lines | **Polynomial feature expansion** (`degree=k`) |
| Sensitive to feature scale | **Built-in standardization**, coefficients converted back to original units |
| Normal equation fails on singular / huge problems | **Pseudo-inverse** closed form + **mini-batch gradient descent** with early stopping |
| No insight into training | Loss `history_`, `cv_results_`, R², adjusted R², RMSE, MAE |

All of it NumPy + standard library only — even the Student's t distribution
(incomplete beta via continued fractions) is implemented from scratch.

## The math

**Model.** ŷ = Xw + b

**Objective.** minimize over w:

```
(1/2n) Σ ℓ(yᵢ − ŷᵢ)  +  λ₁‖w‖₁  +  (λ₂/2)‖w‖²
```

where ℓ is squared loss `r²` or Huber loss (quadratic for |r| ≤ δ, linear beyond),
and the intercept is never penalized.

**Closed form** (squared loss, λ₁ = 0):

```
w = (XᵀX + n·λ₂·I)⁻¹ Xᵀy        # ridge
w = X⁺y                          # λ₂ = 0, pseudo-inverse (stable when XᵀX is singular)
```

**Gradient descent** (everything else): mini-batch updates
`w ← w − η·∇`, where `∇ = Xᵀψ(r)/n + λ₂w` and ψ clips residuals at ±δ for Huber.
The L1 term is handled with a proximal step (soft-thresholding), which is what
lets weights hit exactly zero. Early stopping halts training once the loss stops
improving for `patience` epochs.

**Inference.** For exact OLS fits, the classical formulas:

```
σ̂² = RSS / (n − p − 1)
SE(wⱼ) = √(σ̂² [(XᵀX)⁻¹]ⱼⱼ)          tⱼ = wⱼ / SE(wⱼ)  ~  t(n−p−1)
prediction interval: ŷ₀ ± t* · σ̂ · √(1 + x₀ᵀ(XᵀX)⁻¹x₀)
```

For anything the classical theory doesn't cover (penalized, Huber, or GD fits):
**bootstrap** — refit on `n_bootstrap` resamples, take percentile CIs of the
coefficients, and build prediction intervals from bootstrap predictions plus
resampled residuals. (For penalized fits these intervals describe the
regularized — deliberately biased — estimator.)

**Self-tuning.** k-fold cross-validation over a log grid of alphas (and degrees
1–4 if requested), then the **one-standard-error rule**: choose the simplest
model whose CV error is within one standard error of the best, so a more complex
model never wins on a statistically meaningless improvement.

## Usage

```python
from linear_regression import LinearRegressionPlus

# plain OLS with full statistical inference
model = LinearRegressionPlus().fit(X, y)
print(model.summary())                     # coef, stderr, t, p-value, 95% CI
pred, lo, hi = model.predict_interval(X_new)   # predictions with error bars

# self-tuning: picks alpha and polynomial degree by cross-validation
model = LinearRegressionPlus(penalty="l2", alpha="auto", degree="auto").fit(X, y)
print(model.alpha_, model.degree_, model.cv_results_)

# outlier-robust lasso (inference automatically switches to bootstrap)
model = LinearRegressionPlus(penalty="l1", alpha=0.05, loss="huber").fit(X, y)

model.evaluate(X_test, y_test)   # {'r2', 'adjusted_r2', 'mse', 'rmse', 'mae'}
```

## Demo

```bash
python demo.py
```

Four demonstrations: (1) Huber staying on the true line while outliers drag OLS
off it, (2) p-values correctly flagging a pure-noise feature, (3) CV choosing
the true polynomial degree and regularization strength on its own, and
(4) 95% prediction intervals achieving ~95% coverage on fresh data. Also
cross-checks coefficients against scikit-learn (agreement to ~1e-15); saves
`demo_plot.png`.

Verified against the reference stack: t-distribution p-values and critical
values match scipy to ~1e-13, and standard errors / p-values / CIs match
statsmodels OLS to ~1e-14.
