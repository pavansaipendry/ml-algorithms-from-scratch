# ML Algorithms From Scratch

Machine learning algorithms implemented from scratch in Python with **NumPy only** —
no `model.fit()` black boxes. scikit-learn appears solely in demos, to verify our
implementations produce the same answers.

Each algorithm lives in its own folder with the implementation, a runnable demo,
and a README explaining the math.

## Algorithms

| Algorithm | Status | Upgrades over the textbook version |
|---|---|---|
| [Linear Regression](linear_regression/) | done | self-tuning via cross-validation (one-SE rule), prediction intervals + p-values (classical & bootstrap), ridge / lasso / elastic net, Huber loss (outlier-robust), polynomial features, closed-form + mini-batch gradient descent, early stopping |
| Logistic Regression | planned | |
| K-Nearest Neighbors | planned | |
| Decision Trees | planned | |
| K-Means | planned | |

## Setup

```bash
pip install -r requirements.txt
cd linear_regression
python demo.py
```
