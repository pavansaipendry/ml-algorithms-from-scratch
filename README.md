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
| [Logistic Regression](logistic_regression/) | done | Newton/IRLS solver, multiclass softmax, class_weight for imbalance, Wald p-values + bootstrap, self-tuning CV, L1/L2/elastic net, polynomial boundaries, ROC-AUC/F1 from scratch |
| [K-Nearest Neighbors](k_nearest_neighbors/) | done | KD-tree from scratch (~4% of points examined), classification + regression, distance weighting, self-tuning k via CV, 4 distance metrics, built-in standardization |
| Naive Bayes | planned | |
| Decision Tree | planned | |
| Random Forest | planned | |
| AdaBoost | planned | |
| Gradient Boosting | planned | |
| Support Vector Machine | planned | |
| K-Means | planned | |
| Gaussian Mixture (EM) | planned | |
| DBSCAN | planned | |
| Hierarchical Clustering | planned | |
| PCA | planned | |
| Neural Network (MLP) | planned | |

## Setup

```bash
pip install -r requirements.txt
cd linear_regression
python demo.py
```
