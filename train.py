"""
Wine Cultivar Prediction using Multinomial Logistic Regression
Course: Predictive Analytics

Dataset: Wine Recognition dataset - UCI ML Repository
         (bundled with scikit-learn, so no download is needed)

Pipeline:
  1. Load data & basic EDA
  2. Stratified train/test split
  3. Standardise features + multinomial Logistic Regression (L2), tune C with GridSearchCV
  4. Evaluate: accuracy, macro precision/recall/F1, per-class report, confusion matrix, ROC-AUC (OvR)
  5. Interpret: coefficients per class and odds ratios
  6. Save plots, metrics and the trained model to ./outputs
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
PALETTE = ["#4C78A8", "#F58518", "#54A24B"]

# ------------------------------------------------------------------ 1. DATA
data = load_wine(as_frame=True)
df = data.frame.copy()
class_names = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
X = df.drop(columns="target")
y = df["target"]

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Class counts:")
print(y.value_counts().sort_index())
df.describe().T.to_csv(f"{OUT}/summary_statistics.csv")

# ------------------------------------------------------------------ 2. EDA
labels = y.map(dict(enumerate(class_names)))

plt.figure(figsize=(5.5, 4))
sns.countplot(x=labels, hue=labels, order=class_names, hue_order=class_names, palette=PALETTE, legend=False)
plt.title("Class distribution")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/01_class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, fmt=".1f", cmap="coolwarm", square=True, annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150)
plt.close()

vis = ["alcohol", "flavanoids", "color_intensity", "proline"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, vis):
    sns.boxplot(x=labels, y=X[col], hue=labels, order=class_names, hue_order=class_names,
                palette=PALETTE, legend=False, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/03_feature_boxplots.png", dpi=150)
plt.close()

pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
plt.figure(figsize=(6, 5))
for k, name in enumerate(class_names):
    plt.scatter(pcs[y == k, 0], pcs[y == k, 1], s=30, color=PALETTE[k], label=name)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection of the three cultivars")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_pca_projection.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 3. MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),  # multinomial (softmax) for >2 classes
])
param_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest C:", grid.best_params_["clf__C"], "| CV accuracy:", round(grid.best_score_, 4))

cv_acc = cross_val_score(best, X_train, y_train, cv=cv, scoring="accuracy")
print("5-fold CV accuracy: %.4f +/- %.4f" % (cv_acc.mean(), cv_acc.std()))

# ------------------------------------------------------------------ 4. EVALUATION
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)

metrics = {
    "best_C": grid.best_params_["clf__C"],
    "cv_accuracy_mean": float(cv_acc.mean()),
    "cv_accuracy_std": float(cv_acc.std()),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_precision_macro": float(precision_score(y_test, y_pred, average="macro")),
    "test_recall_macro": float(recall_score(y_test, y_pred, average="macro")),
    "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "test_roc_auc_ovr_macro": float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nTest metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("\n", classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(f"{OUT}/05_confusion_matrix.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 5. INTERPRETATION
coefs = best.named_steps["clf"].coef_  # shape (3, n_features)
coef_df = pd.DataFrame(coefs.T, index=X.columns, columns=class_names)
coef_df.to_csv(f"{OUT}/coefficients_per_class.csv")
odds = np.exp(coef_df).add_prefix("OR_per_SD_")
odds.to_csv(f"{OUT}/odds_ratios_per_class.csv")

order = coef_df.abs().max(axis=1).sort_values(ascending=False).head(10).index[::-1]
coef_df.loc[order].plot(kind="barh", figsize=(9, 7), color=PALETTE, width=0.8)
plt.axvline(0, color="k", lw=0.8)
plt.xlabel("Standardised coefficient (log-odds for that cultivar)")
plt.title("Top 10 features by coefficient size")
plt.tight_layout()
plt.savefig(f"{OUT}/06_feature_coefficients.png", dpi=150)
plt.close()

print("Top 5 features by max |coefficient|:")
print(coef_df.abs().max(axis=1).sort_values(ascending=False).head(5).round(3).to_string())
for k, name in enumerate(class_names):
    top = coef_df[name].sort_values(ascending=False).head(3)
    print(f"{name}: strongest positive drivers ->", ", ".join(f"{i} ({v:.2f})" for i, v in top.items()))

# ------------------------------------------------------------------ 6. SAVE MODEL
joblib.dump(best, f"{OUT}/multinomial_logreg_model.joblib")
print(f"\nAll outputs saved in ./{OUT}/")
