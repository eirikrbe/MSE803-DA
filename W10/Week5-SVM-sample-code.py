# SVM Classification using Iris Dataset
# --------------------------------------

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ------------------------------------------------
# 1. Load the Iris dataset
# ------------------------------------------------

iris = datasets.load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


# ------------------------------------------------
# 2. Split dataset into training and testing
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------
# 3. Standardise the features
# ------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------
# 4. Create the SVM model
# ------------------------------------------------

svm_model = SVC(
    kernel="linear",
    C=1.0,
    random_state=42
)


# ------------------------------------------------
# 5. Train the model
# ------------------------------------------------

svm_model.fit(X_train, y_train)


# ------------------------------------------------
# 6. Make predictions on TEST dataset
# ------------------------------------------------

y_pred = svm_model.predict(X_test)


# =================================================
# 7. CONFUSION MATRIX
# =================================================

cm = confusion_matrix(y_test, y_pred)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)


# ------------------------------------------------
# 8. Calculate Accuracy
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------
# 9. Calculate Precision
# ------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nPrecision:")
print(f"{precision:.4f}")


# ------------------------------------------------
# 10. Calculate Recall
# ------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nRecall:")
print(f"{recall:.4f}")


# ------------------------------------------------
# 11. Calculate F1 Score
# ------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nF1 Score:")
print(f"{f1:.4f}")


# =================================================
# 12. Classification Report
# =================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# =================================================
# 13. Display Confusion Matrix
# =================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("SVM Confusion Matrix - Iris Test Dataset")
plt.show()


# =================================================
# 14. Predict a new flower
# =================================================

new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = svm_model.predict(new_flower_scaled)

print("\n======================================")
print("NEW FLOWER PREDICTION")
print("======================================")

print("Predicted class:", iris.target_names[prediction[0]])


# =================================================
# 15. ANOVA F-statistic vs Linear SVM, one feature at a time
# =================================================
# Question: does a higher One-Way ANOVA F-statistic go with a wider margin
# and fewer confusion matrix errors?
#
# F is computed on all 150 samples so it matches the ANOVA.xlsx workbook.
# The SVM for each feature uses the same standardised train/test split as above.
# Margin width = 2 / |w| for each pair of species (one-vs-one), in standardised
# feature units, so features with different scales can be compared. It is a
# soft margin (C=1.0): where species overlap, the SVM accepts points inside it.

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_selection import f_classif

F_all, p_all = f_classif(X, y)

rows = []
for j, name in enumerate(iris.feature_names):
    Xtr = X_train[:, [j]]
    Xte = X_test[:, [j]]

    model_j = SVC(kernel="linear", C=1.0, random_state=42).fit(Xtr, y_train)
    pred_j = model_j.predict(Xte)
    cm_j = confusion_matrix(y_test, pred_j)

    margins = []
    for a, b in combinations(range(3), 2):
        mask = np.isin(y_train, [a, b])
        pair = SVC(kernel="linear", C=1.0).fit(Xtr[mask], y_train[mask])
        margins.append(2 / np.abs(pair.coef_[0][0]))

    rows.append({
        "feature": name,
        "F": F_all[j],
        "p": p_all[j],
        "mean margin (std units)": np.mean(margins),
        "test accuracy": accuracy_score(y_test, pred_j),
        "test errors": int(len(y_test) - np.trace(cm_j)),
    })

feature_table = pd.DataFrame(rows).sort_values("F", ascending=False)

print("\n======================================")
print("ANOVA F vs LINEAR SVM (ONE FEATURE EACH)")
print("======================================")
print(feature_table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nAll four features together: accuracy {accuracy:.4f}, "
      f"{int(len(y_test) - np.trace(cm))} test errors")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, label in [
    (axes[0], "test errors", "Test errors (45 test samples)"),
    (axes[1], "mean margin (std units)", "Mean pairwise margin width"),
]:
    ax.scatter(feature_table["F"], feature_table[col])
    for _, r in feature_table.iterrows():
        ax.annotate(r["feature"].replace(" (cm)", ""), (r["F"], r[col]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ANOVA F-statistic (log scale)")
    ax.set_ylabel(label)
plt.tight_layout()
plt.show()



# =================================================
# 16. Group 4: Excel rule vs Linear SVM (sepal length only)
# =================================================
# Same data as Group_4_Python.py (ANOVA.xlsx, 'Iris Data' sheet), same single
# feature (sepal length), so the two accuracies can be compared directly.

from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

g4 = pd.read_excel(Path(__file__).parent / "ANOVA.xlsx", sheet_name="Iris Data")
Xg = g4[["sepal length (cm)"]].values
yg = g4["species"].str.replace("Iris-", "").values
species = ["setosa", "versicolor", "virginica"]


def group4_rule(sepal_length):
    # Group 4 formula: =IF(A2<5.4, "setosa", IF(A2<=6.2, "versicolor", "virginica"))
    return np.where(sepal_length < 5.4, "setosa",
                    np.where(sepal_length <= 6.2, "versicolor", "virginica"))


def g4_svm():
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))


# (a) All 150 samples: how the Excel rule was scored in Group_4_Python.py
svm_all = g4_svm().fit(Xg, yg)
acc_rule_all = accuracy_score(yg, group4_rule(Xg[:, 0]))
acc_svm_all = accuracy_score(yg, svm_all.predict(Xg))

# (b) Same 70/30 split as section 2, scored on the 45 unseen test samples
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
    Xg, yg, test_size=0.30, random_state=42, stratify=yg
)
acc_rule_test = accuracy_score(yg_te, group4_rule(Xg_te[:, 0]))
acc_svm_test = accuracy_score(yg_te, g4_svm().fit(Xg_tr, yg_tr).predict(Xg_te))

# (c) 10-fold cross-validation for the SVM (the rule has no fitting step)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_svm_cv = cross_val_score(g4_svm(), Xg, yg, cv=cv).mean()

print("\n======================================")
print("GROUP 4: EXCEL RULE vs LINEAR SVM (SEPAL LENGTH)")
print("======================================")
print(f"{'Evaluation':<34}{'Excel rule':>12}{'Linear SVM':>12}")
print(f"{'All 150 samples':<34}{acc_rule_all:>12.2%}{acc_svm_all:>12.2%}")
print(f"{'45 test samples (70/30 split)':<34}{acc_rule_test:>12.2%}{acc_svm_test:>12.2%}")
print(f"{'10-fold cross-validation':<34}{'-':>12}{acc_svm_cv:>12.2%}")

print("\nSVM confusion matrix (all 150), rows = actual, columns = predicted:")
print(pd.DataFrame(confusion_matrix(yg, svm_all.predict(Xg), labels=species),
                   index=species, columns=species))

# Where does the SVM put its decision boundaries, in cm?
scaler_g4 = svm_all.named_steps["standardscaler"]
svc_g4 = svm_all.named_steps["svc"]
mu, sd = scaler_g4.mean_[0], scaler_g4.scale_[0]
print("\nSVM class boundaries on sepal length (one-vs-one, cm):")
for k, (a, b) in enumerate(combinations(range(3), 2)):
    boundary = -svc_g4.intercept_[k] / svc_g4.coef_[0][0] * sd + mu
    print(f"  {svc_g4.classes_[a]} vs {svc_g4.classes_[b]}: {boundary:.2f}")
print("Excel rule thresholds: 5.4 (setosa|versicolor) and 6.2 (versicolor|virginica)")



# =================================================
# 17. Group 4 hypothesis (sepal width): ANOVA vs Linear SVM
# =================================================
# H0: average sepal width of setosa = versicolor = virginica
# H1: at least one species has a different average sepal width
#
# ANOVA tests whether the means differ. The SVM tests whether sepal width
# alone is enough to tell the species apart flower by flower.
# Same data as the Group_4_Anova sheet (150 flowers, 'Iris Data' sheet).

from scipy import stats

widths = [g4.loc[g4["species"] == f"Iris-{s}", "sepal width (cm)"] for s in species]
F_w, p_w = stats.f_oneway(*widths)

grand_mean = g4["sepal width (cm)"].mean()
ss_between = sum(len(w) * (w.mean() - grand_mean) ** 2 for w in widths)
ss_total = ((g4["sepal width (cm)"] - grand_mean) ** 2).sum()

print("\n======================================")
print("GROUP 4 ANOVA ON SEPAL WIDTH")
print("======================================")
for s, w in zip(species, widths):
    print(f"  {s:<11} mean {w.mean():.3f}  sd {w.std():.3f}")
print(f"F(2, 147) = {F_w:.2f}, p = {p_w:.2e}  ->  "
      f"{'reject' if p_w < 0.05 else 'fail to reject'} H0 at 0.05")
print(f"Eta squared = {ss_between / ss_total:.3f} "
      "(share of sepal width variation explained by species)")

Xw = g4[["sepal width (cm)"]].values

svm_w_all = g4_svm().fit(Xw, yg)
acc_w_all = accuracy_score(yg, svm_w_all.predict(Xw))

Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yg, test_size=0.30, random_state=42, stratify=yg
)
acc_w_test = accuracy_score(yw_te, g4_svm().fit(Xw_tr, yw_tr).predict(Xw_te))
acc_w_cv = cross_val_score(g4_svm(), Xw, yg, cv=cv).mean()

print("\n======================================")
print("LINEAR SVM ON SEPAL WIDTH ONLY")
print("======================================")
print(f"All 150 samples:               {acc_w_all:.2%}")
print(f"45 test samples (70/30 split): {acc_w_test:.2%}")
print(f"10-fold cross-validation:      {acc_w_cv:.2%}")

cm_w = confusion_matrix(yg, svm_w_all.predict(Xw), labels=species)
print("\nConfusion matrix (all 150), rows = actual, columns = predicted:")
print(pd.DataFrame(cm_w, index=species, columns=species))

ConfusionMatrixDisplay(cm_w, display_labels=species).plot()
plt.title(f"SVM on sepal width - all 150 samples\nAccuracy: {acc_w_all:.2%}")
plt.show()
