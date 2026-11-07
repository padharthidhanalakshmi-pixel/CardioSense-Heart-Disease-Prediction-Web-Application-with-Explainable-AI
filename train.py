"""
Heart Disease Prediction - complete ML pipeline
Dataset analysis -> Preprocessing -> (without / with) Feature Selection
-> DT, NB, LR, SVM, RF -> Evaluation comparison -> save best model + XAI artefacts
"""
import json, warnings, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
RS = 42
FEAT_NAMES = {
 "age":"Age (years)","sex":"Sex (1=M,0=F)","cp":"Chest pain type",
 "trestbps":"Resting blood pressure (mm Hg)","chol":"Serum cholesterol (mg/dl)",
 "fbs":"Fasting blood sugar >120 mg/dl","restecg":"Resting ECG",
 "thalach":"Max heart rate achieved","exang":"Exercise-induced angina",
 "oldpeak":"ST depression (oldpeak)","slope":"Slope of peak ST",
 "ca":"Major vessels coloured","thal":"Thalassemia"}

# ---------------- 1. DATASET ANALYSIS ----------------
df = pd.read_csv("heart.csv")
# This Kaggle copy of UCI codes target 0 = disease; recode so 1 = heart disease present
df["target"] = 1 - df["target"]
print("Shape:", df.shape)
print(df.info())
print(df.describe().T.round(2))
print("Missing values:\n", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
print("Class balance:\n", df.target.value_counts())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df.target.map({0:"No disease",1:"Disease"}).value_counts().plot.bar(ax=ax[0], color=["#27ae60","#c0392b"])
ax[0].set_title("Target distribution"); ax[0].tick_params(axis='x', rotation=0)
ax[1].hist([df[df.target==0].age, df[df.target==1].age], bins=15, label=["No disease","Disease"], color=["#27ae60","#c0392b"])
ax[1].set_title("Age by target"); ax[1].legend()
plt.tight_layout(); plt.savefig("static/eda_target_age.png", dpi=110); plt.close()

corr = df.corr()
plt.figure(figsize=(9, 7)); plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i,j]:.1f}", ha="center", va="center", fontsize=7)
plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig("static/eda_corr.png", dpi=110); plt.close()

# ---------------- 2. PREPROCESSING ----------------
df = df.drop_duplicates()
df.loc[df.ca == 4, "ca"] = df.ca.mode()[0]          # invalid code -> mode
df.loc[df.thal == 0, "thal"] = df.thal.mode()[0]    # invalid code -> mode
for c in ["trestbps", "chol", "thalach", "oldpeak"]:   # IQR outlier capping
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    df[c] = df[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
X, y = df.drop(columns="target"), df.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print("After preprocessing:", df.shape, "train", X_tr.shape, "test", X_te.shape)

MODELS = {
 "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RS),
 "Naive Bayes": GaussianNB(),
 "Logistic Regression": LogisticRegression(max_iter=1000),
 "SVM": SVC(kernel="rbf", probability=True, random_state=RS),
 "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RS)}

# ---------------- 3. FEATURE SELECTION ----------------
chi_scores = pd.Series(chi2(X_tr, y_tr)[0], index=X.columns)
mi_scores = pd.Series(mutual_info_classif(X_tr, y_tr, random_state=RS), index=X.columns)
rank = (chi_scores.rank() + mi_scores.rank()).sort_values(ascending=False)
K = 8
selected = list(rank.index[:K])
print("Selected features:", selected)
fs_table = pd.DataFrame({"Chi2": chi_scores.round(2), "MutualInfo": mi_scores.round(3)}).loc[rank.index]
fs_table["Selected"] = ["Yes" if f in selected else "No" for f in fs_table.index]
plt.figure(figsize=(8, 4)); mi_scores.loc[rank.index].plot.bar(color=["#2c7be5" if f in selected else "#aaa" for f in rank.index])
plt.title("Mutual information score (blue = selected)"); plt.tight_layout(); plt.savefig("static/fs_scores.png", dpi=110); plt.close()

# ---------------- 4 & 5. MODELS + EVALUATION ----------------
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
rows, fitted = [], {}
for setting, cols in [("Without FS (13)", list(X.columns)), (f"With FS ({K})", selected)]:
    for name, m in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(m))])
        cvacc = cross_val_score(pipe, X_tr[cols], y_tr, cv=cv).mean()
        pipe.fit(X_tr[cols], y_tr)
        p = pipe.predict(X_te[cols]); pr = pipe.predict_proba(X_te[cols])[:, 1]
        rows.append({"Setting": setting, "Model": name,
            "Accuracy": accuracy_score(y_te, p), "Precision": precision_score(y_te, p),
            "Recall": recall_score(y_te, p), "F1": f1_score(y_te, p),
            "ROC-AUC": roc_auc_score(y_te, pr), "CV Acc": cvacc})
        fitted[(setting, name)] = (pipe, cols, pr, p)
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("static/results.csv", index=False)

# comparison charts
piv = res.pivot(index="Model", columns="Setting", values="Accuracy")
piv.plot.bar(figsize=(9, 4), color=["#2c7be5", "#f39c12"]); plt.ylim(0.6, 1); plt.ylabel("Test accuracy")
plt.title("Accuracy: with vs without feature selection"); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig("static/compare_fs.png", dpi=110); plt.close()

# best model: highest CV accuracy (more stable than one test split), tie-break F1
best = res.sort_values(["CV Acc", "F1"], ascending=False).iloc[0]
pipe, cols, pr, p = fitted[(best.Setting, best.Model)]
print("BEST:", best.Model, best.Setting)

plt.figure(figsize=(6, 5))
for (s, n), (_, _, prob, _) in fitted.items():
    if s == best.Setting:
        fpr, tpr, _ = roc_curve(y_te, prob); plt.plot(fpr, tpr, label=f"{n} ({roc_auc_score(y_te, prob):.2f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC curves - {best.Setting}"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("static/roc.png", dpi=110); plt.close()

cm = confusion_matrix(y_te, p)
plt.figure(figsize=(4, 3.5)); plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["No disease", "Disease"]); plt.yticks([0, 1], ["No disease", "Disease"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best.Model}")
plt.tight_layout(); plt.savefig("static/confusion.png", dpi=110); plt.close()

# ---------------- 6. EXPLAINABLE AI (global) ----------------
pi = permutation_importance(pipe, X_te[cols], y_te, n_repeats=30, random_state=RS)
gimp = pd.Series(pi.importances_mean, index=cols).sort_values()
plt.figure(figsize=(7, 4)); gimp.plot.barh(color="#2c7be5"); plt.title(f"Global XAI - permutation importance ({best.Model})")
plt.tight_layout(); plt.savefig("static/xai_global.png", dpi=110); plt.close()

joblib.dump({"pipeline": pipe, "features": cols, "all_features": list(X.columns),
             "background": X_tr.median().to_dict(), "model_name": best.Model,
             "setting": best.Setting, "labels": FEAT_NAMES}, "best_model.pkl")
json.dump({"best_model": best.Model, "setting": best.Setting, "selected": selected,
           "metrics": best.to_dict(), "fs_table": fs_table.reset_index().rename(columns={"index": "Feature"}).to_dict("records"),
           "shape_raw": [303, 14], "shape_clean": list(df.shape),
           "class_counts": y.value_counts().to_dict(), "global_importance": gimp.round(4).to_dict()},
          open("static/summary.json", "w"), indent=2, default=str)
print("Saved best_model.pkl")
