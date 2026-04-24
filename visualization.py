import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# ── 1. Load the official train/test split ──────────────────────────────────
# Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
train = pd.read_csv('UNSW_NB15_training-set.csv')
test  = pd.read_csv('UNSW_NB15_testing-set.csv')

# ── 2. Pre-processing ──────────────────────────────────────────────────────
DROP = ['id', 'attack_cat']        # remove identifier and multiclass label
TARGET = 'label'                   # binary: 0 = normal, 1 = attack

for df in [train, test]:
    # Encode categorical features (proto, service, state)
    for col in ['proto', 'service', 'state']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X_train = train.drop(columns=DROP + [TARGET])
y_train = train[TARGET]
X_test  = test.drop(columns=DROP + [TARGET])
y_test  = test[TARGET]

# Normalise to [0,1] — important for Naive Bayes
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 3. Train models ────────────────────────────────────────────────────────
models = {
    'Random Forest':   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree':   DecisionTreeClassifier(max_depth=20, random_state=42),
    'Naive Bayes':     GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'report': classification_report(y_test, y_pred, output_dict=True),
        'auc':    roc_auc_score(y_test, y_prob),
        'cm':     confusion_matrix(y_test, y_pred)
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ── 4. Plot confusion matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay(res['cm'], display_labels=['Normal','Attack']).plot(ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

# ── 5. Feature importance (RF and DT only) ────────────────────────────────
feature_names = test.drop(columns=DROP + [TARGET]).columns
for name in ['Random Forest', 'Decision Tree']:
    imp = models[name].feature_importances_
    top10 = pd.Series(imp, index=feature_names).nlargest(10)
    top10.plot(kind='barh', title=f'{name} — top 10 features')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{name.replace(" ","_")}.png', dpi=150)
    plt.show()
