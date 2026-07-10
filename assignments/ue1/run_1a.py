import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

print("=== 1a.1 KKT-Verifikation ===")
w, b = -0.5, -0.5
for x, y in [(-3, +1), (1, -1)]:
    val = y * (w * x + b)
    print(f"  x={x}, y={y}: val={val:.4f} {'OK' if abs(val-1)<1e-9 else 'FAIL'}")
print(f"  Entscheidungsgrenze x={-b/w:.1f}, Margin={2/abs(w):.1f}")

print("\n=== 1a.2 Wine Quality ===")
df = pd.read_csv("winequality-white.csv", sep=";")
df["label"] = df["quality"].apply(lambda q: "low" if q<=4 else ("high" if q>=8 else "medium"))
X = df.drop(columns=["quality","label"]).values
y = df["label"].values
print("Klassen:", pd.Series(y).value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# SVM Linear
print("\n--- SVM Linear ---")
gs = GridSearchCV(SVC(kernel="linear", class_weight="balanced"),
                  {"C": [0.01, 0.1, 1, 10, 100]}, scoring="f1_macro", cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
p = gs.best_estimator_.predict(X_test)
print(f"Bestes C: {gs.best_params_['C']}")
print(f"Accuracy: {accuracy_score(y_test,p):.4f}  Makro-F1: {f1_score(y_test,p,average='macro'):.4f}")
print(classification_report(y_test, p))

# SVM RBF
print("--- SVM RBF ---")
gs2 = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"),
                   {"C":[0.1,1,10,100], "gamma":[0.001,0.01,0.1,1]}, scoring="f1_macro", cv=5, n_jobs=-1)
gs2.fit(X_train, y_train)
p2 = gs2.best_estimator_.predict(X_test)
print(f"Bestes: C={gs2.best_params_['C']}, gamma={gs2.best_params_['gamma']}")
print(f"Accuracy: {accuracy_score(y_test,p2):.4f}  Makro-F1: {f1_score(y_test,p2,average='macro'):.4f}")
print(classification_report(y_test, p2))

# GBT
print("--- Gradient Boosted Trees ---")
gs3 = GridSearchCV(GradientBoostingClassifier(random_state=42),
                   {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.05,0.1,0.2],"subsample":[0.8,1.0]},
                   scoring="f1_macro", cv=5, n_jobs=-1)
gs3.fit(X_train, y_train)
p3 = gs3.best_estimator_.predict(X_test)
print(f"Bestes: {gs3.best_params_}")
print(f"Accuracy: {accuracy_score(y_test,p3):.4f}  Makro-F1: {f1_score(y_test,p3,average='macro'):.4f}")
print(classification_report(y_test, p3))

print("\nFertig!")
