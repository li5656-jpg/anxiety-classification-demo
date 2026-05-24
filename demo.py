import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
n_samples = 200

SAS = np.random.normal(loc=50, scale=15, size=n_samples).astype(int)
BAI = np.random.normal(loc=50, scale=15, size=n_samples).astype(int)

SAS = np.clip(SAS, 20, 80)
BAI = np.clip(BAI, 20, 80)

total = SAS + BAI
anxiety = (total > 100).astype(int)

df = pd.DataFrame({'SAS': SAS, 'BAI': BAI, 'Total': total, 'Anxiety': anxiety})

X = df[['SAS', 'BAI']].values
y = df['Anxiety'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("=== KNN (k=5) ===")
print(f"准确率: {accuracy_score(y_test, y_pred_knn):.2f}")
print(classification_report(y_test, y_pred_knn))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_knn))

print("\n=== 逻辑回归 ===")
print(f"准确率: {accuracy_score(y_test, y_pred_lr):.2f}")
print(classification_report(y_test, y_pred_lr))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_lr))