# Task 5: Decision Trees and Random Forests
# Internship - AI & ML

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Make sure heart.csv is in the same folder
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)  # Features
y = df["target"]               # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 2. Decision Tree Classifier
# -----------------------------
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()

# -----------------------------
# 3. Random Forest Classifier
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# -----------------------------
# 4. Feature Importance
# -----------------------------
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feat_importances.sort_values().plot(kind="barh", color="skyblue")
plt.title("Feature Importance (Random Forest)", fontsize=14)
plt.xlabel("Importance Score")
plt.show()

# -----------------------------
# 5. Cross Validation
# -----------------------------
scores = cross_val_score(rf, X, y, cv=5)
print("Cross Validation Accuracy:", scores.mean())

