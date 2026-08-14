import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Load dataset
# ==============================

df = pd.read_csv("data/sign_language_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

print("Dataset shape:", X.shape)
print("Signs:", sorted(y.unique()))

# ==============================
# 2. Encode labels
# ==============================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==============================
# 3. Split dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# 4. Train model
# ==============================

print("\nTraining sign-language model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# 5. Evaluate
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL ACCURACY:", round(accuracy * 100, 2), "%")
print("==============================\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# ==============================
# 6. Save model
# ==============================

with open("sign_language_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("sign_labels.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model saved as sign_language_model.pkl")
print("✅ Labels saved as sign_labels.pkl")
print("\n🎉 SIGN LANGUAGE MODEL READY!")