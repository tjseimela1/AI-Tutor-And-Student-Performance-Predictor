# deep_learning.py

import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# =========================
# CREATE MODELS DIRECTORY
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATASET
# =========================
DATA_PATH = "data/xAPI-Edu-Data.csv"

df = pd.read_csv(DATA_PATH)

# =========================
# TARGET VARIABLE
# =========================
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Class"])

joblib.dump(
    target_encoder,
    "models/target_encoder.pkl"
)

# =========================
# ENCODE FEATURES
# =========================
X = df.drop("Class", axis=1).copy()

feature_encoders = {}

for column in X.columns:
    if X[column].dtype == "object":
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))
        feature_encoders[column] = encoder

joblib.dump(
    feature_encoders,
    "models/feature_encoders.pkl"
)

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# BUILD MODEL
# =========================
model = Sequential([
    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        len(np.unique(y)),
        activation="softmax"
    )
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# EVALUATE MODEL
# =========================
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {accuracy:.4f}")

# =========================
# PREDICTIONS
# =========================
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# =========================
# SAVE MODEL
# =========================
model.save(
    "models/student_performance_model.keras"
)

# =========================
# SAVE METADATA
# =========================
metadata = {
    "features": list(df.drop("Class", axis=1).columns),
    "classes": list(target_encoder.classes_)
}

joblib.dump(
    metadata,
    "models/model_metadata.pkl"
)

print("\nModel saved successfully.")
print("Path: models/student_performance_model.keras")
