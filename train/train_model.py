import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading dataset...")

df = pd.read_csv("emg_dataset.csv")

# -----------------------------
# NORMALIZE PER SUBJECT + HAND
# -----------------------------

print("Normalizing EMG values per subject...")

df["emg_norm"] = 0

for (subject, hand), group in df.groupby(["subject", "hand"]):

    rest_values = group[group["label"] == "REST"]["emg"]

    if len(rest_values) == 0:
        baseline = group["emg"].mean()
    else:
        baseline = rest_values.mean()

    df.loc[group.index, "emg_norm"] = group["emg"] - baseline


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

WINDOW = 200

def extract_features(signal):

    signal = np.array(signal)

    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.median(signal),
        np.var(signal),
        np.sum(np.abs(np.diff(signal))),
        np.mean(np.abs(signal))
    ]


X = []
y = []

print("Creating training windows...")

for label in df["label"].unique():

    subset = df[df["label"] == label]["emg_norm"].values

    for i in range(0, len(subset) - WINDOW, WINDOW):

        window = subset[i:i+WINDOW]

        features = extract_features(window)

        X.append(features)
        y.append(label)


X = np.array(X)
y = np.array(y)

print("Dataset size:", X.shape)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

print("Training model...")

model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------

pred = model.predict(X_test)

print("\nModel Performance:\n")

print(classification_report(y_test, pred))

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(model, "emg_model.pkl")

print("\nModel saved as emg_model.pkl")