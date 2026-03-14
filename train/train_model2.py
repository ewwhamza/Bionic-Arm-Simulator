import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

print("Loading dataset...")

df = pd.read_csv("emg_dataset.csv")

# Convert labels
df["label"] = df["label"].replace("open", "rest")

print("Dataset size:", len(df))

# FEATURE EXTRACTION

window_size = 50

def extract_features(signal):

    signal = np.array(signal)

    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.median(signal),
        np.var(signal),
        np.sum(np.abs(np.diff(signal))),
        np.mean(np.abs(signal)),
        np.max(signal) - np.min(signal)
    ]

    return features


print("Extracting windows...")

X = []
y = []

for label in df["label"].unique():

    subset = df[df["label"] == label]

    emg = subset["emg"].values

    for i in range(0, len(emg) - window_size, window_size):

        window = emg[i:i+window_size]

        features = extract_features(window)

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# NORMALIZATION

scaler = StandardScaler()

X = scaler.fit_transform(X)

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# MODEL

print("Training RandomForest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATION

y_pred = model.predict(X_test)

print("\nModel evaluation:\n")
print(classification_report(y_test, y_pred))

# SAVE

joblib.dump(model, "modelv2.pkl")
joblib.dump(scaler, "scaler_v2.pkl")

print("\nSaved:")
print("modelv2.pkl")
print("scaler_v2.pkl")