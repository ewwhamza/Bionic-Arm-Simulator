import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading dataset...")
df = pd.read_csv("emg_dataset.csv")

# 1. FIX LABELS (Merge OPEN into REST)
print("Merging OPEN into REST...")
df['label'] = df['label'].replace('OPEN', 'REST')

# 2. NORMALIZE PER SUBJECT + HAND
print("Normalizing EMG values per subject...")
df["emg_norm"] = 0

for (subject, hand), group in df.groupby(["subject", "hand"]):
    rest_values = group[group["label"] == "REST"]["emg"]
    
    if len(rest_values) == 0:
        baseline = group["emg"].mean()
    else:
        baseline = rest_values.mean()
        
    df.loc[group.index, "emg_norm"] = group["emg"] - baseline

# 3. FEATURE EXTRACTION
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

X_temp = []
y_temp = []

print("Creating training windows...")
for label in df["label"].unique():
    subset = df[df["label"] == label]["emg_norm"].values
    
    for i in range(0, len(subset) - WINDOW, WINDOW):
        window = subset[i:i+WINDOW]
        
        # Safety check: only process if the window is exactly 200 samples long
        if len(window) == WINDOW:
            features = extract_features(window)
            X_temp.append(features)
            y_temp.append(label)

# 4. BALANCE THE DATASET (Undersampling)
print("Balancing the dataset...")
# Convert to DataFrame temporarily to make balancing easy
df_features = pd.DataFrame(X_temp)
df_features['label'] = y_temp

# Find the gesture with the smallest number of windows
min_class_size = df_features['label'].value_counts().min()
print(f"Smallest class size is {min_class_size}. Balancing all gestures to match this...")

# Randomly sample 'min_class_size' rows from each gesture
balanced_df = df_features.groupby('label').sample(n=min_class_size, random_state=42)

# Separate back into X and y
X = balanced_df.drop('label', axis=1).values
y = balanced_df['label'].values

print("Final Balanced Dataset size:", X.shape)
print("Class counts:\n", balanced_df['label'].value_counts())

# 5. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. TRAIN MODEL
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

# 7. EVALUATION
pred = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, pred))

# 8. SAVE MODEL

joblib.dump(model, "emg_model3.pkl")
print("\nModel saved as emg_model3.pkl")