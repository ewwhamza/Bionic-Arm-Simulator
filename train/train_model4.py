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

# 2. MVC NORMALIZATION PER SUBJECT
print("Calculating MVC Normalization per subject...")
df["emg_norm"] = 0.0

for (subject, hand), group in df.groupby(["subject", "hand"]):
    # Find Baseline (REST)
    rest_values = group[group["label"] == "REST"]["emg"]
    baseline = rest_values.mean() if len(rest_values) > 0 else group["emg"].mean()
    
    # Find MVC (Maximum Voluntary Contraction - usually from CLOSE)
    close_values = group[group["label"] == "CLOSE"]["emg"]
    
    # We use 95th percentile to ignore sudden noise spikes
    if len(close_values) > 0:
        mvc = np.percentile(close_values, 95)
    else:
        mvc = group["emg"].max()
        
    # Prevent division by zero just in case
    if mvc <= baseline:
        mvc = baseline + 1.0 
        
    # Formula: (Raw - Baseline) / (MVC - Baseline)
    # This scales the data so REST is ~0.0 and Max Squeeze is ~1.0
    df.loc[group.index, "emg_norm"] = (group["emg"] - baseline) / (mvc - baseline)


# 2.5 CLEAN TRANSITION DATA (The Fix)
print("Filtering sloppy transition data...")

# Separate the dataset by label
clean_rest = df[df['label'] == 'REST']

# For POINT, strictly keep data between 15% and 45% effort
clean_point = df[(df['label'] == 'POINT') & (df['emg_norm'] >= 0.15) & (df['emg_norm'] <= 0.45)]

# For CLOSE, strictly keep data that is above 50% effort
clean_close = df[(df['label'] == 'CLOSE') & (df['emg_norm'] > 0.50)]

# Recombine into a clean dataset
df = pd.concat([clean_rest, clean_point, clean_close])

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
        
        if len(window) == WINDOW:
            features = extract_features(window)
            X_temp.append(features)
            y_temp.append(label)

# 4. BALANCE THE DATASET (Undersampling)
print("Balancing the dataset...")
df_features = pd.DataFrame(X_temp)
df_features['label'] = y_temp

min_class_size = df_features['label'].value_counts().min()
balanced_df = df_features.groupby('label').sample(n=min_class_size, random_state=42)

X = balanced_df.drop('label', axis=1).values
y = balanced_df['label'].values

print("Final Balanced Dataset size:", X.shape)
print("Class counts:\n", balanced_df['label'].value_counts())

# 5. TRAIN & EVALUATE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model...")
model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("\nModel Performance:\n")
print(classification_report(y_test, model.predict(X_test)))

# 6. SAVE MODEL
joblib.dump(model, "emg_model4.pkl")
print("\nModel saved as emg_model4.pkl")