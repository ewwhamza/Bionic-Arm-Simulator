import pandas as pd
import numpy as np
import joblib

model = joblib.load("emg_model.pkl")

df = pd.read_csv("emg_dataset.csv")

def extract_features(signal):

    signal = np.array(signal)

    return [[
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.median(signal),
        np.var(signal),
        np.sum(np.abs(np.diff(signal))),
        np.mean(np.abs(signal))
    ]]


# test each gesture
gestures = ["REST", "POINT", "CLOSE", "OPEN"]

for gesture in gestures:

    sample = df[df["label"] == gesture]["emg"].values[:200]

    features = extract_features(sample)

    prediction = model.predict(features)[0]

    print("Actual:", gesture, "| Predicted:", prediction)