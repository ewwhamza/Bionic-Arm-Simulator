import numpy as np
import joblib

model = joblib.load("emg_model.pkl")

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
        np.mean(np.abs(signal))
    ]

    return [features]


def predict(signal):

    features = extract_features(signal)

    prediction = model.predict(features)[0]

    return prediction


def generate_fake_signal(low, high):

    return np.random.randint(low, high, 200)


print("\nTesting with fake EMG signals\n")


# REST
rest = generate_fake_signal(120, 220)
print("Actual: REST | Predicted:", predict(rest))


# POINT
point = generate_fake_signal(260, 450)
print("Actual: POINT | Predicted:", predict(point))


# CLOSE
close = generate_fake_signal(650, 850)
print("Actual: CLOSE | Predicted:", predict(close))


# OPEN
open_hand = generate_fake_signal(700, 820)
print("Actual: OPEN | Predicted:", predict(open_hand))