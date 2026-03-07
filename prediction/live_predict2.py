import serial
import time
import numpy as np
import joblib
from collections import Counter

PORT = "COM3"
BAUD = 115200

print("Loading model...")

model = joblib.load("modelv2.pkl")
scaler = joblib.load("scaler_v2.pkl")

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

print("Starting EMG system...")

# -----------------------
# SENSOR STABILIZATION
# -----------------------

print("Stabilizing sensor for 2 seconds...")

start = time.time()
while time.time() - start < 2:
    ser.readline()

print("Sensor stabilized.")

# -----------------------
# BASELINE CALIBRATION
# -----------------------

print("Keep hand relaxed for 5 seconds for baseline calibration...")

baseline_values = []
start = time.time()

while time.time() - start < 5:

    line = ser.readline().decode().strip()

    if line.isdigit():
        baseline_values.append(int(line))

baseline = np.median(baseline_values)

print("\nBaseline established:", baseline)

# -----------------------
# ESTIMATED GESTURE LEVELS
# -----------------------

point_estimate = baseline + 100
close_estimate = baseline + 500

print("\nEstimated gesture levels:")
print("REST ≈", int(baseline))
print("POINT ≈", int(point_estimate))
print("CLOSE ≈", int(close_estimate))

# -----------------------
# FEATURE FUNCTION
# -----------------------

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
        np.sqrt(np.mean(signal**2))
    ]

    return [features]

# -----------------------
# PARAMETERS
# -----------------------

raw_chunk_size = 10
window_chunks = 5
vote_size = 3

activation_threshold = baseline + 40

raw_buffer = []
feature_buffer = []
prediction_buffer = []

print("\nStarting gesture recognition...\n")

# -----------------------
# LIVE LOOP
# -----------------------

while True:

    line = ser.readline().decode().strip()

    if line.isdigit():

        raw_value = int(line)
        normalized = raw_value - baseline

        raw_buffer.append(normalized)

        # average raw samples
        if len(raw_buffer) >= raw_chunk_size:

            mean_value = np.mean(raw_buffer)
            feature_buffer.append(mean_value)

            raw_buffer = []

        # when prediction window ready
        if len(feature_buffer) >= window_chunks:

            mean_signal = np.mean(feature_buffer)

            if mean_signal < activation_threshold:
                instant_prediction = "REST"

            else:

                features = extract_features(feature_buffer)
                features = scaler.transform(features)

                instant_prediction = model.predict(features)[0]

                if instant_prediction == "OPEN":
                    instant_prediction = "REST"

            prediction_buffer.append(instant_prediction)

            # smoothing vote
            if len(prediction_buffer) >= vote_size:

                stable_prediction = Counter(prediction_buffer).most_common(1)[0][0]

                print(
                    f"Raw:{raw_value} | Norm:{int(normalized)} | Inst:{instant_prediction} | Stable:{stable_prediction}"
                )

                prediction_buffer = []

            feature_buffer = []