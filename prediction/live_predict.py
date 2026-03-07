import serial
import time
import numpy as np
import joblib

PORT = "COM3"
BAUD = 115200

model = joblib.load("emg_model.pkl")

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

baseline = np.mean(baseline_values)

print("\nBaseline established:", baseline)

# -----------------------
# ESTIMATED LEVELS
# -----------------------

point_estimate = baseline + 200
close_estimate = baseline + 500

print("\nEstimated gesture levels:")
print("REST / OPEN ≈", int(baseline))
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
        np.mean(np.abs(signal))
    ]

    return [features]

# -----------------------
# PARAMETERS
# -----------------------

raw_chunk_size = 20
window_chunks = 10

raw_buffer = []
feature_buffer = []

print("\nStarting gesture recognition...\n")

# -----------------------
# LIVE LOOP
# -----------------------

while True:

    line = ser.readline().decode().strip()

    if line.isdigit():

        raw_value = int(line)

        # normalize
        value = raw_value - baseline

        raw_buffer.append(value)

        if len(raw_buffer) >= raw_chunk_size:

            mean_value = np.mean(raw_buffer)

            feature_buffer.append(mean_value)

            raw_buffer = []

        if len(feature_buffer) >= window_chunks:

            features = extract_features(feature_buffer)

            prediction = model.predict(features)[0]

            # merge REST and OPEN
            if prediction == "OPEN":
                prediction = "REST"

            print(f"Raw EMG: {raw_value} | Prediction: {prediction}")

            feature_buffer = []