import serial
import time
import numpy as np
import joblib
from collections import deque

# -----------------------------
# 1. CONFIGURATION & SETUP
# -----------------------------
PORT = "COM3"   # Update if your Arduino moves to a different port
BAUD = 115200
WINDOW_SIZE = 200

print("Loading emg_model3.pkl...")
try:
    model = joblib.load("emg_model3.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print(f"Connecting to Arduino on {PORT}...")
ser = serial.Serial(PORT, BAUD)
time.sleep(2) # Give the serial connection time to initialize

# -----------------------------
# 2. FEATURE EXTRACTION (Must match training exactly)
# -----------------------------
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

# -----------------------------
# 3. STABILIZE & CALIBRATE
# -----------------------------
print("\n[STEP 1] Stabilizing sensor. Please wait 3 seconds...")
end_time = time.time() + 3
while time.time() < end_time:
    ser.readline() # Read and discard junk data

print("\n[STEP 2] CALIBRATION: Keep your hand completely relaxed (REST).")
print("Collecting baseline in 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("Calibrating for 5 seconds...")

baseline_data = []
end_time = time.time() + 5

while time.time() < end_time:
    line = ser.readline().decode().strip()
    if line.isdigit():
        baseline_data.append(float(line))

if not baseline_data:
    print("Error: No data received during calibration. Check your Arduino.")
    exit()

live_baseline = np.mean(baseline_data)
print(f"Calibration complete! Your live baseline is: {live_baseline:.2f}")
# -----------------------------
# 4. LIVE PREDICTION LOOP (UPDATED)
# -----------------------------
import collections # Add this to the top of your file if not there

print("\n[STEP 3] LIVE PREDICTION STARTING...")
print("Perform gestures: REST, POINT, or CLOSE. Press Ctrl+C to stop.\n")

window = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=5) # Stores the last 5 predictions

try:
    while True:
        line = ser.readline().decode().strip()
        
        if line.isdigit():
            raw_val = float(line)
            norm_val = raw_val - live_baseline
            window.append(norm_val)
            
            # Only predict if our window is full
            if len(window) == WINDOW_SIZE:
                features = extract_features(window)
                raw_prediction = model.predict([features])[0]
                
                # 1. Add to our smoothing buffer
                prediction_buffer.append(raw_prediction)
                
                # 2. Only act if the buffer is full
                if len(prediction_buffer) == 5:
                    # Find the most common prediction in the last 5 frames
                    counter = collections.Counter(prediction_buffer)
                    smoothed_prediction = counter.most_common(1)[0][0]
                    
                    print(f"Prediction: {smoothed_prediction} | Raw: {raw_val} | Norm: {norm_val:.2f}")
                    
                    # 3. Flush the window to force a fresh reading
                    window.clear()
                    prediction_buffer.clear()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    ser.close()
    print("Serial port closed.")