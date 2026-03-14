import serial
import time
import numpy as np
import joblib

# 1. CONFIGURATION & SETUP
PORT = "COM3"   
BAUD = 115200
WINDOW_SIZE = 200

print("Loading emg_model4.pkl...")
try:
    model = joblib.load("emg_model4.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print(f"Connecting to Arduino on {PORT}...")
ser = serial.Serial(PORT, BAUD)
time.sleep(2) 

def extract_features(signal):
    signal = np.array(signal)
    return [np.mean(signal), np.std(signal), np.max(signal), np.min(signal), 
            np.median(signal), np.var(signal), np.sum(np.abs(np.diff(signal))), np.mean(np.abs(signal))]

# 2. CALIBRATION (Baseline & MVC)
print("\n[STEP 1] Stabilizing sensor. Please wait 5 seconds...")
end_time = time.time() + 5
while time.time() < end_time:
    ser.readline() 

print("\n[STEP 2] CALIBRATION 1: Keep your hand completely relaxed (REST).")
print("Collecting baseline, wait 10 seconds")
time.sleep(1)

baseline_data = []
end_time = time.time() + 10
while time.time() < end_time:
    line = ser.readline().decode(errors='ignore').strip()
    if line.isdigit():
        baseline_data.append(float(line))

live_baseline = np.mean(baseline_data)
print(f"Resting baseline locked in: {live_baseline:.2f}")

print("\n[STEP 3] CALIBRATION 2: Squeeze your fist AS HARD AS YOU CAN.")
print("Squeeze in 3... 2... 1... SQUEEZE AND HOLD!")

mvc_data = []
end_time = time.time() + 3 
while time.time() < end_time:
    line = ser.readline().decode(errors='ignore').strip()
    if line.isdigit():
        mvc_data.append(float(line))

live_mvc = np.percentile(mvc_data, 95)
if live_mvc <= live_baseline:
    live_mvc = live_baseline + 1.0

print(f"MVC locked in: {live_mvc:.2f}")
print("You can relax now!")

# 3. LIVE PREDICTION LOOP
print("\n[STEP 4] LIVE PREDICTION STARTING...")
print("Perform gestures: REST, POINT, or CLOSE. Press Ctrl+C to stop.\n")

window = []

ser.reset_input_buffer()

try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        
        if line.isdigit():
            raw_val = float(line)
            
            # Normalize to your MVC
            norm_val = (raw_val - live_baseline) / (live_mvc - live_baseline)
            window.append(norm_val)
            
            if len(window) == WINDOW_SIZE:
                features = extract_features(window)
                prediction = model.predict([features])[0]
                
                avg_effort = np.mean(window) * 100
                print(f"AI Prediction: {prediction:<5} | Average Effort: {avg_effort:>5.1f}%")
                
                window = []
                
                ser.reset_input_buffer()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    ser.close()
    print("Serial port closed.")