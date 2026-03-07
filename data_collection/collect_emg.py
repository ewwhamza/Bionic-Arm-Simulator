import serial
import time
import csv
import os

PORT = "COM3"   # change if needed
BAUD = 115200

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

filename = "emg_dataset.csv"

# Ask subject info
subject = input("Enter subject name (e.g. S1, S2, etc): ")
hand = input("Enter hand (LEFT / RIGHT): ")

print("Adjust sensor. Press ENTER to start.")
input()

start_time = time.time()

# Create file with header if it doesn't exist
file_exists = os.path.isfile(filename)

file = open(filename, "a", newline='')
writer = csv.writer(file)

if not file_exists:
    writer.writerow(["subject", "hand", "timestamp", "emg", "label"])


def collect(label, duration):
    print(f"Collecting {label} data...")
    end = time.time() + duration

    while time.time() < end:
        line = ser.readline().decode().strip()

        if line.isdigit():
            timestamp = time.time() - start_time
            writer.writerow([subject, hand, timestamp, line, label])


print("Rest hands for 5 seconds...")
collect("REST", 5)

for i in range(2):

    print("POINT gesture in 3 seconds...")
    time.sleep(3)
    collect("POINT", 10)

    print("CLOSE gesture in 3 seconds...")
    time.sleep(3)
    collect("CLOSE", 10)

    print("OPEN gesture in 3 seconds...")
    time.sleep(3)
    collect("OPEN", 10)

print("Data collection complete.")

file.close()
ser.close()