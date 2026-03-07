#include <Arduino.h>

const int emgPin = 35;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int emgValue = analogRead(emgPin);
  Serial.println(emgValue);
  delay(5);  // faster sampling (~200 Hz)
}