#include <Arduino_BMI270_BMM150.h>

const int WINDOW_SIZE = 50;    // 50 samples = 0.5 seconds at 100 Hz
const int DELAY_MS    = 10;    // 10 ms between samples = 100 Hz

// Buffers to hold one complete gesture window
float ax_buf[WINDOW_SIZE], ay_buf[WINDOW_SIZE], az_buf[WINDOW_SIZE];
float gx_buf[WINDOW_SIZE], gy_buf[WINDOW_SIZE], gz_buf[WINDOW_SIZE];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU failed. Check board selection.");
    while (1);
  }

  Serial.println("=== Gesture Data Collector ===");
  Serial.println("Instructions:");
  Serial.println("  1. Type any character and press Send");
  Serial.println("  2. Wait for 'CAPTURE NOW'");
  Serial.println("  3. Perform your gesture within 0.5 seconds");
  Serial.println("  4. Copy the output row into your label file");
  Serial.println("  5. Repeat 60 times per gesture class");
  Serial.println();
  Serial.println("Ready. Send any character to start.");
}

void loop() {
  if (Serial.available()) {
    while (Serial.available()) Serial.read();  // flush the input

    // Brief pause so you can get your hand ready
    Serial.println("# GET READY...");
    delay(600);
    Serial.println("# CAPTURE NOW — do your gesture!");

    // Collect WINDOW_SIZE samples at ~100 Hz
    for (int i = 0; i < WINDOW_SIZE; i++) {
      // Wait for fresh data from both sensors
      while (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable());
      IMU.readAcceleration(ax_buf[i], ay_buf[i], az_buf[i]);
      IMU.readGyroscope(gx_buf[i], gy_buf[i], gz_buf[i]);
      delay(DELAY_MS);
    }

    // Print all 50 samples as one long CSV row
    // Format: ax[0],ay[0],az[0],gx[0],gy[0],gz[0], ax[1],..., gz[49]
    for (int i = 0; i < WINDOW_SIZE; i++) {
      Serial.print(ax_buf[i], 4); Serial.print(",");
      Serial.print(ay_buf[i], 4); Serial.print(",");
      Serial.print(az_buf[i], 4); Serial.print(",");
      Serial.print(gx_buf[i], 4); Serial.print(",");
      Serial.print(gy_buf[i], 4); Serial.print(",");
      Serial.print(gz_buf[i], 4);
      if (i < WINDOW_SIZE - 1) Serial.print(",");
    }
    Serial.println();  // end of this data row

    Serial.println("# DONE. Send any character for next capture.");
    Serial.println();
  }
}