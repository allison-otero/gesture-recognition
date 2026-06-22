// ── Library includes ──────────────────────────────────────────────────────────
#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ── Generated headers — must be in the same folder as this .ino file ─────────
#include "model.h"       // model_data[] byte array
#include "normalizer.h"  // NUM_FEATURES, FEATURE_MEAN[], FEATURE_STD[]

// ── Gesture class names — must match label order used during training ─────────
// 0=shake, 1=punch, 2=rest — same order as CLASS_NAMES in Python
const int   NUM_CLASSES          = 3;
const char* CLASS_NAMES[NUM_CLASSES] = {"shake", "punch", "rest"};

// ── Data capture parameters — must match Python collection settings ───────────
const int WINDOW_SIZE  = 50;    // samples per gesture window
const int NUM_CHANNELS = 6;     // ax, ay, az, gx, gy, gz
const int DELAY_MS     = 10;    // 10 ms = 100 Hz sample rate

// ── Motion detection threshold ────────────────────────────────────────────────
// At rest, gravity alone gives ~1.0 g. A gesture pushes this above 1.5 g.
// Lower this if your gestures aren't triggering capture.
const float MOTION_THRESHOLD = 1.5f;

// ── TFLite runtime objects ────────────────────────────────────────────────────
// TENSOR_ARENA_SIZE: scratch memory for TFLite. Must be large enough for your
// model. Increase to 12*1024 if you see "tensor allocation failed" at startup.
const int TENSOR_ARENA_SIZE = 10 * 1024;
uint8_t   tensor_arena[TENSOR_ARENA_SIZE];

tflite::AllOpsResolver     resolver;
const tflite::Model*       tflite_model  = nullptr;
tflite::MicroInterpreter*  interpreter   = nullptr;
TfLiteTensor*              input_tensor  = nullptr;
TfLiteTensor*              output_tensor = nullptr;

// ── IMU data buffer ───────────────────────────────────────────────────────────
// Stores one captured gesture window: [sample_index][channel_index]
float imu_window[WINDOW_SIZE][NUM_CHANNELS];

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction — must exactly mirror Python's extract_features() function.
// Output order: [ax_mean, ax_std, ax_rms, ax_p2p,
//                ay_mean, ay_std, ay_rms, ay_p2p, ..., gz_p2p]
// ─────────────────────────────────────────────────────────────────────────────
void extractFeatures(float* features) {
  int feat_idx = 0;

  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum    = 0.0f;
    float sq_sum = 0.0f;
    float min_v  =  1e9f;
    float max_v  = -1e9f;

    // First pass: mean, rms numerator, min, max
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float v = imu_window[i][ch];
      sum    += v;
      sq_sum += v * v;
      if (v < min_v) min_v = v;
      if (v > max_v) max_v = v;
    }
    float mean = sum / WINDOW_SIZE;
    float rms  = sqrt(sq_sum / WINDOW_SIZE);

    // Second pass: standard deviation
    float var_sum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float diff = imu_window[i][ch] - mean;
      var_sum   += diff * diff;
    }
    float std_dev = sqrt(var_sum / WINDOW_SIZE);
    float p2p     = max_v - min_v;

    features[feat_idx++] = mean;
    features[feat_idx++] = std_dev;
    features[feat_idx++] = rms;
    features[feat_idx++] = p2p;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply the same normalization as Python's StandardScaler.
// Each feature is shifted and scaled so the model sees the same distribution
// it was trained on.
// ─────────────────────────────────────────────────────────────────────────────
void normalizeFeatures(float* features) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("=== Gesture Classifier Starting Up ===");

  // ── Initialize IMU ──────────────────────────────────────────────────────────
  if (!IMU.begin()) {
    Serial.println("FATAL: IMU initialization failed. Check board selection.");
    while (1);
  }
  Serial.println("[OK] IMU initialized.");

  // ── Initialize TFLite ───────────────────────────────────────────────────────
  tflite_model = tflite::GetModel(model_data);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("FATAL: TFLite model version mismatch.");
    while (1);
  }

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FATAL: TFLite tensor allocation failed.");
    Serial.println("       Increase TENSOR_ARENA_SIZE and re-upload.");
    while (1);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.print("[OK] TFLite ready. Tensor arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes.");
  Serial.println();
  Serial.println("Waiting for motion...");
  Serial.println("(Shake, punch, or rest — the board will classify each gesture)");
  Serial.println();
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  float ax, ay, az;

  // ── Poll for significant motion ───────────────────────────────────────────
  // Total acceleration magnitude at rest ≈ 1.0 g (gravity alone).
  // Any gesture pushes this significantly above 1.0 g.
  if (!IMU.accelerationAvailable()) {
    delay(5);
    return;
  }
  IMU.readAcceleration(ax, ay, az);
  float accel_magnitude = sqrt(ax*ax + ay*ay + az*az);

  if (accel_magnitude < MOTION_THRESHOLD) {
    delay(5);
    return;   // no meaningful motion — keep polling
  }

  // ── Motion detected — capture a full window ───────────────────────────────
  Serial.print("Motion detected (magnitude=");
  Serial.print(accel_magnitude, 2);
  Serial.println("g). Capturing...");

  float gx, gy, gz;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    while (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable());
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    imu_window[i][0] = ax;
    imu_window[i][1] = ay;
    imu_window[i][2] = az;
    imu_window[i][3] = gx;
    imu_window[i][4] = gy;
    imu_window[i][5] = gz;
    delay(DELAY_MS);
  }

  // ── Extract features ──────────────────────────────────────────────────────
  float features[NUM_FEATURES];
  extractFeatures(features);
  normalizeFeatures(features);

  // ── Copy features into the TFLite input tensor ────────────────────────────
  for (int i = 0; i < NUM_FEATURES; i++) {
    input_tensor->data.f[i] = features[i];
  }

  // ── Run inference ─────────────────────────────────────────────────────────
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: TFLite inference failed.");
    return;
  }

  // ── Read output probabilities and find winner ─────────────────────────────
  float max_confidence  = 0.0f;
  int   predicted_class = 0;

  Serial.print("Scores: ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    float conf = output_tensor->data.f[i];
    Serial.print(CLASS_NAMES[i]);
    Serial.print("=");
    Serial.print(conf * 100, 1);
    Serial.print("%  ");
    if (conf > max_confidence) {
      max_confidence  = conf;
      predicted_class = i;
    }
  }
  Serial.println();

  Serial.print(">>> GESTURE: ");
  Serial.print(CLASS_NAMES[predicted_class]);
  Serial.print("  (");
  Serial.print(max_confidence * 100, 1);
  Serial.println("% confident)");
  Serial.println();

  // ── Cooldown period ───────────────────────────────────────────────────────
  // Prevents re-triggering on the tail of the same gesture.
  delay(800);
}