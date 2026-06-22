# IMU Gesture Recognition

This project uses motion data from an Arduino Nano 33 BLE Sense to classify simple gesture movements with embedded machine learning concepts.

The system collects acceleration and gyroscope data, extracts features from short time windows, and runs an embedded classifier for gesture recognition.

## Project Overview

The gesture classes are:

- Shake
- Punch
- Rest

The project demonstrates an Arduino-to-machine-learning workflow for sensor-based classification, including data collection, feature extraction, normalization, model deployment, and real-time inference concepts.

## Current Features

- IMU data collection from acceleration and gyroscope channels.
- Six input channels: ax, ay, az, gx, gy, gz.
- Windowed capture using 50 samples per gesture.
- Feature extraction using mean, standard deviation, RMS, and peak-to-peak values.
- Embedded TensorFlow Lite Micro classifier setup.
- Motion-triggered classification using acceleration magnitude.

## Repository Structure

```text
arduino/
  data_collection/
    Week-4-Data-Collection.ino
  classifier/
    Week-4-AI-Gesture_Model.ino
    model.h
    normalizer.h
data/
  shake_data.txt
  punch_data.txt
  rest_data.txt
```

## Hardware and Tools

- Arduino Nano 33 BLE Sense
- Arduino BMI270/BMM150 IMU library
- Arduino IDE
- TensorFlow Lite Micro
- Python/Jupyter machine learning workflow

## Notes

The Jupyter notebook training workflow can be added once the final notebook file is included in the repository.
