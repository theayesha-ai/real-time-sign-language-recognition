# 🤟 Real-Time Sign Language Recognition

A real-time Sign Language Recognition system built using **Python, OpenCV, MediaPipe, and Machine Learning**.

The project uses a webcam to detect a hand, extract hand landmarks using MediaPipe, and classify predefined sign-language gestures in real time.


## 🎥 Demo

### Real-Time Sign Language Recognition Demo

▶️ **[Watch the Demo Video](./demo/sign_language_demo.mp4)**

The 40-second demo shows the system recognizing the trained hand signs through a webcam in real time.


## ✨ Features

- 🤟 Real-time sign language recognition
- 📷 Webcam-based hand detection
- 🖐️ Hand landmark detection using MediaPipe
- 🤖 Machine Learning-based classification
- ⚡ Real-time prediction
- 📊 Custom sign-language dataset
- 🎯 Recognition of 5 custom sign class

## 🤟 Supported Signs

The current model recognizes the following five trained sign classes:

| Sign Label | Gesture |
|------------|---------|
| **A** | ✋ |
| **B** | ☝️ |
| **C** | 🤟 |
| **D** | ✋ |
| **E** | 🖐️ |

> These are the custom gestures used to collect the dataset and train the model.

## 🧠 How It Works

The system follows this pipeline:

text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Feature Extraction
   ↓
Trained Machine Learning Model
   ↓
Predicted Sign