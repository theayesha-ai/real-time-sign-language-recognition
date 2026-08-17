🖐️ Real-Time Sign Language & Color Recognition

A real-time computer vision project that uses a webcam to recognize predefined hand signs and colors.

🚀 Features

🖐️ Sign Language Recognition

The system recognizes:

- A
- B
- C
- D
- E
- Hello
- Yes
- No
- Thank You
- I Love You

🎨 Color Recognition

The system recognizes:

- 🔴 Red
- 🔵 Blue
- 🟡 Yellow
- 🟢 Green

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Machine Learning
- Computer Vision

## ⚙️ How It Works

1. The webcam captures the user's hand.
2. MediaPipe detects hand landmarks.
3. The trained machine learning model processes the hand landmarks.
4. The system predicts the corresponding hand sign.
5. OpenCV displays the prediction in real time.
6. The application can also detect predefined colors.

🎯 Recognized Signs

| Sign | Prediction |
|---|---|
| A | A |
| B | B |
| C | C |
| D | D |
| E | E |
| 👋 | Hello |
| 👍 | Yes |
| 👎 | No |
| 🙏 | Thank You |
| ❤️ | I Love You |

🎨 Recognized Colors

| Color | Prediction |
|---|---|
| 🔴 | Red |
| 🔵 | Blue |
| 🟡 | Yellow |
| 🟢 | Green |

💻 Project Structure

text
real-time-sign-language-recognition/
│
├── dataset/
├── model/
├── demo/
├── src/
├── requirements.txt
├── README.md
└── ...
