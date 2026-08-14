import cv2
import mediapipe as mp
import numpy as np
import pickle

# ==========================================
# LOAD SIGN LANGUAGE MODEL
# ==========================================

with open("sign_language_model.pkl", "rb") as f:
    sign_model = pickle.load(f)

with open("sign_labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("✅ Sign language model loaded!")
print("Signs:", list(label_encoder.classes_))


# ==========================================
# MEDIAPIPE HAND DETECTION
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================
# COLOUR DETECTION
# ==========================================

def detect_color(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "RED": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255]))
        ],

        "GREEN": [
            (np.array([35, 80, 50]), np.array([85, 255, 255]))
        ],

        "BLUE": [
            (np.array([90, 80, 50]), np.array([130, 255, 255]))
        ],

        "YELLOW": [
            (np.array([20, 100, 100]), np.array([35, 255, 255]))
        ]
    }

    detected_color = "NO COLOR"
    largest_area = 0

    for color, ranges in color_ranges.items():

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(hsv, lower, upper)
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > largest_area and area > 1500:
                largest_area = area
                detected_color = color

    return detected_color


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not open camera.")
    exit()

print("✅ Camera started!")
print("Show a hand sign or a coloured object.")
print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read camera.")
        break

    frame = cv2.flip(frame, 1)

    # --------------------------------------
    # MediaPipe hand detection
    # --------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    sign = None

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ----------------------------------
            # Extract 21 landmarks
            # ----------------------------------

            landmarks = []

            for landmark in hand_landmarks.landmark:

                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # ----------------------------------
            # Predict sign
            # ----------------------------------

            if len(landmarks) == 63:

                input_data = np.array(landmarks).reshape(1, -1)

                prediction = sign_model.predict(input_data)[0]

                sign = label_encoder.inverse_transform(
                    [prediction]
                )[0]


    # ======================================
    # COLOUR DETECTION
    # ======================================

    detected_color = detect_color(frame)


    # ======================================
    # DISPLAY RESULT
    # ======================================

    if sign is not None:

        text = f"SIGN: {sign}"

    elif detected_color != "NO COLOR":

        text = f"COLOR: {detected_color}"

    else:

        text = "Show a sign or colored object"


    # ======================================
    # DISPLAY TEXT
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (650, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Sign Language + Color Detection",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()

print("Camera closed.")