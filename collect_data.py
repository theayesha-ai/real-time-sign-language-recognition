import cv2
import mediapipe as mp
import pandas as pd
import os
import time

DATA_FILE = "data/sign_language_data.csv"
SAMPLES_PER_SIGN = 300

# We will ADD A and redo C, D, E
NEW_SIGNS = ["A", "C", "D", "E"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

os.makedirs("data", exist_ok=True)

# Load existing dataset
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

    print("\nExisting dataset:")
    print(df["label"].value_counts())

    # Remove OLD C, D and E data so we can collect them again
    df = df[~df["label"].isin(["C", "D", "E"])].reset_index(drop=True)

    print("\nOld C, D, E data removed.")
else:
    df = pd.DataFrame()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()


def collect_sign(label):

    global df

    print("\n" + "=" * 50)
    print(f"GET READY: {label}")
    print("Show ONE HAND only.")
    print("Press SPACE to start collecting")
    print("Press Q to quit")
    print("=" * 50)

    started = False
    samples = 0

    while samples < SAMPLES_PER_SIGN:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            if started:

                landmarks = []

                for landmark in hand.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)

                landmarks.append(label)

                df.loc[len(df)] = landmarks

                samples += 1

        cv2.putText(
            frame,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {samples}/{SAMPLES_PER_SIGN}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if not started:
            cv2.putText(
                frame,
                "Press SPACE to START",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Sign Language Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            started = True
            print(f"▶ Collecting {label}...")

        elif key == ord("q"):
            df.to_csv(DATA_FILE, index=False)
            cap.release()
            cv2.destroyAllWindows()
            print("\nDataset saved!")
            exit()

    print(f"✅ Finished {label}")

    df.to_csv(DATA_FILE, index=False)

    time.sleep(1)


# Collect A, C, D, E
for sign in NEW_SIGNS:
    collect_sign(sign)


cap.release()
cv2.destroyAllWindows()

print("\n🎉 DATA COLLECTION COMPLETE!")

print("\nFinal dataset:")
print(df["label"].value_counts())