import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Ask gesture label
label = input("Enter gesture label (Example: A/B/C): ")

# Create CSV file if not exists
file_exists = os.path.isfile("data.csv")

with open("sign.csv", mode="a", newline="") as f:
    writer = csv.writer(f)

    # Write header only once
    if not file_exists:
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

    print("Press 's' to save sample")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                key = cv2.waitKey(1)

                # Save sample
                if key == ord('s'):
                    landmarks.append(label)
                    writer.writerow(landmarks)
                    print(f"{label} sample saved!")

        cv2.putText(
            frame,
            f"Gesture: {label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()