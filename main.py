# main.py
import cv2
import mediapipe as mp
import pickle
import numpy as np

MODEL_FILE = "sign_model.pkl"

# Load trained model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("🎥 Starting Sign Language Detector... (press ESC to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not detected.")
        break

    image = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw the landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # collect the landmarks as features
            lm = []
            for point in hand_landmarks.landmark:
                lm.extend([point.x, point.y, point.z])

            # predict
            X = np.array(lm).reshape(1, -1)
            pred = model.predict(X)[0]

            # display prediction
            cv2.putText(image, f"Detected: {pred}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Sign Language Detector", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Closed.")
