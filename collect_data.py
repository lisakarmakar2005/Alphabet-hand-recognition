# collect_data.py  (Q key fixed version — press ESC to quit)
import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_FILE = "sign_data.csv"
SAMPLES_PER_LABEL = 100  # number of samples per sign

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

# Load existing data if any
if os.path.exists(DATA_FILE):
    df_existing = pd.read_csv(DATA_FILE)
    print(f"📂 Loaded existing data: {len(df_existing)} samples")
else:
    df_existing = None

print("\n=== SIGN LANGUAGE DATA COLLECTION ===")
print("Instructions:")
print("- Press any LETTER key (A–Z) while the camera window is active to record that label.")
print(f"- Each press collects {SAMPLES_PER_LABEL} samples for that label.")
print("- Press ESC anytime to quit.\n")

all_rows = []
labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not detected.")
        break

    image = cv2.flip(frame, 1)
    display = image.copy()
    cv2.putText(display, "Press A–Z to record | Press ESC to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collect Data - Sign Language", display)
    key = cv2.waitKey(1) & 0xFF

    # 27 = ESC key → quit
    if key == 27:
        print("👋 Exiting and saving data...")
        break

    # Handle letter keys
    if key != 255:
        ch = chr(key).upper()
        if ch.isalpha():
            label = ch
            print(f"🖐️ Collecting {SAMPLES_PER_LABEL} samples for: {label}")
            collected = 0

            while collected < SAMPLES_PER_LABEL:
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                image2 = cv2.flip(frame2, 1)
                results2 = hands.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

                if results2.multi_hand_landmarks:
                    for hand_landmarks in results2.multi_hand_landmarks:
                        lm = []
                        for point in hand_landmarks.landmark:
                            lm.extend([point.x, point.y, point.z])
                        all_rows.append(lm)
                        labels.append(label)
                        collected += 1

                        mp_drawing.draw_landmarks(image2, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(image2, f"Label: {label}  Collected: {collected}/{SAMPLES_PER_LABEL}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Collect Data - Sign Language", image2)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    collected = SAMPLES_PER_LABEL
                    break

            print(f"✅ Done collecting for {label}\n")

print("💾 Saving data...")
if all_rows:
    df_new = pd.DataFrame(all_rows, columns=columns)
    df_new['label'] = labels
    if df_existing is not None:
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(DATA_FILE, index=False)
    print(f"✅ Saved {len(all_rows)} new samples to {DATA_FILE}")
else:
    print("⚠️ No new data collected.")

cap.release()
cv2.destroyAllWindows()
