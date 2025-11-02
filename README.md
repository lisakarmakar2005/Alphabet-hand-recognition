# Alphabet-hand-recognition
A real-time alphabet detector that recognizes hand signs (A–Z) using MediaPipe and Machine Learning.
Built completely in Python inside PyCharm.

🚀 Features

✅ Detects A–Z hand signs in real-time using webcam
✅ Built using MediaPipe for hand tracking
✅ Trained with a Random Forest model
✅ Works offline — no internet needed
✅ Simple, fast, and beginner-friendly

🧠 How It Works

collect_data.py → Captures your hand landmarks and saves them to sign_data.csv.

train_model.py → Trains a machine learning model (sign_model.pkl) using the collected data.

main.py → Opens your webcam and predicts the alphabet sign in real-time.
