# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_FILE = "sign_data.csv"
MODEL_FILE = "sign_model.pkl"

print("📂 Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} samples")

X = df.drop('label', axis=1)
y = df['label']

print("🧠 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Training complete. Accuracy: {accuracy * 100:.2f}%")

print("💾 Saving model...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"🎉 Model saved as {MODEL_FILE}")
