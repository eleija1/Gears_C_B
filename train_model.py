
import os
import cv2
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Settings
# -----------------------------
DATASET_PATH = "dataset"
CATEGORIES = ["bad", "good"]
IMG_SIZE = 100

# -----------------------------
# Load Images
# -----------------------------
def load_data():
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        class_num = CATEGORIES.index(category)

        if not os.path.exists(path):
            print(f"Folder not found: {path}")
            continue

        for img_name in os.listdir(path):

            try:
                img_path = os.path.join(path, img_name)

                # Read image in grayscale
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Resize image
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # Flatten image
                data.append(resized_array.flatten())
                labels.append(class_num)

            except Exception as e:
                print("Error:", e)

    return np.array(data), np.array(labels)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading images...")
X, y = load_data()

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
print("Training model...")

model = SVC(kernel="linear", probability=True)

model.fit(X_train, y_train)

# -----------------------------
# Test Model
# -----------------------------
predictions = model.predict(X_test)

print(classification_report(
    y_test,
    predictions,
    target_names=CATEGORIES
))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "gear_model.pkl")

print("Model saved as gear_model.pkl")
