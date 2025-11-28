# src/evaluate.py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Paths
ROOT = Path(r"C:\Projects\CropDiseaseDetection")
MODEL_DIR = ROOT / "models"
CLASSES_PATH = MODEL_DIR / "classes.json"
TEST_DIR = ROOT / "data" / "processed" / "test"

# Pick the best model file available
if (MODEL_DIR / "final_model.keras").exists():
    MODEL_PATH = MODEL_DIR / "final_model.keras"
elif (MODEL_DIR / "final_model.h5").exists():
    MODEL_PATH = MODEL_DIR / "final_model.h5"
elif (MODEL_DIR / "best_model.h5").exists():
    MODEL_PATH = MODEL_DIR / "best_model.h5"
else:
    raise FileNotFoundError("No model file found in models/ folder.")

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

# Load class mapping
with open(CLASSES_PATH) as f:
    class_map = json.load(f)
idx2class = {v: k for k, v in class_map.items()}

# Test generator
test_gen = ImageDataGenerator(rescale=1./255)
test_flow = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predictions
preds = model.predict(test_flow, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_flow.classes

# Unique labels actually present in test set
unique_labels = np.unique(y_true)
target_names = [idx2class[i] for i in unique_labels]

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


