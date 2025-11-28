# src/gradcam.py
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog  # for file chooser

# Paths
ROOT = Path(r"C:\Projects\CropDiseaseDetection")
MODEL_DIR = ROOT / "models"
CLASSES_PATH = MODEL_DIR / "classes.json"

# Pick the best model available
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

# Load class mappings
with open(CLASSES_PATH) as f:
    class_map = json.load(f)
idx2class = {v: k for k, v in class_map.items()}

# --- Grad-CAM function ---
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(model.input, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize to [0,1]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# --- File chooser for test image ---
Tk().withdraw()  # hide main Tk window
file_path = filedialog.askopenfilename(
    initialdir=ROOT / "data" / "processed" / "test",
    title="Select a test image",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
)
if not file_path:
    raise ValueError("No file selected!")

IMAGE = Path(file_path)
print(f"Selected image: {IMAGE}")

# --- Load and preprocess image ---
img = image.load_img(IMAGE, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict class
preds = model.predict(img_array)
pred_class = np.argmax(preds[0])
pred_label = idx2class[pred_class]
print(f"Predicted class: {pred_label} ({preds[0][pred_class]*100:.2f}%)")

# --- Generate Grad-CAM heatmap ---
last_conv_layer_name = [layer.name for layer in model.layers if "conv" in layer.name][-1]
heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)

# --- Overlay heatmap on original image ---
img_cv = cv2.imread(str(IMAGE))
img_cv = cv2.resize(img_cv, (224, 224))
heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

# --- Display results ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Original: {pred_label}")
plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()


