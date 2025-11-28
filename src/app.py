from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2, os, json
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CLASSES_PATH = MODEL_DIR / "classes.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

STATIC_DIR.mkdir(exist_ok=True)

# Pick best model available
if (MODEL_DIR / "final_model.keras").exists():
    MODEL_PATH = MODEL_DIR / "final_model.keras"
elif (MODEL_DIR / "final_model.h5").exists():
    MODEL_PATH = MODEL_DIR / "final_model.h5"
else:
    MODEL_PATH = MODEL_DIR / "best_model.h5"

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

# Load class names
with open(CLASSES_PATH) as f:
    class_map = json.load(f)
idx2class = {v: k for k, v in class_map.items()}

# Remedies dictionary (short suggestions)
remedies = {
    "Tomato___Late_blight": "Apply fungicides (e.g., Mancozeb) and avoid waterlogging.",
    "Tomato___Early_blight": "Remove infected leaves, rotate crops, and apply fungicides.",
    "Tomato___Leaf_Mold": "Improve air circulation and use resistant varieties.",
    "Potato___Late_blight": "Use copper-based fungicides and ensure good field drainage.",
    "Potato___Early_blight": "Practice crop rotation and apply fungicides early.",
    "Apple___Black_rot": "Prune infected branches and apply protective fungicides.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove infected leaves and spray appropriate fungicide.",
    "Corn_(maize)___Common_rust": "Use resistant hybrids and apply fungicides when needed.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops and use resistant hybrids.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Avoid continuous corn planting and use resistant varieties."
    # Add more if needed, unknown classes will get "General good practices..."
}

# Grad-CAM function
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

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# Flask app
app = Flask(__name__, static_folder=str(STATIC_DIR))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filepath = STATIC_DIR / file.filename
        file.save(filepath)

        # Preprocess
        img = image.load_img(filepath, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])
        pred_label = idx2class[pred_class]
        confidence = preds[0][pred_class] * 100

        # Remedy lookup
        remedy_text = remedies.get(pred_label, "Practice crop rotation, remove infected parts, and consult experts if disease persists.")

        # Grad-CAM
        last_conv_layer_name = [layer.name for layer in model.layers if "conv" in layer.name][-1]
        heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)

        img_cv = cv2.imread(str(filepath))
        img_cv = cv2.resize(img_cv, (224, 224))
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

        gradcam_filename = f"gradcam_{file.filename}"
        gradcam_path = STATIC_DIR / gradcam_filename
        cv2.imwrite(str(gradcam_path), superimposed_img)

        return render_template("index.html",
                               prediction=pred_label,
                               confidence=f"{confidence:.2f}%",
                               remedy=remedy_text,
                               user_image=f"static/{file.filename}",
                               gradcam_image=f"static/{gradcam_filename}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


