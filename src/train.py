# src/train.py (major parts)
import json, os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from model import build_model
from pathlib import Path
import tensorflow as tf

ROOT = Path(r"C:\Projects\CropDiseaseDetection")
TRAIN_DIR = ROOT / "data" / "processed" / "train"
VAL_DIR = ROOT / "data" / "processed" / "val"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224,224)
BATCH = 32

train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
val_flow = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

num_classes = len(train_flow.class_indices)

# save class mapping
with open(MODEL_DIR / "classes.json", "w") as f:
    json.dump(train_flow.class_indices, f)

# compute class weights
labels = train_flow.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: w for i,w in enumerate(class_weights)}

# build model
model = build_model(num_classes=num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(str(MODEL_DIR / "best_model.h5"), monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

history = model.fit(train_flow,
                    validation_data=val_flow,
                    epochs=15,
                    class_weight=class_weights,
                    callbacks=callbacks)

# Optionally unfreeze some layers for fine-tuning
base = model.layers[0]  # ResNet base may be the first layer/Model
base.trainable = True
# unfreeze last N layers
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_flow, validation_data=val_flow, epochs=10, callbacks=callbacks, class_weight=class_weights)

# Save final
model.save(MODEL_DIR / "final_model.h5")

# After fine-tuning / final training
# Save final model in multiple formats
print("Saving final model...")

final_h5_path = MODEL_DIR / "final_model.h5"
final_keras_path = MODEL_DIR / "final_model.keras"

# Save in old HDF5 format
model.save(final_h5_path)

# Save in new .keras format (recommended by TensorFlow >=2.15)
model.save(final_keras_path)

print(f"Model saved as: {final_h5_path}")
print(f"Model also saved as: {final_keras_path}")
