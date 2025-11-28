# src/model.py (build_model function)
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def build_model(num_classes, input_shape=(224,224,3), dropout=0.5):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False   # freeze for initial training
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model
