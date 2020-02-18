from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import partial
import os
import numpy as np
import time

# constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64

# create convolutional base
CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS) # MobileNetV2 is particular about input shapes (Has to be square)
base = keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights = 'imagenet'
)
base.trainable = False

# add top layers
model = keras.Sequential([
    base,
    keras.layers.Dropout(0.2),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(5, activation = 'softmax')
])
model.trainable = False
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001 / 1000),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.summary()

# load weights
checkpoint_path = 'weights/W'
try:
    model.load_weights(checkpoint_path)
    print('Weights detected.')
except:
    print('No weights detected.')

# save model
model_version = '1'
model_name = 'flower_classifier'
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)























#
