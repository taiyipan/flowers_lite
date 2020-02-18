from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import partial
import os
import numpy as np
import time

# create augmented dataset
TRAIN_DIR = 'split_data/training/'
VAL_DIR = 'split_data/validation/'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale = 1 / 255,
    rotation_range = 45,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    zoom_range = 0.4,
    shear_range = 0.2,
) # data augmentation applied

val_datagen = ImageDataGenerator(
    rescale = 1 / 255
) # no data augmentation

train_generator = train_datagen.flow_from_directory(
    batch_size = BATCH_SIZE,
    directory = TRAIN_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'sparse'
) # 3916 images
total_train = 3916

val_generator = val_datagen.flow_from_directory(
    batch_size = BATCH_SIZE,
    directory = VAL_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'sparse'
) # 407 images
total_val = 407

# create convolutional base
CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS) # MobileNetV2 is particular about input shapes (Has to be square)
base = keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights = 'imagenet'
)
base.trainable = False
# base.summary()

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
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.summary()

# define save weights callback
checkpoint_path = 'weights/W'
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only = True
)

try:
    model.load_weights(checkpoint_path)
    print('Weights detected.')
except:
    print('No weights detected.')

# define tensorboard callback
root_logdir = os.path.join(os.curdir, 'logs')

def get_run_logdir():
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()

tensorboard_callback = keras.callbacks.TensorBoard(run_logdir)

# define nan callback
nan_callback = keras.callbacks.TerminateOnNaN()

# define early stop callback
early_stop_callback = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 4,
    restore_best_weights = True
)

# train model
EPOCHS = 30

history = model.fit(
    train_generator,
    steps_per_epoch = total_train // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = val_generator,
    validation_steps = total_val // BATCH_SIZE,
    callbacks = [
        checkpoint_callback,
        tensorboard_callback,
        nan_callback,
        early_stop_callback
    ]
)

# evaluate model
model.evaluate(val_generator)

























#
