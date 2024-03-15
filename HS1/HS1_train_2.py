####################################################
###########텐서플로우를 활용한 이미지 처리############
####################################################
import tensorflow as tf
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
from tensorflow.keras import layers, Sequential, Input
import sys

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

directory = './HS1'
sample = cv2.imread(os.path.join(directory, 'HS1_fail', 'HS1_0.jpg'))
height, width, channel = sample.shape
# 원본 이미지 사이즈 확인
####################################################
##################opencv와의 차이점##################
####################################################
train_data = utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    # image_size=(height, width),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

val_data = utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    # image_size=(height, width),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

normalization_layer = layers.Rescaling(1./255)

class_names = train_data.class_names
num_classes = len(class_names)

normalized_train = train_data.map(lambda x, y: (normalization_layer(x), y))
normalized_val = val_data.map(lambda x, y: (normalization_layer(x), y))

# for images, labels in train_data.take(1):
#     image = images[0].numpy().astype("uint8")
#     plt.imshow(image)
#     plt.show()
#     break

model = Sequential([
#   layers.Input(shape=(height, width, 3)),
    layers.Input(shape=(256, 256, 3)), 
    layers.Conv2D(16, 10, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 10, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 10, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  normalized_train,
  validation_data=normalized_val,
  epochs=epochs
)

