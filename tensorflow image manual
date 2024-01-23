## 텐서플로우 이미지 로드 메뉴얼
import tensorflow as tf
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
directory = './cleandata/(3)'
# 원본 이미지 사이즈 확인
image = cv2.imread('./cleandata/(3)/pass/2023-08-31(4).jpeg')
print(image.shape)
Height, Width, Channel = image.shape
# 이미지 정보 불러오기
train_data = utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(Height, Width), # 원래 이미지의 사이즈를 그대로 사용하였음.
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
    image_size=(Height, Width),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

class_names = train_data.class_names
print(class_names)
# 이미지 실제로 불러와서 확인하기
for images, labels in train_data.take(1):
    image = images[0].numpy().astype("uint8")
    break
plt.imshow(image)
plt.show()
print(image.shape)

# 이미지 배치사이즈 확인하기
for image_batch, labels_batch in train_data:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE
train_data_set = train_data.cache().prefetch(buffer_size=AUTOTUNE)
print(train_data_set)
from tensorflow.keras import layers, Sequential, Input
normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

#######################################

# 1. normalization을 모델 안에 넣는 방법.

model = Sequential([
    layers.Input(shape=(Height, Width, 3)),
    normalization_layer,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

#######################################

# 2. Dataset.map을 사용하는 방법

normalized_train = train_data.map(lambda x, y: (normalization_layer(x), y))
normalized_val = val_data.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_train))
first_image = image_batch[0]
np.max(first_image)
np.min(first_image)

image_batch, labels_batch = next(iter(normalized_val))
first_image = image_batch[0]
np.max(first_image)
np.min(first_image)


model = Sequential([
  layers.Input(shape=(Height, Width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#######################################

# 3. map을 이용하여 generation과 crop을 하는 방법

def data_augmentation(image, label):
    # 회전: -10도에서 10도 사이의 무작위 회전
    angle = tf.random.uniform([], minval=-10, maxval=10, dtype=tf.float32)
    image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))
    
    # 이동: x와 y 축으로 각각 -10에서 10 사이의 무작위 픽셀 이동
    dx = tf.random.uniform([], minval=-10, maxval=10, dtype=tf.float32)
    dy = tf.random.uniform([], minval=-10, maxval=10, dtype=tf.float32)
    image = tf.image.transform(image, [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0])

    # 밝기 조정: 최대 ±10%의 밝기 조정
    delta = tf.random.uniform([], minval=-0.1, maxval=0.1, dtype=tf.float32)
    image = tf.image.adjust_brightness(image, delta)

    return image, label

def image_crop(image, label):
    # 이미지 크롭 작업 수행
    # 예: 이미지의 일부 영역을 크롭 (예: (400, 400)에서 (600, 700) 영역을 크롭)
    cropped_image = tf.image.crop_to_bounding_box(image, 400, 400, 200, 300)
    return cropped_image, label

# 데이터셋에 이미지 크롭 함수 적용
augmented_train_data = train_data.map(data_augmentation)
cropped_train_data = augmented_train_data.map(image_crop)

#######################################
# 4. 함수형 모델을 쓰는 방법

def functional_model(HEIGHT, WIDTH, outdim):
    # Define the input layer
    inputs = Input(shape=(HEIGHT, WIDTH, 3))

    # 첫번째 conv+pooling layer
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 두번째 conv+pooling layer
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 세번째 conv+pooling layer
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'gradlayer')(x)

    
    x = Flatten()(x)

    # Dense layer with 512 units
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # Output layer with softmax activation
    outputs = Dense(outdim, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = functional_model(Height, Width, num_classes)

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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
