import os
import sys
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

datagen = ImageDataGenerator(
featurewise_center=False,  # set input mean to 0 over the dataset
samplewise_center=False,  # set each sample mean to 0
featurewise_std_normalization=False,  # divide inputs by std of the dataset
samplewise_std_normalization=False,  # divide each input by its std
zca_whitening=False,  # apply ZCA whitening
brightness_range=[0.8,1.2], # brightness control
rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
zoom_range = 0.05, # Randomly zoom image 
width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
horizontal_flip=False,  # randomly flip images
vertical_flip=False)  # randomly flip images

image_path1 = 'REAL FAIL/fake2_1.jpg'
image_path2 = 'REAL FAIL/fake2_2.jpg'
image_path3 = 'REAL FAIL/fake2_3.jpg'
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)
image3 = cv2.imread(image_path3)

sample = np.stack([image1, image2, image3])

image = random.choice(sample)
image = np.expand_dims(image, 0)

for i in range(1000):
    datagen.fit(sample)
    iteration = datagen.flow(image, batch_size=10)
    new_image = next(iteration).astype("uint8")[0]
    image_name = os.path.join('./HS1_fail', f'HS1_{i}.jpg')
    cv2.imwrite(image_name, new_image)
    print("A", i)


