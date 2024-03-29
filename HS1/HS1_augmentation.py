import os
import sys
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.random.set_seed(42)
# np.random.seed(42)
# random.seed(42)

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

image_path1 = 'REAL FAIL/fake2_1.jpg' # fake2_1 이미지 파일
image_path2 = 'REAL FAIL/fake2_2.jpg' # fake2_2 이미지 파일
image_path3 = 'REAL FAIL/fake2_3.jpg' # fake2_3 이미지 파일
image1 = cv2.imread(image_path1) # fake2_1 이미지 불러오기
image2 = cv2.imread(image_path2) # fake2_2 이미지 불러오기
image3 = cv2.imread(image_path3) # fake2_3 이미지 불러오기

sample = np.stack([image1, image2, image3]) # fake2_1, fake2_2, fake2_3, 합치기

n = np.random.randint(0, 3)
image = random.choice(sample) # 이미지 3개 중에서 
image = np.expand_dims(image, 0)

for i in range(1000):
    # datagen.fit(sample)
    n = np.random.randint(0, 3)
    image = np.expand_dims(sample[n], 0) # expand dimension of this array to bind many images generated
    iteration = datagen.flow(image, batch_size=10)
    new_image = next(iteration).astype("uint8")[0]
    image_name = os.path.join('./HS1_fail', f'HS1_{i}.jpg')

    check = new_image[200:600, 400:900, :]
    # plt.imshow(check)
    # plt.show()
    # print("B")
    # sys.exit()
    cv2.imwrite(image_name, new_image)
    print("A", i)

# print(len(os.listdir('./HS1/HS1_pass'))) # 5816
