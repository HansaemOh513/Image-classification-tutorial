'''Jupyter Notebook 사용하는 분들을 위한 tip:
    (1) '###...###' 표시 되어 있는 부분마다 끊어서 사용하면 편해요~
    (2) 함수는 따로 모아 놓는다? -> good '''

'''
[ References ]

< PCA >
(1) https://velog.io/@swan9405/PCA
(2) https://angeloyeo.github.io/2019/07/27/PCA.html 

< SVM >
(1) https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-2%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0-SVM
(2) https://angeloyeo.github.io/2020/09/30/SVM.html
'''

'''
< 결과 이미지 >  :
(1) pass_images_ex.png     (2) fail_images_ex.png     (3) PCA_result_2d.png     (4) SVM_with_PCA.png
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####


def load_images_to_numpy_list(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.jpeg'))]
    images_list = []
    for file in image_files[:100]:  # 첫 100개의 이미지 파일만을 처리합니다.
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        if image is not None:
            images_list.append(image)

    # 경로
    if 'fail' in folder_path:
        print("Shape of fail images", np.shape(images_list))
    else:
        print("Shape of pass images", np.shape(images_list))
    return images_list

HS1_pass_images = load_images_to_numpy_list('/home/cannon/project/jun/python/HS1/HS1_pass')
HS1_fail_images = load_images_to_numpy_list('/home/cannon/project/jun/python/HS1/HS1_fail')

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

def crop_and_display(images, y1, y2, x1, x2, n):
    crop_images = []

    # 이미지를 자르고 리스트에 추가
    for image in images:
        crop_image = image[y1:y2, x1:x2]
        crop_images.append(crop_image)
    
    num_images = min(len(crop_images), n)


    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(crop_images[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', rotation=0)
        plt.title(f'Image {i+1}')
    plt.show()

    print("Shape of crop images : ", np.shape(crop_images))
    return crop_images


crop_HS1_fail_images = crop_and_display(HS1_fail_images, 250, 550, 400, 900, 3)
crop_HS1_pass_images = crop_and_display(HS1_pass_images, 250, 550, 400, 900, 6)

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

# 이진분류 상황의 라벨링 함수 (다중분류시 수정 필요)
def make_label_of_images(pass_images, fail_images):
    num_pass = np.shape(pass_images)[0]
    num_fail = np.shape(fail_images)[0]

    labels = []
    for i in range(0, num_pass):
        labels.append([1,0])
    for i in range(0, num_fail):
        labels.append([0,1])
    print(np.shape(labels))

    return(labels)

labels_all = make_label_of_images(crop_HS1_pass_images, crop_HS1_fail_images)

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

def apply_pca_and_visualize(pass_images, fail_images):
    # 이미지 데이터셋을 벡터 형태로 변환
    all_images = np.array(pass_images + fail_images)
    num_images, h, w, c = all_images.shape
    # 'PCA'를 함수로 생각해봅시다. (입력과 출력이 있죠?)
        # PCA의 입력은 2차원 행렬이 아니라 1차원 벡터가 필요합니다.
    flattened_images = all_images.reshape(num_images, h*w*c)
    
    # PCA 적용, 2개의 주성분으로 데이터 축소
    # ( 2개의 주성분(축)으로 설정해야 2차원 그래프에 점으로 찍어볼 수 있겠죠? )
    # '3'으로 한다면? -> 3차원 플롯으로 그릴 수도 있을지도...
    pca = PCA(n_components=2) # <--- 'pca'란 함수를 만드는 부분
    transformed_images = pca.fit_transform(flattened_images) # <--- 위에서 만든 함수를 사용하는 부분
    
    # 라벨 생성: 'pass'는 0, 'fail'는 1
    # 학습 시에는 one-hot-encoded vector가 필요하지만 PCA 경우에는 불필요함!
    labels = np.array([0] * len(pass_images) + [1] * len(fail_images))
    
    # 시각화
    plt.figure(figsize=(10, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        mask = labels == label
        plt.scatter(transformed_images[mask, 0], transformed_images[mask, 1], label=('Pass' if label==0 else 'Fail'), c=color)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA of HS1 - Images')
    plt.show()

    # 2차원으로 변환한 2개의 축 변수값들을 리스트로 반환
    pc1 = transformed_images[:, 0].tolist()
    pc2 = transformed_images[:, 1].tolist()

    # 총 이미지 개수만큼 나오면 잘된거겠죠?
    print("Length of two principal components variavles : ", len(pc1))

    return pc1, pc2

# 이미지 데이터셋과 라벨링 적용
HS1_pc1, HS1_pc2 = apply_pca_and_visualize(crop_HS1_pass_images, crop_HS1_fail_images)

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####
'''
이제 새로 정의한 위의 변수들을 가지고 분류모델을 만들어 보겠습니다~
'''
#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

# (1) 정규화
    # : 입력변수에는 정규화 과정이 필요합니다.
        # ( 정규화 방법에는 여러가지 : 데이터의 특성에 맞는 방법 선택 필요 )

# HS1_pc1과 HS1_pc2를 numpy 배열로 변환
# ( 리스트는 두개였지만! 결국 모두 X변수로 사용하기 위해서... )
X = np.column_stack((HS1_pc1, HS1_pc2))

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

# (2) SVM (support vector machine)

# labels_all은 이미 one-hot encoded 상태이므로, SVM에 적합한 형태로 변환 필요
# (여기서 'labels_all'은 위에서 정의한 'make_label_of_images'함수를 통해 얻은 one-hot-encoding vector)
y = np.array(labels_all)[:, 0]
# [1, 0] 혹은 [0, 1] 형태의 라벨을 1 혹은 0으로 변환

# SVM 분류기 학습
clf = SVC(kernel='linear')
clf.fit(X_scaled, y)

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

# (3) 결정 경계 확인

# 시각화 함수 정의
def plot_decision_boundary(X, y, model):
    # 결정 경계 시각화를 위한 그리드 설정
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 모델을 사용하여 그리드 포인트에 대한 예측 수행
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 결정 경계 시각화
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

    # 결정 경계에 검정색 선 추가
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=3)

    # 클래스 별로 데이터 점 시각화
    scatter_markers = ['o', 's']  # 클래스 별 마커 형태
    colors = ['blue', 'red']  # 클래스 별 내부 색상
    edge_colors = ['yellow', 'black']  # 클래스 별 테두리 색상
    # 여기서 클래스란? : 'pass' vs 'fail'
    for idx, (marker, color, edge_color) in enumerate(zip(scatter_markers, colors, edge_colors)):
        plt.scatter(X[y == idx, 0], X[y == idx, 1], c=color, marker=marker, edgecolors=edge_color, label=f'Class {idx}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.show()

# 결정 경계 시각화
plot_decision_boundary(X_scaled, y, clf)

'''
단순 시각화 용도일 뿐만이 아니라,
차원축소를 위한 용도로 사용 가능하다~

ex:
차원축소 전 1개 이미지 -> (300, 500, 3)이므로 즉 300*500*3 = 450000차원
차원축소 후 2개의 변수 -> 2차원
'''

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####

'''
여기서는 train data, test data를 나누지 않았지만
그렇게 해보는 시도? good
'''

#####     #####     #####     #####     #####     #####     #####     #####     #####     #####






















'''YJS'''
