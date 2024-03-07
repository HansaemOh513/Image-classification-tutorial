# 폴더 정리하기
import os
import shutil
import glob
import cv2
import sys
import re

destination_path = "./HS1/HS1_pass"
def move(k):
    path_1 = "./images" # 데이터가 들어있는 가장 상위 폴더
    folder_1 = os.listdir(path_1) # 대오더 라벨의 정보(프린트 기종)
    for label in folder_1:
        path_2 = os.path.join(path_1, label) 
        folder_2 = os.listdir(path_2) # 사진 넘버
        path_3 = os.path.join(path_2, folder_2[k-1], 'pass')
        folder_3 = os.listdir(path_3)
        for name in folder_3:
            path_4 = os.path.join(path_3, name)
            shutil.copy(path_4, os.path.join(destination_path, name))

move(2)
