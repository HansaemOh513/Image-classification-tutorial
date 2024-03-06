# 폴더 정리하기
import os
import shutil
import glob
import cv2
import sys
import re

destination_path = "./images"
def move():
    path_1 = "./Image Data_캐논코리아_231019" # 데이터가 들어있는 가장 상위 폴더
    folder_1 = os.listdir(path_1) # 대오더 라벨의 정보(프린트 기종)
    for label in folder_1:
        path_2 = os.path.join(path_1, label) 
        folder_2 = os.listdir(path_2) # image

        path_3 = os.path.join(path_2, folder_2[0])
        folder_3 = os.listdir(path_3) # 날짜 정보
        for date in folder_3:
            path_4 = os.path.join(path_3, date)
            folder_4 = os.listdir(path_4) # NG or OK 정보
            for test in folder_4: # fail or pass
                test = 'pass'
                path_5 = os.path.join(path_4, test)
                folder_5 = os.listdir(path_5) # 이미지 정보
                for name in folder_5 :
                    path_6 = os.path.join(path_5, name) # 실제 이미지 path
                    if re.search(pattern, name):
                        os.makedirs(os.path.join(destination_path, label, pattern_, test), exist_ok=True)
                        shutil.copy(path_6, os.path.join(destination_path, label, pattern_, test, name))

pattern = r'\(1\)'
pattern_ = '1'
move()

pattern = r'\(2\)'
pattern_ = '2'
move()

pattern = r'\(3\)'
pattern_ = '3'
move()

pattern = r'\(4\)'
pattern_ = '4'
move()

pattern = r'\(5\)'
pattern_ = '5'
move()
