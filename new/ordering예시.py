# 폴더 정리하기
import os

path_1 = "./Image Data_캐논코리아_231019" # 데이터가 들어있는 가장 상위 폴더
folder_1 = os.listdir(path_1) # 대오더 라벨의 정보(프린트 기종)

path_2 = os.path.join(path_1, folder_1[0]) 
folder_2 = os.listdir(path_2)

path_3 = os.path.join(path_2, folder_2[0])
folder_3 = os.listdir(path_3) # 날짜 정보

path_4 = os.path.join(path_3, folder_3[0])
folder_4 = os.listdir(path_4) # NG or OK 정보

path_5 = os.path.join(path_4, folder_4[0])
folder_5 = os.listdir(path_5) # 이미지 정보

path_6 = os.path.join(path_5, folder_5[0]) # 실제 이미지 path


