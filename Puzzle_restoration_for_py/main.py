# python3
import cv2
from PIL import Image
import numpy as np
import os
import shutil
import threading

# 读取目标图片
source = cv2.imread(r"res/016.jpg")

# 拼接结果
target = Image.fromarray(np.zeros(source.shape, np.uint8))
# 图库目录
dirs_path = r"res/dirs_path"
# 差异图片存放目录
dst_path = r"res/dst_path"


def match(temp_file):
    # 读取模板图片
    template = cv2.imread(temp_file)
    # 获得模板图片的高宽尺寸
    theight, twidth = template.shape[:2]
    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(source, template, cv2.TM_SQDIFF_NORMED)
    # 归一化处理
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    target.paste(Image.fromarray(template), min_loc)
    return abs(min_val)


class MThread(threading.Thread):
    def __init__(self, file_name):
        threading.Thread.__init__(self)
        self.file_name = file_name

    def run(self):
        real_path = os.path.join(dirs_path, k)
        rect = match(real_path)
        if rect > 1e-10:
            print(rect)
            shutil.copy(real_path, dst_path)


count = 0
dirs = os.listdir(dirs_path)
threads = []
for k in dirs:
    if k.endswith('jpg'):
        count += 1
        print("processing on pic" + str(count))
        mt = MThread(k)
        mt.start()
        threads.append(mt)
    else:
        continue
# 等待所有线程完成
for t in threads:
    t.join()
target.show()
target.save(r"res/target/target.jpg")

# python3

# from cv2 import cv2
# from PIL import Image
# import os
# import shutil
# # 读取目标图片
# target = cv2.imread(r"res/1.png")
#
# def match(temp_file):
#     # 读取模板图片
#     template = cv2.imread(temp_file)
#     #获得模板图片的高宽尺寸
#     theight, twidth = template.shape[:2]
#     #执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
#     result = cv2.matchTemplate(target,template,cv2.TM_SQDIFF_NORMED )
#     #归一化处理
#     cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
#     #寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     return abs(min_val)
#
#
# dst_path = r"C:\Users\Administrator.WQ-20160501NYYU\Downloads\233"
# dirs = os.listdir(r"C:\Users\Administrator.WQ-20160501NYYU\Downloads\ddctf\file_d0wnl0ad")
# count = 0
# for k in dirs:
#     if k.endswith('png'):
#         count += 1
#         print("processing on pic"+str(count))
#         real_path = os.path.join(r"C:\Users\Administrator.WQ-20160501NYYU\Downloads\ddctf\file_d0wnl0ad", k)
#         rect = match(real_path)
#         if rect > 1e-10:
#             print(rect)
#             shutil.move(real_path, dst_path)
#     else:
#         continue



