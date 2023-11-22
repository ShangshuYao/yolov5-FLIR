# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 16:09
# @Author  : YaoShangshu
# @File    : dataset_split.py
# @Software: PyCharm
import os
import random
import shutil
import cv2


class SampleDataset:
    """
        将数据集随机分为train_num数量的训练集以及val_num数量的验证集
        result_path = r'D:\StudyFile\Yao\small_dataset_test'        小型化数据集路径
        train_img_path = r'D:\dataset\FLIR_ADAS_1_3\images\train'  训练集图片路径
        val_img_path = r'D:\dataset\FLIR_ADAS_1_3\images\val'      验证集图片路径
    """

    def __init__(self, result_path, train_img_path, val_img_path, train_num=2100, val_num=900):
        self.result_img_train_path = result_path + r'\images\train'
        self.result_img_val_path = result_path + r'\images\val'
        self.result_lab_train_path = result_path + r'\labels\train'
        self.result_lab_val_path = result_path + r'\labels\val'
        self.train_num = train_num
        self.val_num = val_num
        self.train_img_path = train_img_path
        self.val_img_path = val_img_path

    def random_sample_imgdatset(self):
        train_img_list = os.listdir(self.train_img_path)  # 训练集图片文件名列表
        index = 0
        for _ in train_img_list:
            train_img_list[index] = os.path.join(self.train_img_path, _)  # 转换为图片文件路径列表
            index += 1
        val_img_list = os.listdir(self.val_img_path)  # 验证集图片文件名列表
        index = 0
        for _ in val_img_list:
            val_img_list[index] = os.path.join(self.val_img_path, _)  # 转换为图片文件路径列表
            index += 1

        train_img_list.extend(val_img_list)  # 将两个列表合并为一个

        train_img_list_sample = list()  # 小数据集训练集图片列表
        for _ in range(self.train_num):
            img = random.choice(train_img_list)
            train_img_list_sample.append(img)
            train_img_list.remove(img)

        val_img_list_sample = list()  # 小数据集验证集图片列表
        for _ in range(self.val_num):
            img = random.choice(train_img_list)
            val_img_list_sample.append(img)
            train_img_list.remove(img)

        tra_lab_sample_path = list()  # 小型化数据集 训练集标签文件路径
        val_lab_sample_path = list()  # 小型化数据集 验证集标签文件路径
        for _ in train_img_list_sample:
            img_f_pth = _.split('images')  # .../images  路径
            file_name = img_f_pth[1].split('.')[0]  # 文件名
            lab_name = file_name + '.txt'  # 标签文件名和后缀
            lab_pth = img_f_pth[0] + 'labels' + lab_name
            tra_lab_sample_path.append(lab_pth)  # 标签路径及文件名

        for _ in val_img_list_sample:
            img_f_pth = _.split('images')  # .../images/  路径
            file_name = img_f_pth[1].split('.')[0]  # 文件名  /val/xxx
            lab_name = file_name + '.txt'  # 标签文件名和后缀  /val/xxx.txt
            lab_pth = img_f_pth[0] + 'labels' + lab_name
            val_lab_sample_path.append(lab_pth)  # 标签路径及文件名

        self.save_dataset(self.result_img_train_path, train_img_list_sample)
        self.save_dataset(self.result_img_val_path, val_img_list_sample)
        self.save_dataset(self.result_lab_train_path, tra_lab_sample_path)
        self.save_dataset(self.result_lab_val_path, val_lab_sample_path)

    def save_dataset(self, path, file_list):
        file_num = 0
        if not os.path.isdir(path):
            os.makedirs(path)
            print('目录创建成功{}'.format(path))
        for file in file_list:
            shutil.copy(file, path)
            file_num += 1
        print('目录{}中共计{}个文件'.format(path, file_num))


class VideoSplit:
    """
        将视频分帧为图片
        source_path: 视频文件存储地址
        result_path： 图片结果文件保存地址
        frame: 帧率，每frame帧保存一张图片
    """

    def __init__(self, source_path, result_path, frame=10):
        self.source_path = source_path
        if not os.path.exists(self.source_path):
            raise Exception("源文件路径不存在！")

        self.result_path = result_path
        self.frame = frame

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print("创建文件夹{},".format(self.result_path))

    def split_video(self):
        # 获取视频文件名列表
        video_list = os.listdir(self.source_path)
        for i, name in enumerate(video_list):
            # 获取每个视频文件的路径
            video_list[i] = os.path.join(self.source_path, name)

            # 视频文件的名称
            basename = name.split('.')[0]

            # 以视频文件名称创建子文件夹，分别保存每个视频的图片文件
            video_result_path = os.path.join(self.result_path, basename)
            if not os.path.exists(video_result_path):
                os.makedirs(video_result_path)
                print("创建子文件夹{},".format(basename))

            # 利用VideoCapture捕获视频
            cap = cv2.VideoCapture(video_list[i])
            print("视频{}开始分帧...".format(name))

            # sum用于计算多少帧保存一次图片
            sum = 0
            i = 0  # i表示为图片数量
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                sum += 1
                # 保存图片
                if sum == self.frame:
                    sum = 0
                    i += 1
                    imgname = basename + '_' + str(i) + '.jpg'
                    imgPath = os.path.join(video_result_path, imgname)
                    cv2.imwrite(imgPath, frame)
                    print(imgname)
            print("{}视频文件提取完成".format(basename))

        print("完成")


if __name__ == "__main__":
    source_path = r'C:\Users\Htu\Desktop\avi'
    result_path = r'C:\Users\Htu\Desktop\image'
    tst = VideoSplit(source_path, result_path)
    tst.split_video()
