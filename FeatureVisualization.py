# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 22:50
# @Author  : YaoShangshu
# @File    : FeatureVisualization.py
# @Software: PyCharm
import cv2
import numpy as np
import torch
from utils.augmentations import letterbox
from torch.autograd import Variable
from torchvision import models
from models.yolo import Model, Detect



class FeatureVisualization():
    def __init__(self,img_path,selected_layer,model):
        self.img_path=img_path
        self.selected_layer=selected_layer
        self.pretrained_model = model

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        return tensor

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        print(input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features=self.get_feature()
        print(features.shape)

        feature=features[:,0,:,:]
        print(feature.shape)

        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)

        return feature

    def save_feature_to_img(self):
        #to numpy
        feature=self.get_single_feature()
        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        print(feature[0])

        cv2.imwrite('./img.jpg',feature)


if __name__=='__main__':
    # get class
    myClass = FeatureVisualization('./input_images/home.jpg',5)
    print(myClass.pretrained_model)

    myClass.save_feature_to_img()

