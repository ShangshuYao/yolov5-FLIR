# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 21:30
# @Author  : YaoShangshu
# @File    : dailytest.py
# @Software: PyCharm
# @comments: This file is just for test

import argparse
import torch.nn as nn
import torch.nn.functional as F
import yaml

import torch
from models.common import *
from utils.general import check_yaml, intersect_dicts
from models.yolo import DetectionModel
from utils.torch_utils import select_device
from torchsummary import summary
from utils.autoanchor import kmean_anchors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:\\StudyFile\\Yao\\yolov5-FLIR\\runs\\train\\exp71\\weights\\best.pt',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='C2fCCnet-Bifpn-upsample.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--hyp', type=str,
                        default='D:\\StudyFile\\Yao\\yolov5-FLIR\\data\\hyps\\hyp.scratch-low-adam.yaml',
                        help='hyperparameters path')
    opt = parser.parse_args()
    # hyp = opt.hyp
    # # load model
    device = select_device(opt.device)
    # model = DetectionModel(check_yaml(opt.cfg), ch=3, nc=3, anchors=None).to(device)
    # weight = torch.load(opt.weights)
    # # print(weight)
    #
    # exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) else []  # exclude key
    # csd = weight['model'].float().state_dict()  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load
    # tst = dict()
    #
    # with open('weight.txt', 'a') as f:
    #     for name in model.state_dict():
    #         tst[name] = model.state_dict()[name]
    #         f.write(str(name) + ':' + str(tst[name]) + '\n')
    im = torch.rand(1, 8, 32, 32).to(device)
    model = ShuffleBottle(8, 16, 3).to(device)
    out = model(im)
    print(out.shape)




