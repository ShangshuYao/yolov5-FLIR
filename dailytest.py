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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:\\StudyFile\\Yao\\yolov5-FLIR\\yolov5s.pt',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--hyp', type=str,
                        default='D:\\StudyFile\\Yao\\yolov5-FLIR\\data\\hyps\\hyp.scratch-low-flgamma.yaml',
                        help='hyperparameters path')
    parser.add_argument('--fl_gamma', type=float, default=1.0)
    opt = parser.parse_args()

    # load model
    # Hyperparameters
    # if isinstance(opt.hyp, str):
    #     with open(opt.hyp, errors='ignore') as f:
    #         hyp = yaml.load(f.read(), yaml.FullLoader)  # load hyps dict
    #         f.close()
    #
    #     hyp['fl_gamma'] = opt.fl_gamma
    device = select_device(opt.device)
    #     with open(opt.hyp, 'w', errors='ignore') as f:
    #         yaml.dump(hyp, f)
    #         f.close()
    #     print(hyp['fl_gamma'])

    # weight = torch.load(opt.weights)
    # model = DetectionModel(check_yaml(opt.cfg), ch=3, nc=3, anchors=None).to(device)
    # exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) else []  # exclude key
    # csd = weight['model'].float().state_dict()  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load
    # model = C3InvertBottleneckInception(64, 64, 3, True, 16).to(device)
    # for i in model.modules():
    #     print(i)
    im = torch.ones(1, 1, 4, 4)
    model1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    model1.weight.data = torch.Tensor([[[[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]]])

    model2 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    model2.weight.data = torch.Tensor([[[[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]]])

    model3 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    model3.weight.data = model1.weight.data + model2.weight.data

    res1 = model1(im)
    res2 = model2(im)
    res = res1 + res2

    out = model3(im)
    print(res, out)




