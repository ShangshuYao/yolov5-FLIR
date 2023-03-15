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
from models.common import CBAM, ResCBAM, C3ResCBAM
from utils.general import check_yaml, intersect_dicts
from models.yolo import DetectionModel
from utils.torch_utils import select_device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:\\StudyFile\\Yao\\yolov5-FLIR\\yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--hyp', type=str, default='D:\\StudyFile\\Yao\\yolov5-FLIR\\data\\hyps\\hyp.scratch-low.yaml', help='hyperparameters path')
    opt = parser.parse_args()

    # load model
    # Hyperparameters
    # if isinstance(opt.hyp, str):
    #     with open(opt.hyp, errors='ignore') as f:
    #         hyp = yaml.safe_load(f)  # load hyps dict
    device = select_device(opt.device)
    # weight = torch.load(opt.weights)
    # model = DetectionModel(check_yaml(opt.cfg), ch=3, nc=3, anchors=None).to(device)
    # exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) else []  # exclude key
    # csd = weight['model'].float().state_dict()  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load
    model = C3ResCBAM(64, 64, 3, True).to(device)
    print(model)
    im = torch.ones(1, 64, opt.imgsz, opt.imgsz).to(device)
    res = model(im)
    print(res, res.shape)



