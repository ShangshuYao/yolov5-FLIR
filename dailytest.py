# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 21:30
# @Author  : YaoShangshu
# @File    : dailytest.py
# @Software: PyCharm
import argparse
import os
import shutil
import random

import cv2
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
import torch
from utils.dataloaders import LoadImages
from pytorch_grad_cam import EigenCAM
from utils.general import check_yaml, non_max_suppression, intersect_dicts
# from utils.gradcam import GradCAM, show_cam_on_image, center_crop_img
from models.yolo import DetectionModel
from utils.torch_utils import select_device
import matplotlib





target_layers = []
#
#
# def imag_cam(model, img_path, target_category, use_cuda):
#     # for name, layer in model.model.named_children():
#     #     if name == '23':
#     #         target_layers.append(layer)
#     target_layer = [model.features[-1]]
#     data_transform = transforms.Compose([transforms.Resize([640, 640]),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
#     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     img = Image.open(img_path).convert('RGB')
#     # img = np.array(img, dtype=np.uint8)
#     # img = center_crop_img(img, 224)
#
#     # [C, H, W]
#     img_tensor = data_transform(img)
#     # expand batch dimension
#     # [C, H, W] -> [N, C, H, W]
#     input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)
#     #
#     cam = GradCAM(model=model, target_layers=target_layer, use_cuda=use_cuda)
#     # target_category = 895  # tabby, tabby cat
#     # target_category = 254  # pug, pug-dog
#
#     grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
#
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
#                                       grayscale_cam,
#                                       use_rgb=True)
#     plt.imshow(visualization)
#     plt.show()


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
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    device = select_device(opt.device)
    weight = torch.load(opt.weights)
    model = DetectionModel(check_yaml(opt.cfg), ch=3, nc=3, anchors=None).to(device)
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) else []  # exclude key
    csd = weight['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load

    # cls_Model = models.mobilenet_v3_large(pretrained=True)
    for name, layer in model.model.named_children():
        if name == '23':
            target_layers.append(layer)
    # load image
    img_path = r"C:\Users\Htu\Desktop\1.jpg"
    # imag_cam(cls_Model, img_path, target_category=None, use_cuda=True)

    # # load image
    # img_path = r"C:\Users\Htu\Desktop\1.jpg"
    # img = Image.open(img_path)    # PIL
    # resize = transforms.Resize([640, 640])
    # img = resize(img)
    # rgb_img = img.copy()
    # data_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #
    # img_tensor = data_transform(img)
    # input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)
    dataset = LoadImages(img_path, img_size=opt.imgsz)
    model.fuse().eval()
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        # im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = non_max_suppression(pred,  max_det=1000)
        print(pred)
    # cam = EigenCAM(model, target_layers, use_cuda=False)
    # grayscale_cam = cam(input_tensor)[0, :, :]

    # if isinstance(out, tuple):
    #     out = out[0]
    # detections = out.pandas().xyxy[0]
    #
    #
    # out = non_max_suppression(out, max_det=1000)

    # imag_cam(opt.weights, opt.cfg, opt.device, img_path, target_category=4)
    # main()
