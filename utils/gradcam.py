# # -*- coding: utf-8 -*-
# # @Time    : 2023/3/3 21:28
# # @Author  : YaoShangshu
# # @File    : gradcam.py
# # @Software: PyCharm
import warnings

warnings.simplefilter('ignore')

import torch
import cv2
import os
import shutil
import numpy as np

np.random.seed(0)
from tqdm import trange
from PIL import Image
from models.yolo import Model, Detect
from utils.general import intersect_dicts
from utils.augmentations import letterbox
from utils.general import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


class HeatmapDetect(Detect):
    def __init__(self):
        super().__init__()
        self.inplace = False


class yolov5_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        module = model.model.modules()
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.fuse().eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[..., 4:]
        boxes_ = result[..., :4]
        sorted, indices = torch.sort(logits_[..., 0], descending=True)
        return logits_[0][indices[0]], xywh2xyxy(boxes_[0][indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # init ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # postprocess to yolo output
        post_result, post_boxes = self.post_process(result[0])
        # torch.autograd.set_detect_anomaly(True)
        for i in trange(int(post_result.size(0) * self.ratio)):
            if post_result[i][0] < self.conf_threshold:
                break

            self.model.zero_grad()
            if self.backward_type == 'conf':
                post_result[i, 0].backward(retain_graph=True)
            else:
                # get max probability for this prediction
                score = post_result[i, 1:].max()
                score.backward(retain_graph=True)

            # process heatmap
            gradients = grads.gradients[0]
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # add heatmap and box to image
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            # cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, 1:].argmax())],
            #                                  f'{self.model_names[int(post_result[i, 1:].argmax())]} {post_result[i][0]:.2f}',
            #                                  cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')


def get_params():
    params = {
        # YOLO-TIR
        'weight': 'D:\\StudyFile\\Yao\\yolov5-FLIR\\runs\\train\\exp125\\weights\\best.pt',
        # YOLOV5s
        # 'weight': 'D:\\StudyFile\\Yao\\yolov5-FLIR\\runs\\train\\exp99\\weights\\best.pt',
        # YOLOV5m
        # 'weight': 'D:\\StudyFile\\Yao\\yolov5-FLIR\\runs\\train\\exp107\\weights\\best.pt',
        # YOLO-FIRI
        # 'weight': 'D:\\StudyFile\\Yao\\yolov5-FLIR\\runs\\train\\exp119\\weights\\best.pt',
        # YOLOV3
        # 'weight': 'D:\\StudyFile\\Yao\\yolov5-FLIR\\runs\\train\\exp103\\weights\\best.pt',


        # YOLO-TIR
        'cfg': r'D:\StudyFile\Yao\yolov5-FLIR\models\diy_model\C3GhostforsmallSPPFCSPC-Bifpn-upsample.yaml',
        # YOLOV5s
        # 'cfg': r'D:\StudyFile\Yao\yolov5-FLIR\models\diy_model\yolov5s-baseline.yaml',
        # YOLOV5m
        # 'cfg': r'D:\StudyFile\Yao\yolov5-FLIR\models\diy_model\yolov5m-baseline.yaml',
        # YOLO-FIRI
        # 'cfg': r'D:\StudyFile\Yao\yolov5-FLIR\models\diy_model\yolov5-FIRI.yaml',
        # YOLOV3
        # 'cfg': r'D:\StudyFile\Yao\yolov5-FLIR\models\hub\yolov3.yaml',
        'device': 'cuda:0',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[-2]',
        'backward_type': 'conf',  # class or conf
        'conf_threshold': 0.6,  # 0.6
        'ratio': 0.02  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    model = yolov5_heatmap(**get_params())

    model(r'C:\Users\Htu\Desktop\video-57kWWRyeqqHs3Byei-frame-003704-hnZEGkFcsDGbQ4ZK2.jpg', r'C:\Users\Htu\Desktop\result\YOLO-TIR\video-57kWWRyeqqHs3Byei-frame-003704-hnZEGkFcsDGbQ4ZK2')
    # model(r'C:\Users\Htu\Desktop\video-zp8ed5vPKfAJ2fKWh-frame-006344-Nz9o8Kg86cij7hLtA.jpg', r'C:\Users\Htu\Desktop\result\YOLOV3\video-zp8ed5vPKfAJ2fKWh-frame-006344-Nz9o8Kg86cij7hLtA')
    # model(r'C:\Users\Htu\Desktop\video-57kWWRyeqqHs3Byei-frame-008515-3QTbXjfmsfueXdv7R.jpg', r'C:\Users\Htu\Desktop\result\YOLOV3\video-57kWWRyeqqHs3Byei-frame-008515-3QTbXjfmsfueXdv7R')
    # model(r'C:\Users\Htu\Desktop\AC062801.jpg', r'C:\Users\Htu\Desktop\result\YOLOV3\AC062801')
    # model(r'C:\Users\Htu\Desktop\AA062801.jpg', r'C:\Users\Htu\Desktop\result\YOLOV3\AA062801')


