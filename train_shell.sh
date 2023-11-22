#!/usr/bin/env bash
# 测试网络模块可用性
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM.yaml'
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM-1.yaml'
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM-2.yaml'
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM-2-1.yaml'
#python ./models/yolo.py --cfg 'yolo-C3InvertBottleneck.yaml'
#python ./models/yolo.py --cfg 'yolo-C3InvertBottleneck-1.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-CBAM.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-Bifpn.yaml'
#python ./models/yolo.py --cfg 'yolov5m-baseline.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottleDW-Bifpn-1.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottleDW-Bifpn-ConvSppf.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-Bifpn-CBAM.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-Bifpn-ConvSppf.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-BifpnL.yaml'
#python ./models/yolo.py --cfg 'C3NoActInvertBottle-Bifpn.yaml'
#python ./models/yolo.py --cfg 'C3PreActInvertBottle-Bifpn.yaml'
#python ./models/yolo.py --cfg 'InvertBottleincep-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'InvertBottleincep-Bifpn-upsample-1.yaml'
#python ./models/yolo.py --cfg 'InvertBottleincep-Bifpn-upsample-2.yaml'
#python ./models/yolo.py --cfg 'CoT3InvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3NoActInvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3-Bifpn-Invertbottle-upsample.yaml'
#python ./models/yolo.py --cfg 'PconvInvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3RepTest-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3ELANBlock-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3RepTest-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3GhostInvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-BifpnCBAM-upsample.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-BifpnSA-upsample.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottleCat-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-3.yaml'
#python ./models/yolo.py --cfg 'ForSmallC3InvertBottle-Bifpn2-upsample-2.yaml'
#python ./models/yolo.py --cfg 'C2f-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample-2.yaml'
#python ./models/yolo.py --cfg 'ablation-IGC3.yaml'
#python ./models/yolo.py --cfg 'ablation-SFPN.yaml'
#python ./models/yolo.py --cfg 'ablation-SUP.yaml'
#python ./models/yolo.py --cfg 'ablation-SPPFCSPC.yaml'
#python ./models/yolo.py --cfg 'ablation-IGC3-SFPN.yaml'
#python ./models/yolo.py --cfg 'ablation-IGC3-SFPN-SUP.yaml'
#python ./models/yolo.py --cfg 'C3BottleNewSPPFCSPC-Bifpn-upsample.yaml'
#python ./models/yolo.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample-3.yaml'



# 开始训练
#python ./train.py --cfg 'yolo-CBAM-4.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#
#python ./train.py --cfg 'yolo-C3ResCBAM.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3ResCBAM-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3ResCBAM-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3ResCBAM-2-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#
#python ./train.py --cfg 'yolo-C3InvertBottleneck.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3InvertBottleneck-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3InvertBottleneck-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'

#python ./train.py --cfg 'yolov5s-baseline.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolov5s-BiFPN.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolov5s-BiFPN-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-CBAM-1-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#
#python ./train.py --cfg 'yolo-C3ConvNext.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3ConvNext-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolo-C3ConvNext-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#
#python ./train.py --cfg 'C3InvertBottle-CBAM.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./train.py --cfg 'yolov5m-baseline.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'

#python ./train.py --cfg 'C3InvertBottleDW-Bifpn-UpSample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottleDW-Bifpn-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottleDW-Bifpn-ConvSppf.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn-CBAM.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn-ConvSppf.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-BifpnL.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn-CBAM-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3NoActInvertBottle-Bifpn.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3PreActInvertBottle-Bifpn.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'InvertBottleincep-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'InvertBottleincep-Bifpn-upsample-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'InvertBottleincep-Bifpn-upsample-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'CoT3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3NoActInvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Cat-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3-Bifpn-Invertbottle-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'PconvInvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./dailytest.py --fl_gamma 0.3
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./dailytest.py --fl_gamma 0.6
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./dailytest.py --fl_gamma 1.0
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./dailytest.py --fl_gamma 1.3
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./dailytest.py --fl_gamma 1.6
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./dailytest.py --fl_gamma 2.0
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-flgamma.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3RepTest-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottleDW-Bifpn-UpSample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3ELANBlock-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3RepTest-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostInvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-BifpnCBAM-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-BifpnSA-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottleCat-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-3.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolov5s.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolov5s.yaml' --epochs '200' --optimizer 'SGD' --hyp './data/hyps/hyp.scratch-low.yaml'
#python ./train.py --cfg './models/hub/yolov3.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg './models/hub/yolov3.yaml' --epochs '200' --optimizer 'SGD' --hyp './data/hyps/hyp.scratch-low.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample-2.yaml' --epochs '200' --optimizer 'SGD' --hyp './data/hyps/hyp.scratch-low.yaml'
#python ./train.py --cfg 'yolov5m-baseline.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C2f-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C2fNAM-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ForSmallC3InvertBottle-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottle-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottleforsmall-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3forsmallSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3InvertBottleNAMforsmall-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'InvertNAMSPPFCSPCforsmall-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostShuffleforsmallSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'BottleneckSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolov5-FIRI.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample-2.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostCBAMSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample-2.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ablation-IGC3.yaml'
#python ./train.py --cfg 'ablation-SFPN.yaml'
#python ./train.py --cfg 'ablation-SUP.yaml'
#python ./train.py --cfg 'ablation-SPPFCSPC.yaml'
#python ./train.py --cfg 'ablation-IGC3-SFPN.yaml'
#python ./train.py --cfg 'ablation-IGC3-SFPN-SUP.yaml'
#python ./train.py --cfg 'ablation-IGC3-SFPN-SUP-SPPFCSPC.yaml'
#python ./train.py --cfg 'ablation-IGC3.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ablation-IGC3-SFPN.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'ablation-IGC3-SFPN-SUP.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'yolov3.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
#python ./train.py --cfg 'C3BottleNewSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'

# 迁移学习
#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp125/weights/best.pt' --data './data/HTU_TIR.yaml' --patience '100'
#python ./train.py --cfg 'yolov5s-baseline.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp99/weights/best.pt' --data './data/HTU_TIR.yaml' --patience '100'
#python ./train.py --cfg 'yolov3.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp103/weights/best.pt' --data './data/HTU_TIR.yaml' --patience '100'
#python ./train.py --cfg 'yolov5m-baseline.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp107/weights/best.pt' --data './data/HTU_TIR.yaml' --patience '100'
#python ./train.py --cfg 'yolov5-FIRI.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp119/weights/best.pt' --data './data/HTU_TIR.yaml' --patience '100'

#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample-3.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'

# 增量学习
#python ./train.py --cfg 'C3GhostforsmallSPPFCSPC-Bifpn-upsample.yaml' --epochs '200' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp125/weights/best.pt' --data './data/HTU_TIR+FLIR.yaml' --patience '50'
#
#python ./train.py --cfg 'yolov5s-baseline.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp99/weights/best.pt' --data './data/HTU_TIR+FLIR.yaml'
#python ./train.py --cfg 'yolov3.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp103/weights/best.pt' --data './data/HTU_TIR+FLIR.yaml'
#python ./train.py --cfg 'yolov5m-baseline.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp107/weights/best.pt' --data './data/HTU_TIR+FLIR.yaml'
#python ./train.py --cfg 'yolov5-FIRI.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml' --weights './runs/train/exp119/weights/best.pt' --data './data/HTU_TIR+FLIR.yaml'

# detect

## 5s-baseline
#python ./detect.py --weights './runs/train/exp99/weights/best.pt'
## 3s-baseline
#python ./detect.py --weights './runs/train/exp103/weights/best.pt'
## YOLOv5m-baseline.yaml
#python ./detect.py --weights './runs/train/exp107/weights/best.pt'
## YOLO-FIRI
#python ./detect.py --weights './runs/train/exp119/weights/best.pt'
## C3GhostforsmallSPPFCSPC-Bifpn-upsample
#python ./detect.py --weights './runs/train/exp125/weights/best.pt'

## C3GhostforsmallSPPFCSPC-Bifpn-upsample HTU迁移学习
#python ./detect.py --weights './runs/train/exp143/weights/best.pt' --source "D:/dataset/ir_vis/train_ir"
# 5s-baseline
#python ./detect.py --weights './runs/train/exp149/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"
## 3s-baseline
#python ./detect.py --weights './runs/train/exp150/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"
## YOLOv5m-baseline.yaml
#python ./detect.py --weights './runs/train/exp151/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"
## YOLO-FIRI
#python ./detect.py --weights './runs/train/exp152/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"


## 5s-baseline
#python ./detect.py --weights './runs/train/exp99/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"
## 3s-baseline
#python ./detect.py --weights './runs/train/exp103/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"
## YOLOv5m-baseline.yaml
#python ./detect.py --weights './runs/train/exp107/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"
## YOLO-FIRI
#python ./detect.py --weights './runs/train/exp119/weights/best.pt' --source "D:/dataset/ir_vis/avi/vid_img/xiuxiuimg"

## 5s-baseline
#python ./detect.py --weights './runs/train/exp99/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
## 3s-baseline
#python ./detect.py --weights './runs/train/exp103/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
## YOLOv5m-baseline.yaml
#python ./detect.py --weights './runs/train/exp107/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
## YOLO-FIRI
#python ./detect.py --weights './runs/train/exp119/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
#
#
#
## 5s-baseline
#python ./detect.py --weights './runs/train/exp149/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
## 3s-baseline
#python ./detect.py --weights './runs/train/exp150/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
## YOLOv5m-baseline.yaml
#python ./detect.py --weights './runs/train/exp151/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
## YOLO-FIRI
#python ./detect.py --weights './runs/train/exp152/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"



## 5s-baseline
#python ./detect.py --weights './runs/train/exp149/weights/best.pt' --source "D:/dataset/FLIR_ADAS_v2/images_thermal_val/data"
## 3s-baseline
#python ./detect.py --weights './runs/train/exp150/weights/best.pt' --source "D:/dataset/FLIR_ADAS_v2/images_thermal_val/data"
## YOLOv5m-baseline.yaml
#python ./detect.py --weights './runs/train/exp151/weights/best.pt' --source "D:/dataset/FLIR_ADAS_v2/images_thermal_val/data"
## YOLO-FIRI
#python ./detect.py --weights './runs/train/exp152/weights/best.pt' --source "D:/dataset/FLIR_ADAS_v2/images_thermal_val/data"
## C3GhostforsmallSPPFCSPC-Bifpn-upsample
#python ./detect.py --weights './runs/train/exp143/weights/best.pt' --source "D:/dataset/FLIR_ADAS_v2/images_thermal_val/data"

# 增量学习之前的结果
# 5s-baseline
python ./detect.py --weights './runs/train/exp99/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# 3s-baseline
python ./detect.py --weights './runs/train/exp103/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# YOLOv5m-baseline.yaml
python ./detect.py --weights './runs/train/exp107/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# YOLO-FIRI
python ./detect.py --weights './runs/train/exp119/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# C3GhostforsmallSPPFCSPC-Bifpn-upsample
python ./detect.py --weights './runs/train/exp125/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"

# 增量学习之后
# 5s-baseline
python ./detect.py --weights './runs/train/exp154/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# 3s-baseline
python ./detect.py --weights './runs/train/exp155/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# YOLOv5m-baseline.yaml
python ./detect.py --weights './runs/train/exp156/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# YOLO-FIRI
python ./detect.py --weights './runs/train/exp157/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"
# C3GhostforsmallSPPFCSPC-Bifpn-upsample
python ./detect.py --weights './runs/train/exp153/weights/best.pt' --source "C:/Users/Htu/Desktop/HTU_test_detect"



# val
# C3GhostforsmallSPPFCSPC-Bifpn-upsample     HTU迁移学习
#python ./val.py --weights './runs/train/exp140/weights/best.pt' --data './data/HTU_TIR.yaml'


