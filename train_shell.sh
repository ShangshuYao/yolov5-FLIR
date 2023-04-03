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
python ./models/yolo.py --cfg 'C3RepTest-Bifpn-upsample.yaml'


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

#python ./train.py --cfg 'C3InvertBottleDW-Bifpn.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
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




