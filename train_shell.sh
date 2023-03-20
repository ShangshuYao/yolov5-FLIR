#!/usr/bin/env bash
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
python ./train.py --cfg 'C3InvertBottleDW-Bifpn-ConvSppf.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
