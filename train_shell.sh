#!/usr/bin/env bash
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM.yaml'
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM-1.yaml'
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM-2.yaml'
#python ./models/yolo.py --cfg 'yolo-C3ResCBAM-2-1.yaml'
#python ./models/yolo.py --cfg 'yolo-CBAM-4.yaml'
#python ./models/yolo.py --cfg 'yolov5s-BiFPN-1.yaml'
#python ./models/yolo.py --cfg 'yolo-CBAM-4.yaml'


python ./train.py --cfg 'yolo-CBAM-4.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'

python ./train.py --cfg 'yolo-C3ResCBAM.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
python ./train.py --cfg 'yolo-C3ResCBAM-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
python ./train.py --cfg 'yolo-C3ResCBAM-2.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
python ./train.py --cfg 'yolo-C3ResCBAM-2-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'

python ./train.py --cfg 'yolov5s-BiFPN.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
python ./train.py --cfg 'yolov5s-BiFPN-1.yaml' --epochs '100' --optimizer 'Adam' --hyp './data/hyps/hyp.scratch-low-adam.yaml'
