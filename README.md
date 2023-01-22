# YOLOV7-mini

Minified version of YoloV7 for inference.

## Instructions to run

Clone this repo inside the [Yolov7 Repository](https://github.com/WongKinYiu/yolov7) and copy its content to YoloV7 repo.

```
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
git clone https://github.com/chirag4798/yolov7-mini.git
cp -r yolov7-mini/ ./
```

Use the instructions from the help menu below
```
usage: main.py [-h]
               [--confidence-threshold CONFIDENCE_THRESHOLD]
               [--iou-threshold IOU_THRESHOLD]
               weights source output

Detect and visualize output from YoloV7 model.

positional arguments:
  weights               path to yolov7 weights file.
  source                path to image or video file.
  output                path to output directory for
                        saving visualizations.

optional arguments:
  -h, --help            show this help message and exit
  --confidence-threshold CONFIDENCE_THRESHOLD
                        confidence threshold for object
                        detection.
  --iou-threshold IOU_THRESHOLD
                        iou threshold for object
                        detection.
```

Example
```
python3 main.py weights/deepfashion_detector.py images/sample.jpg output
```