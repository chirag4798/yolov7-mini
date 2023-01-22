# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This code is for research and experimentation purpose     #
# Forked from https://github.com/WongKinYiu/yolov7          #
# Author: Chirag Shetty                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import argparse
from yolov7_mini.yolov7 import YoloV7
from yolov7_mini.utils import plot_label, process_image, process_video

if __name__ == "__main__":
    # Argument parser for CLI
    parser = argparse.ArgumentParser(description="Detect and visualize output from YoloV7 model.")
    parser.add_argument("weights", type=str, help="path to yolov7 weights file.")
    parser.add_argument("source", type=str, help="path to image or video file.")
    parser.add_argument("output", type=str, help="path to output directory for saving visualizations.")
    parser.add_argument("--confidence-threshold", default=0.25, type=float, help="confidence threshold for object detection.")
    parser.add_argument("--iou-threshold", default=0.6, type=float, help="iou threshold for object detection.")
    args = parser.parse_args()
    
    # Model Settings
    model = YoloV7(
                args.weights, 
                confidence_threshold=args.confidence_threshold, 
                iou_threshold=args.iou_threshold
            )

    if (args.source.lower().endswith(".mp4")):
        process_video(args.source, args.output, model)
    else:
        process_image(args.source, args.output, model)
    