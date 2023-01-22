import os
import cv2
import time
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
from .conv import Conv
from .trace import TracedModel
from .ensemble import Ensemble


class YoloV7:
    """
    YOLOV7 model implemeneted to be integrated with the Sunergy Framework.
    """
    def __init__(
        self, 
        weigths_path, 
        img_size = (640,640),
        confidence_threshold = 0.25, 
        iou_threshold = 0.45,
        device = "0",
        visualize = True,
        half = True,
        trace = True
    ):
        print(" 🚀 Loading YOLOV7...")
        self.img_size             = img_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.visualize            = visualize
        self.half                 = half 
        self.trace                = trace

        # Set environment variable fro CUDA
        cpu = device.lower() == 'cpu'
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

        # Select device        
        cuda        = not cpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        
        # Load Model
        self.model = self.load_model(weigths_path, map_location=self.device)

        # Load Traced Model
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            print("Loading model in half precision!")
            self.model.half()

        # Get names and colors
        self.names  = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.names  = [name if name.lower() != "bagpack" else "Backpack" for name in self.names]
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.stride = int(self.model.stride.max())
        print(" 💥 Loaded!")


    def load_model(self, weigths_path, map_location):
        """
        Load model using weights file/
        """
        model = Ensemble()
        ckpt = torch.load(weigths_path, map_location=map_location)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        
        if len(model) == 1:
            return model[-1]  # return model
        else:
            print('Ensemble created with %s\n' % weigths_path)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model  # return ensemble


    def letterbox(self, img, color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
        """
        Pad and Resize image, keep aspect ratio.
        """
        # Resize and pad image while meeting self.stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size)

        # Scale ratio (new / old)
        r = min(self.img_size[0] / shape[0], self.img_size[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.img_size[1] - new_unpad[0], self.img_size[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.img_size[1], self.img_size[0])
            ratio = self.img_size[1] / shape[1], self.img_size[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def pre_process(self, img0):
        """
        Pre-process image for YOLOV7 inference.
        """
        # Pad image
        img = self.letterbox(img0)[0]
        # BGR to RGB Conversion
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        # Load array to device
        img = torch.from_numpy(img).to(self.device)
        # Convert to Half precision if required
        img = img.half() if self.half else img.float()
        # Normalize
        img /= 255.0
        # Unsqueeze
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img


    def box_area(self, box):
        """
        Calculate area of Bbox usinf W * H.
        """
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])


    def box_iou(self, box1, box2):
        """
        https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        area1 = self.box_area(box1.T)
        area2 = self.box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


    def xywh2xyxy(self, x):
        """
        Convert x, y, w, h coordinates to x1, y1, x2, y2 coordinates.
        """
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


    def non_max_suppression(self, prediction, classes=None, agnostic=False, multi_label=False, labels=()):
        """
        Performs Non Maximum Suppression using Torchvision.
        """
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > self.confidence_threshold  # candidates

        # Settings
        min_wh, max_wh = 2, 1920  # (pixels) minimum and maximum box width and height
        max_det = 100  # maximum number of detections per image
        max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 3.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            if nc == 1:
                x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                    # so there is no need to multiplicate.
            else:
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > self.confidence_threshold).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.confidence_threshold]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iou_threshold)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > self.iou_threshold  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output


    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        """
        Plot Bbox on the original image.
        """
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    def clip_coords(self, boxes, img_shape):
        """
        Clip coordinates in range of image width and height.
        """
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2


    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords


    def post_process(self, img0, img, pred):
        """
        Process boxes and create a Bounding Box List. Visualize if enabled.
        """
        boxes = []
        detection_index = 0
        for i, det in enumerate(pred):
            if not len(det):
                continue
            det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                detection_index += 1
                boxes.append({
                    "index": detection_index, 
                    "classes": int(cls), 
                    "left": int(xyxy[0]), 
                    "top": int(xyxy[1]), 
                    "right": int(xyxy[2]), 
                    "bottom": int(xyxy[3]), 
                    "confident": float(conf)
                })
                
                # BBox visualization
                label = f'{self.names[int(cls)]} {conf:.2f}'
                if (self.visualize):
                    self.plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
        return boxes


    def __call__(self, img0):
        """
        Forward pass with pre and post processing of YOLOV7
        """
        # Predictions on pre-processed image
        img = self.pre_process(img0)
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # Apply NMS to filter out BBox
        pred = self.non_max_suppression(pred)

        # Visualize BBox on original image
        boxes = self.post_process(img0, img, pred)
        return img0, boxes