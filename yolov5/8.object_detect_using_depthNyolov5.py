#!/usr/bin/env python
import os, sys
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

@torch.no_grad()
def run():

    weights='yolov5/yolov5s.pt'  # model.pt path(s)
    imgsz=640  # inference size (pixels)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=10  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    stride = 32
    device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    update=False  # update all models
    name='exp'  # save results to project/name

    # Initialize
    set_logging()
    device = select_device(device_num)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location = device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location = device)['model']).to(device).eval()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    while(True):
        t0 = time.time()

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_threshold = depth_image.copy()
        depth_threshold[depth_threshold > 300] = 0                     # tresholding
        depth_threshold[depth_threshold < 200] = 0
        depth_threshold[depth_threshold != 0] = 1                      # binary
        depth_image_8bit = depth_threshold.astype(np.uint8)
        depth_image_8bit = cv.erode(depth_image_8bit, (5,5))
        depth_image_8bit = cv.dilate(depth_image_8bit, (5,5))
        depth_image_8bit = cv.blur(depth_image_8bit, (3,3))

        for i in range(color_image.shape[2]):
            color_image[:,:,i] = color_image[:,:,i]*depth_image_8bit


        # check for common shapes
        s = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in color_image], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        color_image_copy = color_image.copy()
        color_image = color_image[np.newaxis, :, :, :]        

        # Stack
        color_image = np.stack(color_image, 0)

        # Convert
        color_image = color_image[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        color_image = np.ascontiguousarray(color_image)

        color_image = torch.from_numpy(color_image).to(device)
        color_image = color_image.half() if half else color_image.float()  # uint8 to fp16/32
        color_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if color_image.ndimension() == 3:
            color_image = color_image.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(color_image, augment=augment,
                     visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, color_image, color_image_copy)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % color_image.shape[2:]  # print string
            annotator = Annotator(color_image_copy, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from color_image to im0 size
                det[:, :4] = scale_coords(color_image.shape[2:], det[:, :4], color_image_copy.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        cv.imshow("st_arm_eyes", color_image_copy)
        # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        # cv.imshow("DEPTH", depth_image_8bit)

        keyCode = cv.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break

if __name__ == '__main__':
    run()