import cv2
import numpy as np
from detection.utils import load_model, detect_lp, im2single


def wpod_net(img):
    # load model
    model = load_model("detection\\wpod-net.json")
    # Max and min of image dimension
    d_max = 608
    d_min = 288

    # Calculate ratio between width and height + find min dimension
    ratio = float(max(img.shape[:2])) / min(img.shape[:2])
    side = int(ratio * d_min)
    bound_dim = min(side, d_max)

    _, lpImg, lp_type = detect_lp(model, im2single(img), bound_dim, lp_threshold=0.5)

    if len(lpImg):
        # Show the first license detected (change to predict multiple licences)
        # change type to uint8
        lpImage = cv2.cvtColor(lpImg[0], cv2.COLOR_RGB2BGR)
        lpImage = (lpImage * 255).astype(np.uint8)
        return lpImage


def yolo_v4(img):
    net = cv2.dnn_DetectionModel("detection\\yolo-tinyv4-obj.cfg", "detection\\yolo-tinyv4-obj_best.weights")
    net.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    _, confidence, box = net.detect(img, confThreshold=0.2, nmsThreshold=0.4)
    left, top, width, height = box[0]
    lpImg = img[top:top + height, left:left + width]
    return lpImg


def detection_process(img, model):
    if model == "wpod":
        return wpod_net(img)
    elif model == "yolo":
        return yolo_v4(img)
    else:
        return img



