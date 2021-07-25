from detection.LP_detection import detection_process
from recognition.text_recognition import recognition_process
from enhancement.super_res_image import enhancement_process
import argparse
import cv2
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be LP-detected")
ap.add_argument("-d", "--detect", required=True, help="path to detection model")
ap.add_argument("-e", "--enhance", required=True, help="path to enhance resolution model")
args = vars(ap.parse_args())

# read input image
image = cv2.imread(args["image"])
cv2.imshow("Input image", image)

# detect license plate
# Load WPOD model LP detection
# detect_model = load_model(args["detect"])
lpImage = detection_process(image, args["detect"])
# change type to uint8
lpImage = (lpImage*255).astype(np.uint8)
cv2.imshow("License plate image", lpImage)

# enhance resolution
# down-scale LP detected by 4
scale = 4
width = int(lpImage.shape[1] / scale)
height = int(lpImage.shape[0] / scale)
dim = (width, height)
# resize image
resized = cv2.resize(lpImage, dim, interpolation=cv2.INTER_AREA)
enhance_img = enhancement_process(resized, args["enhance"])
# cv2.imshow("Enhance LP image", enhance_img)

# text recognition
plate_number = recognition_process(enhance_img)
print("Plate number: {}".format(plate_number))
cv2.waitKey()
cv2.destroyAllWindows()


