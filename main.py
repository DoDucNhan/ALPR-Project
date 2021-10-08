from detection.LP_detection import detection_process
from recognition.text_recognition import recognition_process
from enhancement.super_res_image import enhancement_process
import argparse
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be LP-detected")
ap.add_argument("-d", "--detect", default="wpod", help="model name: wpod or yolo")
args = vars(ap.parse_args())

# read input image
image = cv2.imread(args["image"])
cv2.imshow("Input image", image)

# detect license plate
# Load WPOD model LP detection
# detect_model = load_model(args["detect"])
lpImage = detection_process(image, args["detect"])
cv2.blur(lpImage, (5, 5), cv2.BORDER_DEFAULT)
cv2.imshow("License plate image", lpImage)

# text recognition
plate_number = recognition_process(lpImage)
print("Plate number: {}".format(plate_number))
cv2.waitKey()
cv2.destroyAllWindows()


