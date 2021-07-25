# import the necessary packages
import argparse
import time
import cv2
import os


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to super resolution model")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image we want to increase resolution of")
# args = vars(ap.parse_args())

# extract the model name and model scale from the file path
model_path = "models/EDSR_x4.pb"
modelName = model_path.split(os.path.sep)[-1].split("_")[0].lower().split('/')[1]
modelScale = model_path.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print("[INFO] loading super resolution model: {}".format(model_path))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel(modelName, 1)

# load the input image from disk and display its spatial dimensions
image_path = "plates/resized.jpg"
image = cv2.imread(image_path)
cv2.imshow("Original", image)
print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
# use the super resolution model to upscale the image, timing how
# long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] super resolution took {:.6f} seconds".format(end - start))
# show the spatial dimensions of the super resolution image
print("[INFO] w: {}, h: {}".format(upscaled.shape[1], upscaled.shape[0]))

# resize the image using standard bicubic interpolation
# start = time.time()
# bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC)
# end = time.time()
# print("[INFO] bicubic interpolation took {:.6f} seconds".format(end - start))

# show the original input image, bicubic interpolation image, and
# super resolution deep learning output
# cv2.imshow("Bicubic", bicubic)
cv2.imshow("Super Resolution", upscaled)
cv2.imwrite("res1.jpg", upscaled)
cv2.waitKey(0)
