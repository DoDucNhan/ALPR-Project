# import the necessary packages
import time
import cv2
import os


def enhancement_process(image, model_path):
    # extract the model name and model scale from the file path
    # model_path = "models\\EDSR_x4.pb"
    modelName = model_path.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = model_path.split("_x")[-1]
    modelScale = int(modelScale[:modelScale.find(".")])
    if modelName != "edsr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize OpenCV's super resolution DNN object, load the super
    # resolution model from disk, and set the model name and scale
    print("[INFO] loading super resolution model: {}".format(model_path))
    print("[INFO] model name: {}".format(modelName))
    print("[INFO] model scale: {}".format(modelScale))
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(modelName, modelScale)

    # load the input image from disk and display its spatial dimensions
    print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
    # use the super resolution model to upscale the image, timing how
    # long it takes
    start = time.time()
    enhance_img = sr.upsample(image)
    end = time.time()
    print("[INFO] super resolution took {:.6f} seconds".format(end - start))
    return enhance_img

if __name__ == "__main__":
    image_path = "plates\\0.jpg"
    lpImage = cv2.imread(image_path)
    # down-scale LP detected by 4
    scale = 4
    width = int(lpImage.shape[1] / scale)
    height = int(lpImage.shape[0] / scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(lpImage, dim, interpolation=cv2.INTER_AREA)
    enhance_img = enhancement_process(resized, "models\\EDSR_x4.pb")
    cv2.imshow("Enhance LP image", enhance_img)
    cv2.waitKey()
