import cv2
from detection.utils import load_model, detect_lp, im2single


def detection_process(img, model_path):
    # load model
    model = load_model(model_path)
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
        return cv2.cvtColor(lpImg[0], cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    # Image path
    img_path = "test/3.jpg"
    save_path = "plates/3.jpg"
    # Read input image
    img_org = cv2.imread(img_path)
    cv2.imshow("Origin", img_org)
    scale = 0.5
    w = int(img_org.shape[1] * scale)
    h = int(img_org.shape[0] * scale)
    dim = (w, h)
    # resize image
    resized = cv2.resize(img_org, dim, interpolation=cv2.INTER_AREA)

    # Load WPOD model LP detection
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)

    plate_img = detection_process(resized, wpod_net)
    cv2.imshow("Plate", plate_img)
    cv2.waitKey()
    # cv2.imwrite(save_path, cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)*255)

