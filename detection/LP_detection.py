import cv2
from utils import load_model, detect_lp, im2single


# Image path
img_path = "test/4.jpg"
save_path = "plates/4.jpg"

# Load WPOD model LP detection
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Read input image
img_org = cv2.imread(img_path)
cv2.imshow("Origin", img_org)

# Max and min of image dimension
d_max = 608
d_min = 288

# Calculate ratio between width and height + find min dimension
ratio = float(max(img_org.shape[:2])) / min(img_org.shape[:2])
side = int(ratio * d_min)
bound_dim = min(side, d_max)

_, lpImg, lp_type = detect_lp(wpod_net, im2single(img_org), bound_dim, lp_threshold=0.5)

if len(lpImg):
    # Show the first license detected (change to predict multiple licences)
    plate_img = cv2.cvtColor(lpImg[0], cv2.COLOR_RGB2BGR)
    cv2.imshow("Plate", plate_img)
    cv2.waitKey()
    cv2.imwrite(save_path, cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)*255)

cv2.destroyAllWindows()

