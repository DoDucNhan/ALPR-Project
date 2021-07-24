import pytesseract
import cv2
from skimage.segmentation import clear_border
# from detection.utils import load_model, detect_lp, im2single

# Dinh nghia cac ky tu tren bien so
char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'
custom_config = r"-c tessedit_char_whitelist={} --psm 6".format(char_list)

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = "plates/1.jpg"

# Đọc file ảnh đầu vào
plate_img = cv2.imread(img_path)

# Convert to grayscale
gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

# Blur
# blur = cv2.medianBlur(gray, 5)

# Chuyen doi anh bien so
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# thresh = clear_border(thresh)

cv2.imshow("Anh bien so sau threshold", thresh)


# Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
text = pytesseract.image_to_string(thresh, lang="eng", config=custom_config)
print(text)

# Viet bien so len anh
# cv2.putText(Ivehicle,fine_tune(text),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
#
# # Hien thi anh va luu anh ra file output.png
# cv2.imshow("Anh input", Ivehicle)
# cv2.imwrite("output.png", Ivehicle)
cv2.waitKey()

cv2.destroyAllWindows()

