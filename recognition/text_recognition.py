from easyocr import Reader
import cv2

char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'


def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def recognition_process(image):
	# break the input languages into a comma separated list
	# langs = args["langs"].split(",")
	print("[INFO] OCR'ing with the following languages: {}".format("en"))
	reader = Reader(['en'])
	results = reader.readtext(image)

	# loop over the results
	final_text = ""
	for (bbox, text, prob) in results:
		# display the OCR'd text and associated probability
		# print("[INFO] {:.4f}: {}".format(prob, text))
		# # unpack the bounding box
		# (tl, tr, br, bl) = bbox
		# tl = (int(tl[0]), int(tl[1]))
		# tr = (int(tr[0]), int(tr[1]))
		# br = (int(br[0]), int(br[1]))
		# bl = (int(bl[0]), int(bl[1]))
		# cleanup the text and draw the box surrounding the text along
		# with the OCR'd text itself
		text = cleanup_text(text)
		# cv2.rectangle(image, tl, br, (0, 255, 0), 2)
		# cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		# show the output image
		# cv2.imshow("Image", image)
		# cv2.waitKey(0)
		text = fine_tune(text)
		final_text += text
	return final_text


if __name__ == "__main__":
	img_path = "plates\\0.jpg"
	image = cv2.imread(img_path)
	plate_num = recognition_process(image)
	print("Plate number: {}".format(plate_num))

