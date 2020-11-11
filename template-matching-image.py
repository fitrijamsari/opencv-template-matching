import cv2
import numpy as np

# load image file in color
img = cv2.imread("images/simpsons.jpg")

# load image file in greyscale
# gray_img = cv2.imread("simpsons.jpg", 0)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# load template file and convert to grayscale
template = cv2.imread("images/barts_face.jpg", cv2.IMREAD_GRAYSCALE)

# get width and height of template image for bouding box and invert the array
w, h = template.shape[::-1]

# run opencv template matching method on both image and template
res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
# print(res)

# set threshold and find location
thresh = 0.9
loc = np.where(res >= thresh)
# print(loc)

# unpack loc and point on the high thresh
for pt in zip(*loc[::-1]):
    # print(pt)
    # draw bounding box with size of template image on the original image
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

# show image
cv2.imshow("image", img)
# cv2.imshow("result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# draw bounding box
