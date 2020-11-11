import cv2
import numpy as np

# load camera video
cap = cv2.VideoCapture(0)

# load template
template = cv2.imread("images/board.png", cv2.IMREAD_GRAYSCALE)

# get width and height of template image for bouding box and invert the array
w, h = template.shape[::-1]

# loop video frame
while True:
    # Capture frame-by-frame
    _, frame = cap.read()

    # convert to grayscale of the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run opencv template matching method on both image and template
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # set threshold and find location
    thresh = 0.8
    loc = np.where(res >= thresh)

    # unpack loc and point on the high thresh
    for pt in zip(*loc[::-1]):
        # print(pt)
        # draw bounding box with size of template image on the original image
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
