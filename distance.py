import imutils
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from imutils import contours
import time

#    sorting of points in top-left, top-right, bottom-right, bottom-left order
def order_points(pts):

    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

#    returns midpoint of 2 points
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#    input image
image = cv2.imread("test.jpg")

#    width of reference object
width = 5

ratio = image.shape[0] / 500

#    resize the image as it is too large
image = imutils.resize(image, height=500)

#    Gray Scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

#    Edge Detection and noise reduction
edge = cv2.Canny(blur, 50, 100)
edge = cv2.dilate(edge, None, iterations=1)
edge = cv2.erode(edge, None, iterations=1)

#    Displaying edge detected image
cv2.imshow("edge detection", edge)
cv2.waitKey(0)

#    Find underlying objects from the edge detecting image
cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#    Sorting the objects from left to right
(cnts, _) = contours.sort_contours(cnts)

#    colors ref to all 4 corners and mid point
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))

refObj = None

for c in cnts:

    #    continue only if the object is sufficiently large
    if cv2.contourArea(c) < 500:
        continue

    #    extracting the corner points of the object and converting to numpy array
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    #    correcting the order of corner points
    box = order_points(box)

    #    calculating center
    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])

    #    assigning the reference object
    if refObj is None:
        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        refObj = (box, (cX, cY), D/width)

        continue

    orig = image.copy()

    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

    #    coordinates of reference object and the object in question
    refCoords = np.vstack([refObj[0], refObj[1]])
    objCoords = np.vstack([box, (cX, cY)])
    
    #    looping over all detected object from the image for calculating distance
    for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):

        #    start time
        st = time.time()

        cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
        cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
        cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)

        D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
        (mX, mY) = midpoint((xA, yA), (xB, yB))

        print(c)

        cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        #    end time
        print("--- %s seconds ---" % (time.time() - st))
	
        cv2.imshow("contours", orig)
        cv2.waitKey(0)

cv2.destroyAllWindows()
