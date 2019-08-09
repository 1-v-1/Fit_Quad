import cv2 as cv
import imutils
import numpy as np


def fit_quad(filename):
    image = cv.imread(filename)
    image = cv.resize(image, (512, 512))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    edged = cv.Canny(gray, 30, 200)

    cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.015 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    return image, screenCnt


if __name__ == '__main__':
    for i in range(15):
        filename = 'images/img_back_{}_img.png'.format(i)
        print(filename)
        image, screenCnt = fit_quad(filename)
        cv.imwrite('images/img_{}.jpg'.format(i), image)

        if screenCnt is not None:
            cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
            cv.imwrite('images/out_{}.jpg'.format(i), image)

        print(np.squeeze(screenCnt, 1))
