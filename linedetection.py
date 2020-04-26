import numpy as np
from cv2 import cv2
import imutils

def do_canny(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray[:, :, 1] = 0
    #gray[:, :, 2] = 0

    cv2.imshow('gr', gray)
    cv2.waitKey()
    gray = cv2.cvtColor(gray, cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    _, gray = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    gray = 255 - gray

    blur = cv2.GaussianBlur(gray, (3, 3), 10)
    cv2.imshow('blur', blur)
    cv2.waitKey()

    #kernel = np.ones((3, 3), np.uint8)
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], np.uint8)
    erosion = cv2.dilate(gray, kernel, iterations=5)
    erosion = cv2.erode(erosion, kernel, iterations=3)
    cv2.imshow('erode', erosion)
    cv2.waitKey()

    canny = cv2.Canny(erosion, 50, 150)
    cv2.imshow("canny", canny)
    cv2.waitKey()

    lines = cv2.HoughLines(canny,1,1 * np.pi/180,120)
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('lines',frame)
    cv2.waitKey()



def do_circles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    gray = 255 - gray
    cv2.imshow('bw', gray)
    cv2.waitKey()

    kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], np.uint8)

    kernel2 = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], np.uint8)
    erosion = cv2.dilate(gray, kernel2, iterations=5)
    cv2.imshow('dilate', erosion)
    cv2.waitKey()

    erosion = cv2.erode(erosion, kernel2, iterations=5)
    cv2.imshow('rode', erosion)
    cv2.waitKey()

    erosion = cv2.dilate(erosion, kernel2, iterations=5)
    cv2.imshow('dilate2', erosion)
    cv2.waitKey()

    erosion = cv2.erode(erosion, kernel2, iterations=5)
    cv2.imshow('rode2', erosion)
    cv2.waitKey()

    #erosion = cv2.dilate(erosion, kernel, iterations=2)
    #cv2.imshow('dilate', erosion)
    #cv2.waitKey()

    canny = cv2.Canny(erosion, 50, 150)
    cv2.imshow('canny', canny)
    cv2.waitKey()

    circles = cv2.HoughCircles(erosion, cv2.HOUGH_GRADIENT, 2, 50,
                                param1=150,param2=100, minRadius=50)


    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(erosion,(i[0],i[1]),i[2],(255,255,255),2)
        # draw the center of the circle
        cv2.circle(erosion,(i[0],i[1]),2,(255,255,255),3)

    cv2.imshow('circles', erosion)
    cv2.waitKey()


def dilate_erode(inputimg, kernel, iterations):
    dil = cv2.dilate(inputimg, kernel, iterations=iterations)
    erod = cv2.erode(dil, kernel, iterations=iterations)
    return erod

def do_rectangles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)[1]
    thresh = 255 - thresh
    cv2.imshow('thresh', thresh)
    cv2.waitKey()

    kernel = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ], np.uint8)
    kernel3 = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], np.uint8)
    #dilation = cv2.dilate(thresh, kernel, iterations=4)
    #erosion = cv2.erode(dilation, kernel, iterations=2)
    #
    erosion = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel3)
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel3)
    
    cv2.imshow('erosion7', erosion)
    cv2.imshow('dilation7', dilation)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('dilation3', dilation)
    cv2.waitKey()

    #exit()

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	                            cv2.CHAIN_APPROX_NONE)
    
    cnts = imutils.grab_contours(cnts)
    disp = img.copy()
    clr = [0, 0, 0]
    for i in range(len(cnts)):
        idx = i % 3
        clr[idx] = np.random.randint(0, 256)
        #x, y, w, h = cv2.boundingRect(cnts[i])
        cv2.drawContours(disp, cnts, i, tuple(clr), 2)
        #cv2.rectangle(disp, (x, y), (x + w, y + h), clr, 2)

        rect = cv2.minAreaRect(cnts[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(disp, [box], 0, tuple(clr), 2)

    cv2.imshow('disp', disp)
    cv2.waitKey()

    for c in cnts:
        eps = 0.045 * cv2.arcLength(c, False)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) > 3:
            cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()

img = cv2.imread('yolo/images/basketball4.jpg')
#do_canny(img)
do_rectangles(img)
