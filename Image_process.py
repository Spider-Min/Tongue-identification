import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorFliter(image,hue,bias):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_color = np.array([(hue-bias),0,0])
    upper_color = np.array([(hue+bias),255,255])
    mask = cv2.inRange(hsv,lower_color,upper_color)
    res = cv2.bitwise_and(image,image,mask=mask)
    return res

def applySobel(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray,ddepth = cv2.CV_32F, dx=1,dy=0,ksize=-1)
    gradY = cv2.Sobel(gray,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=-1)
    gradient = cv2.subtract(gradX,gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient

def imageExpand(image,num):
    yOrigin = image.shape[0]
    xOrigin = image.shape[1]
    xyratio = yOrigin/xOrigin
    xNew = int(xOrigin*num)
    yNew = int(xNew*xyratio)
    image = cv2.resize(image,(xNew,yNew))
    return image

if "__name__"=="__main__":
    #
    img = cv2.imread("C:/Users/wang/Desktop/ton_images/2.jpg") # read from file
    img = imageExpand(img,0.2)


    img1 = colorFliter(img,30,21)
    img2 = cv2.bitwise_not(img1,img1)
    img = cv2.bitwise_and(img,img2)
    sobel = applySobel(img)
    blurred = cv2.bilateralFilter(img, 20, 75,75) #模糊
    gray1 = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY) #转灰度
    (_, thresh) = cv2.threshold(gray1, 20, 255, cv2.THRESH_BINARY) #设阈值
    kernel =np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)

    dilation = cv2.dilate(erosion,kernel,iterations=1)
    image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    imgCon = cv2.drawContours(img, contours, -1, (0,255,0), 3)


    cv2.imshow('Tongue',dilation)
    # cv2.imshow('Tongue',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

