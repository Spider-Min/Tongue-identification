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

img = cv2.imread("C:/Users/wang/Desktop/final_tons/1-171-1.jpg")#image read be 'gray'
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

print ("this is a test")


cv2.imshow("shit",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

