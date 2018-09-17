import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os

img = cv2.imread('C:/Users/wang/Desktop/final_tons/1-3-1.jpg')
img = cv2.resize(img, (200,200))
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# img1 = cv2.resize(hsv, (30, 30))
# print (img1)

# hsv = cv2.resize(hsv, (30, 30))
# print (hsv)

# img = cv2.imread('C:/Users/wang/Desktop/blue.jpg')

row = np.where(hsv[:,:,1]>80)

for i in range(np.size(row)//2):
    y = row[0][i]
    x = row[1][i]
    hsv[y][x][2] = 0

img0 = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('Tongue',img0)

cv2.waitKey(0)
cv2.destroyAllWindows()






