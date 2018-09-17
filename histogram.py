import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/wang/Desktop/square_images/1-10-1.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hist = cv2.calcHist([hsv],[1],None,[256],[0,256])
# print (hist[3][35])
# plt.imshow(hist,interpolation = 'nearest')
plt.plot(hist,color = 'b')
plt.xlim([0,256])
plt.show()