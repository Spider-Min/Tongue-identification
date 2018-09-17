import cv2
import numpy as np
import matplotlib.pyplot as plt
import Image_process

img = cv2.imread('C:/Users/wang/Desktop/final_tons/1-7-1.jpg')
img = Image_process.imageExpand(img, 2)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("shit",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()