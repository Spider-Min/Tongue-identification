import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os

# img = mpimg.imread('C:/Users/wang/Desktop/TONTEST/T2.jpg')
img = mpimg.imread('C:/Users/wang/Desktop/square_images/1-6-1.jpg')
img=cv2.resize(img, (80, 80))
mask = np.zeros((80,80), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
newmask = np.ones((80, 80), np.uint8) * 100
cv2.rectangle(newmask, (0, 0), (80, 80), 0, 5)
cv2.line(newmask, (10, 0), (0, 10), 0, 5)
cv2.line(newmask, (70, 0), (80, 10), 0, 5)
cv2.line(newmask, (0, 70), (10, 80), 0, 5)
cv2.line(newmask, (80, 70), (70, 80), 0, 5)
cv2.circle(newmask, (40, 40), 10, 255, -1)
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask[newmask == 100] = 3

mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img2 = img * mask[:, :, np.newaxis]
plt.imshow(img2), plt.colorbar(), plt.show()