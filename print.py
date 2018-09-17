import cv2
import numpy as np
import Image_process
img_path = 'C:/Users/wang/Desktop/printshit.jpg'
img = cv2.imread(img_path)
img = Image_process.imageExpand(img, 0.5)

colorFilter1 = img[:, :, 0]/img[:,:,1] >= 0.8
colorFilter2 = img[:, :, 0]/img[:,:,1] <= 1.2
colorFilter3 = img[:, :, 0]/img[:,:,2] >= 0.8
colorFilter4 = img[:, :, 0]/img[:,:,2] <= 1.2
filter = colorFilter1 * colorFilter2 * colorFilter3 * colorFilter4

xyLocations = np.where(filter[:, :] == True)
filteredImg = []

# pick pixels
for i, j in enumerate(xyLocations[0]):
    x = xyLocations[0][i]
    y = xyLocations[1][i]
    img[x][y] = np.array([255,255,255]).astype('uint8')

cv2.imwrite('C:/Users/wang/Desktop/printshit2.jpg',img)
cv2.imshow("a", img)
cv2.waitKey(0)
cv2.destroyAllWindows()