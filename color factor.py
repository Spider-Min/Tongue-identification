import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread("C:/Users/wang/Desktop/final_tons/1-2-1.jpg")
# img = cv2.resize(img,(200,200))
# img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
a = []
a  = np.zeros((200,200,3))
a[:,:,0] = 15
a[:,:,1] = 255
a[:,:,2] = 255
a = np.array(a).astype('uint8')
# img[:,:,1] = 255
# img[:,:,2] = 255
# img = np.array(img).astype('uint8')

# print (a)

# a = cv2.cvtColor(a,cv2.COLOR_HSV2BGR)
b = cv2.cvtColor(a, cv2.COLOR_HSV2BGR)

# img = cv2.cvtColor(b,cv2.COLOR_HSV2BGR)
cv2.imshow("a",b)
cv2.waitKey(0)
cv2.destroyAllWindows()