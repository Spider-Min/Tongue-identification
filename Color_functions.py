import cv2
import numpy as np

# 由于cv2里暂时没找到转化单个像素点颜色空间的函数，所以自己写一个
def HSVtoRGB(point):
    p = np.zeros((1,1,3),np.uint8)
    p[:,:,0] = point[0]
    p[:,:,1] = point[1]
    p[:,:,2] = point[2]
    p_rgb = cv2.cvtColor(p,cv2.COLOR_HSV2RGB)
    return [p_rgb[0,0,0],p_rgb[0,0,1],p_rgb[0,0,2]]

def RGBtoHSV(point):
    p = np.zeros((1,1,3),np.uint8)
    p[:,:,0] = point[0]
    p[:,:,1] = point[1]
    p[:,:,2] = point[2]
    p_hsv = cv2.cvtColor(p,cv2.COLOR_RGB2HSV)
    return [p_hsv[0,0,0],p_hsv[0,0,1],p_hsv[0,0,2]]

def distance(point1,point2):

    dis_R = point1[0] - point2[0]
    dis_G = point1[1] - point2[1]
    dis_B = point1[2] - point2[2]
    return np.square(dis_R) + np.square(dis_G) + np.square(dis_B)

if __name__ == '__main__':
    colorBlock = np.zeros((200, 200, 3))
    colorBlock[:, :, 0] = 107
    colorBlock[:, :, 1] = 84
    colorBlock[:, :, 2] = 76
    colorBlock = np.array(colorBlock).astype('uint8')

    colorBlock = cv2.cvtColor(colorBlock, cv2.COLOR_RGB2BGR)

    cv2.imshow("a", colorBlock)
    cv2.waitKey(0)
    cv2.destroyAllWindows()










