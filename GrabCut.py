import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os


i = 0
path = "C:/Users/wang/Desktop/square_images"
filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
for files in filelist:  # 遍历所有图片
    filename = os.path.splitext(files)[0]
    img_dir = os.path.join(path, files);  # 文件路径
    if os.path.isdir(img_dir):  # 如果是文件夹则跳过
        continue;
    img = mpimg.imread(img_dir)
    img = cv2.resize(img, (80, 80))

    mask = np.zeros((80, 80), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    newmask = np.ones((80, 80), np.uint8) * 100
    cv2.rectangle(newmask, (0, 0), (80, 80), 0, 5)
    cv2.line(newmask, (10, 0), (0, 10), 0, 5)
    cv2.line(newmask, (70, 0), (80, 10), 0, 5)
    cv2.line(newmask, (0, 70), (10, 80), 0, 5)
    cv2.line(newmask, (80, 70), (70, 80), 0, 5)
    cv2.circle(newmask, (40, 40), 20, 255, -1)
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    mask[newmask == 100] = 3

    # 函数的返回值是更新的 mask, bgdModel, fgdModel
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img2 = img * mask[:, :, np.newaxis]
    # plt.imshow(img2), plt.colorbar(), plt.show()
    img_rgb = np.zeros(img2.shape, img2.dtype)
    img_rgb[:, :, 0] = img2[:, :, 2]
    img_rgb[:, :, 1] = img2[:, :, 1]
    img_rgb[:, :, 2] = img2[:, :, 0]
    cv2.imwrite('C:/Users/wang/Desktop/final_tons/' +filename + '.jpg', img_rgb)
    print (i)
    i = i+1




















#
# newmask = cv2.imread('newmask.png',0)
# # whereever it is marked white (sure foreground), change mask=1
# # whereever it is marked black (sure background), change mask=0
# mask[newmask == 0] = 0
# mask[newmask == 255] = 1
# mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# print (img.shape)
# rect = (2,2,76,76)
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask[:,:,np.newaxis]