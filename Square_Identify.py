import cv2
import numpy as np
# import GrabCut
import os

def imageExpand(image,num):
    yOrigin = image.shape[0]
    xOrigin = image.shape[1]
    xyratio = yOrigin/xOrigin
    xNew = int(xOrigin*num)
    yNew = int(xNew*xyratio)
    image = cv2.resize(image,(xNew,yNew))
    return image

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

def rename(path):
    i = 0
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件
        i = i + 1
        Olddir = os.path.join(path, files);  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue;
        filename = os.path.splitext(files)[0];  # 文件名
        filetype = os.path.splitext(files)[1];  # 文件扩展名
        Newdir = os.path.join(path, str(i) + filetype);  # 新的文件路径
        os.rename(Olddir, Newdir)  # 重命名

def findRectangles(img,minArea,maxArea,minWdivH,maxWdivH):

    # initialization
    i = 0
    xyList = []
    subImages = {}
    # preprocess the image
    blurred = cv2.bilateralFilter(img,20,75,75)
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,100,200,3)
    image, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find rectangle for each contour
    for c in contours:
        flag = 0
        x, y, w, h = cv2.boundingRect(c)
        # restrain the size of rectangles
        if w * h > minArea and w * h < maxArea and w / h > minWdivH and w / h < maxWdivH:
            # Eliminate same element
            for k in xyList:
                if (x,y) == k:
                    flag = 1
            if flag == 1:
                continue
            xyList.append((x,y))
            # save rectangle images
            subImages[i] = img[y:y + h, x:x + w]
            i = i+1
    return subImages

# Start to preprocess original images
path = "C:/Users/wang/Desktop/ton_images/2"
filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
for files in filelist:  # 遍历所有图片
    subRects = []
    filename = os.path.splitext(files)[0]
    img_dir = os.path.join(path, files);  # 文件路径
    if os.path.isdir(img_dir):  # 如果是文件夹则跳过
        continue;

    img = cv2.imread(img_dir)
    img = cv2.resize(img,(450,600))
    subRects = findRectangles(img,40000,1000000,0.5,1.5)

    # write to file
    for j in subRects:
        cv2.imwrite('C:/Users/wang/Desktop/square_images_2/2' + '-' + filename + '-' + str(j+1) + '.jpg',subRects[j])
    print (files)




