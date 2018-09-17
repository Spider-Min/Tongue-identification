import xlrd
import numpy as np
import os
import cv2
import AreaCalculator
from matplotlib import pyplot as plt

path = 'C:/Users/wang/Desktop/min.xlsx'

data=xlrd.open_workbook(path)

sheet=data.sheet_by_name(u'Sheet1')

rowList = []
imgName = sheet.col_values(0)
saturation = sheet.col_values(1)

for i,j in enumerate(imgName):
    rowList.append([imgName[i],saturation[i]])
rowList = np.array(rowList)

# print (rowList.shape)
# print (rowList[:,0])
newList = []
for i in rowList:
    if i[1] != '':
        saturation = round(float(i[1])*2.55)
        filename = os.path.splitext(i[0])[0]
        filetype = os.path.splitext(i[0])[1]
        newList.append([filename,saturation])

newList = np.array(newList)
# print (newList)


percents = []
areas = []
# Start to preprocess original images
path = "C:/Users/wang/Desktop/final_tons"
tonList = os.listdir(path)  # the name list of images（include folder）
# ton is the number of image
i = 0
for ton in newList:
    img_dir = os.path.join(path, ton[0]+'.jpg')
    # print (img_dir)
    img = cv2.imread(img_dir,1)
    areaMax = AreaCalculator.AreaOfMaxObject(img)
    areas.append(areaMax)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixelNum = AreaCalculator.countPixelNum(hsv, 1, int(ton[1])-8, int(ton[1])+8)
    percent = round((pixelNum / areaMax) * 100, 2)
    percents.append(percent)
    # print (areaMax)
    i = i+1
    print (i)
    # print (ton[0],percent)

areas = np.array(areas)
areas = areas[:,np.newaxis]

percents = np.array(percents)
print (percents)
# print (areas)
print (min(percents))

plt.hist(percents,50);
plt.show()

# hist = cv2.calcHist([percents],[0],None,[2],[0,100])
# plt.plot(hist,color = 'b')
# plt.xlim([0,100])
# plt.show()









