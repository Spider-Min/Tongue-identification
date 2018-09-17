import cv2
import numpy as np
from scipy.optimize import fsolve
import FindColor
import Color_functions
from matplotlib import pyplot as plt
import os

# alpha 为不透明度(0表示完全透明，1表示完全不透明)
def rangeAlpha(color_mix,color_base):
    alpha_list = []
    if color_mix[0] > color_base[0]:
        alpha_list.append(round((color_mix[0]-color_base[0])/(255-color_base[0]),2))
    else:
        alpha_list.append(round((color_base[0]-color_mix[0])/color_base[0],2))

    if color_mix[1] > color_base[1]:
        alpha_list.append(round((color_mix[1]-color_base[1])/(255-color_base[1]),2))
    else:
        alpha_list.append(round((color_base[1]-color_mix[1])/color_base[1],2))

    if color_mix[2] > color_base[2]:
        alpha_list.append(round((color_mix[2]-color_base[2])/(255-color_base[2]),2))
    else:
        alpha_list.append(round((color_base[2]-color_mix[2])/color_base[2],2))
    # print (alpha_list)

    alpha_min = max(alpha_list)
    if alpha_min != 0:
        alpha_min = alpha_min-round(alpha_min%0.1,2)+0.1

    return np.arange(alpha_min,1.01,0.1) # [alpha_max, alpha_max + 0.1, alpha_max + 0.2, ..., 1]

def isCoat(color_base,color_rgb):
    color_hsv = Color_functions.RGBtoHSV(color_rgb)
    # if (color_hsv[0]<=180 and color_hsv[0]>=150) or (color_hsv[0]>=0 and color_hsv[0]<14):
    #     if (color_hsv[1]<50):
    #         return True
    #     else:
    #         return False
    if (color_hsv[0]>=14 and color_hsv[0]<80):
        return True
    else:
        if color_hsv[1]>50:
            return False
        else:
            return True

def solveColor(alpha_range,color_mix,color_base):

    colors = []
    distances_normal = []
    distances_yellow = []
    # the ideal color of the coating on the tonge, can be adjusted
    normal_color = [235,230,210]
    # the ideal color of the yellow coating on the tonge, can be adjusted
    yellow_color = [200,200,100]

    for alpha in alpha_range:
        # according to the possible values of alpha, calculate possible colors
        def func(i):
            r,g,b = i[0],i[1],i[2]
            return [
                color_base[0] * (1 - alpha) + r * alpha - color_mix[0],
                color_base[1] * (1 - alpha) + g * alpha - color_mix[1],
                color_base[2] * (1 - alpha) + b * alpha - color_mix[2],
            ]
        color = fsolve(func,[0,0,0])
        np.round(color)
        # pick the color which is most similiar to the normal color or yellow color
        distance_normal = np.square(normal_color[0]-color[0])+np.square(normal_color[1]-color[1])+np.square(normal_color[2]-color[2])
        distance_yellow = np.square(yellow_color[0]-color[0])+np.square(yellow_color[1]-color[1])+np.square(yellow_color[2]-color[2])
        colors.append(color)

        # green color will not be considered as coating color
        if color[0] * 1.1 > color[1]:
            distances_normal.append(distance_normal)
            distances_yellow.append(distance_yellow)
        else:
            distances_normal.append(distance_normal+300000)
            distances_yellow.append(distance_yellow+300000)
    colors = np.array(colors)
    # print (colors)
    min_normal = min(distances_normal)
    min_yellow = min(distances_yellow)
    distance_min = min(min_normal,min_yellow)

    if min_normal<min_yellow:
        index = distances_normal.index(min(distances_normal))
    else:
        index = distances_yellow.index(min(distances_yellow))

    # Whether is coat
    if isCoat(color_base,colors[index]):
        return [colors[index], round(alpha_range[index],1)]
    else:
        return [colors[index],0]

if __name__ == '__main__':
    path = "C:/Users/wang/Desktop/tonge_color/train/4"
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）

    for files in filelist:  # 遍历所有图片
        try:
            filename = os.path.splitext(files)[0]
            print (filename)
            img_dir = os.path.join(path, files);  # 文件路径
            if os.path.isdir(img_dir):  # 如果是文件夹则跳过
                continue;
            # img_path = 'C:/Users/wang/Desktop/identify/final_tons/1-17-1.jpg'
            img_bgr = cv2.imread(img_dir)

            img_hsv= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_origin = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

            # filter
            filt = img_hsv[:, :, 2] >= 25
            xyLocations = np.where(filt[:, :] == True)

            # preprocess, eliminate dark points
            value_min = 80
            img_rgb= cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
            img_rgb[img_hsv[:,:,2]<80]=np.array([0,0,0])

            # the color of tonge
            color_base = FindColor.findBaseColor(img_dir)
            print(color_base)
            img_thickness =  np.zeros((img_bgr.shape[0],img_bgr.shape[1]))
            img_thickness = np.array(img_thickness).astype('uint8')

            tonge_coating = np.zeros((img_bgr.shape[0],img_bgr.shape[1],3))
            tonge_coating = np.array(tonge_coating).astype('uint8')

            tonge = np.zeros((img_bgr.shape[0],img_bgr.shape[1],3))
            tonge = np.array(tonge).astype('uint8')

            tonge_coating_mix = np.zeros((img_bgr.shape[0],img_bgr.shape[1],3))
            tonge_coating_mix= np.array(tonge_coating).astype('uint8')

            for i, j in enumerate(xyLocations[0]):
                x = xyLocations[0][i]
                y = xyLocations[1][i]
                # print(x,y)
                if (img_rgb[x][y]!=[0,0,0]).all():
                    color_mix = img_rgb[x][y]
                    alpha_range = rangeAlpha(color_mix, color_base)
                    colors = solveColor(alpha_range, color_mix, color_base)
                    img_thickness[x][y] = colors[1]*255
                    if img_thickness[x][y] != 0:
                        tonge_coating[x][y] = colors[0]
                        tonge_coating_mix[x][y] = colors[0]*colors[1]
                        tonge[x][y]  =  np.array([0, 0, 0]).astype('uint8')
                        if img_thickness[x][y] <= 0.2:
                            tonge[x][y] = img_origin[x][y]
                    else:
                        tonge_coating[x][y] = np.array([0, 0, 0]).astype('uint8')
                        tonge_coating_mix[x][y] = np.array([0, 0, 0]).astype('uint8')
                        tonge[x][y] = img_origin[x][y]
                else:
                    img_thickness[x][y] = 0

            tonge_coating = cv2.cvtColor(tonge_coating, cv2.COLOR_RGB2BGR)
            tonge = cv2.cvtColor(tonge, cv2.COLOR_RGB2BGR)
            cv2.imwrite('C:/Users/wang/Desktop/tonge_color/train_new/4/' + filename + '.jpg', tonge)
        except:
            print("The color of tonge is not found!")
            pass
        continue


        # p1=plt.subplot(231), plt.imshow(img_origin)
        # plt.title('original image')
        # p2=plt.subplot(232), plt.imshow(tonge_coating)
        # plt.title('tonge coating image')
        # p3=plt.subplot(233), plt.imshow(img_thickness,cmap='gray')
        # plt.title('tonge coating thickness')
        # p4=plt.subplot(234),plt.imshow(tonge_coating_mix)
        # plt.title('tonge coating mutiple thickness')
        # p5=plt.subplot(235),plt.imshow(tonge)
        # plt.title('tonge')
        # plt.show()
        # color_mix = [250, 250, 250]


        # alpha_range = rangeAlpha(color_mix,color_base)
        #
        # print (alpha_range)
        #
        # colors = solveColor(alpha_range,color_mix,color_base)
        # print (colors)






