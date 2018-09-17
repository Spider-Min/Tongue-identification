import cv2
import numpy as np
import AreaCalculator
import Color_functions

def findBaseColor(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    tonArea = AreaCalculator.AreaOfMaxObject(img)

    # set filters to eliminate unqualified pixels
    colorFilter1 = img[:,:,0]>=100
    colorFilter2 = img[:,:,0]<=10
    valueFilter = img[:,:,2]>=100
    filter = (colorFilter1 + colorFilter2) * valueFilter

    xyLocations = np.where(filter[:,:]==True)
    filteredImg = []

    # pick pixels
    for i,j in enumerate(xyLocations[0]):
        x = xyLocations[0][i]
        y = xyLocations[1][i]
        filteredImg.append(img[x][y])

    filteredImg = np.array(filteredImg)
    filteredImg = np.array([filteredImg])

    # count pixel number for pixels with different saturation
    pixelAreas = []
    slide = 16

    for i in range(15,256,slide):
        pixelNum = AreaCalculator.countPixelNum(filteredImg,1,i-(slide-1),i)
        areaEach = round(float(pixelNum/tonArea)*100,1)
        pixelAreas.append(areaEach)

    pixelAreas = np.array(pixelAreas)

    # find the max saturation of which pixels account for at least 3.5% area of all
    colors_HSV = []
    min_area = 3.5

    for i in range(1,8):
        pixelAreas_index = np.where(pixelAreas>=min_area+0.5-i*0.5)[-1]
        if len(pixelAreas_index)!=0:
            break
    if len(pixelAreas_index) == 0:
        print ("tonge color not find")

    for i in pixelAreas_index:
        lowBar = i*slide + 4
        highBar = lowBar+slide - 4

        # calculate average color according the range of saturation
        satFilter_1 = img[:,:,1]>=lowBar
        satFilter_2 = img[:,:,1]<=highBar
        satFilter = satFilter_1 * satFilter_2

        pixels_Location = np.where(satFilter[:,:]==True)
        pixels_list = []

        # pick pixels
        for i,j in enumerate(pixels_Location[0]):
            x = pixels_Location[0][i]
            y = pixels_Location[1][i]
            pixels_list.append(img[x][y])

        pixels_list = np.array(pixels_list).astype(np.int64)

        for i in pixels_list:
            if i[0]>=90:
                i[0] = i[0]-180


        hueAver = sum(pixels_list[:,0])/pixels_list.shape[0]
        if hueAver<0:
            hueAver = hueAver+180
        hueAver = round(hueAver,2)
        satAver = sum(pixels_list[:,1])/pixels_list.shape[0]
        satAver = round(satAver,2)
        valueAver = sum(pixels_list[:,2])/pixels_list.shape[0]
        valueAver = round(valueAver,2)

        #calculate the average of filtered pixels
        new_pixels_list = []
        pixels_list = pixels_list.tolist()
        for i in pixels_list:

            if hueAver>100:
                hueAver_temp = hueAver-180
            else:
                hueAver_temp = hueAver

            if i[0]>100:
                i_temp = i[0]-180
            else:
                i_temp = i[0]

            filter1 = i_temp > hueAver_temp - 30 and i_temp < hueAver_temp + 30
            filter2 = (i[0]>-80 and i[0]<0) or (i[0]<=10 and i[0]>0)
            filter3 = i[1]<satAver+30 and i[1]>satAver-30
            filter4 = i[2]<valueAver+30 and i[2]>valueAver-30
            if filter1 and filter2 and filter3 and filter4:
                new_pixels_list.append(i)

        new_pixels_list = np.array(new_pixels_list).astype(np.int64)


        new_hueAver = round(sum(new_pixels_list[:,0])/new_pixels_list.shape[0],2)
        if new_hueAver<0:
            new_hueAver = new_hueAver+180
        new_satAver = round(sum(new_pixels_list[:,1])/new_pixels_list.shape[0],2)
        new_valueAver = round(sum(new_pixels_list[:,2])/new_pixels_list.shape[0],2)
        colors_HSV.append([new_hueAver,new_satAver,new_valueAver])

    colors_HSV= np.array(colors_HSV)
    colors_RGB = []
    for i in colors_HSV:
        RGB = Color_functions.HSVtoRGB(i)
        colors_RGB.append(RGB)

    colors_RGB = np.array(colors_RGB)

    base = [205, 62, 62]
    distances = []

    for i in colors_RGB:
        distance = Color_functions.distance(i, base)
        distances.append(distance)


    dis_min = min(distances)
    dis_index = distances.index(dis_min)
    lost = int((len(distances)-1-dis_index)/2)
    if  dis_index + lost + 1 > len(distances)-1:
        the_color = colors_RGB[dis_index+lost]
    else:
        the_color = colors_RGB[dis_index+lost+1]

    return the_color

# print (the_color)
if __name__=="__main__":
    colorBlock = np.zeros((200, 200, 3))
    the_color = findBaseColor("C:/Users/wang/Desktop/final_tons/1-6-1.jpg")
    print (the_color)
    colorBlock[:,:,0] = the_color[0]
    colorBlock[:,:,1] = the_color[1]
    colorBlock[:,:,2] = the_color[2]
    colorBlock = np.array(colorBlock).astype('uint8')

    colorBlock = cv2.cvtColor(colorBlock,cv2.COLOR_RGB2BGR)

    cv2.imshow("Base_Color",colorBlock)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






