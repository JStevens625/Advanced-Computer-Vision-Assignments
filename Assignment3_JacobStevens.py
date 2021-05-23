import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift

#Image Reading
meanshiftimg1 = cv2.imread('input data/Mean shift/S00-150x150.png')
meanshiftimg2 = cv2.imread('input data/Mean shift/S02-150x150.png')
meanshiftimg3 = cv2.imread('input data/Mean shift/S03-150x150.png')
meanshiftimg4 = cv2.imread('input data/Mean shift/set1Seq2_L-150x150.png')
meanshiftimg5 = cv2.imread('input data/Mean shift/set2Seq1_L-150x150.png')

OTSU2_1 = cv2.imread('input data/OTSU 2 classes/andreas_L-150x150.png',0)
OTSU2_2 = cv2.imread('input data/OTSU 2 classes/edge_L-150x150.png',0)
OTSU2_3 = cv2.imread('input data/OTSU 2 classes/south_L-150x150.png',0)

OTSUMult_1 = cv2.imread('input data/OTSU multiple classes/blocks_L-150x150.png',0)
OTSUMult_2 = cv2.imread('input data/OTSU multiple classes/S01-150x150.png',0)
OTSUMult_3 = cv2.imread('input data/OTSU multiple classes/S04-150x150.png',0)
OTSUMult_4 = cv2.imread('input data/OTSU multiple classes/S05-150x150.png',0)

def histogram(image):
    histogram = cv2.calcHist([image],[0],None,[256],[0,256])
    histogram = np.int32(np.around(histogram,0))
    return histogram

def newImgThresholdTop(Threshold,histogram):
    Threshold = int(Threshold)
    height = 0
    TotalNumberofPixels = 0
    for i in range(Threshold,255):
        TotalNumberofPixels += histogram[i][0]
    for j in range(TotalNumberofPixels-1,0,-1):
        if TotalNumberofPixels % j == 0:
            height = j
            break
    width = int(round(TotalNumberofPixels/height,0))
    img = np.zeros((width,height), dtype=np.uint8)
    counter = Threshold
    for x in range(width):
        for y in range(height):
            img[x][y] = counter
            histogram[counter] -= 1
            if histogram[counter] == 0:
                counter += 1
    return img

def newImgThresholdBottom(Threshold,histogram):
    Threshold = int(Threshold)
    height = 0
    TotalNumberofPixels = 0
    for i in range(Threshold):
        TotalNumberofPixels += histogram[i][0]
    for j in range(TotalNumberofPixels-1,0,-1):
        if TotalNumberofPixels % j == 0:
            height = j
            break
    width = int(round(TotalNumberofPixels/height,0))
    img = np.zeros((width,height), dtype=np.uint8)
    counter = 0
    for x in range(width):
        for y in range(height):
            img[x][y] = counter
            histogram[counter] -= 1
            if histogram[counter] == 0:
                counter += 1
    return img

def SeparateImage(image,thresholdbot,thresholdmid,thresholdtop):
    height,width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i][j] < thresholdbot:
                image[i][j] = 0
            elif image[i][j] >= thresholdbot and image[i,j] < thresholdmid:
                image[i][j] = 86
            elif image[i][j] >= thresholdmid and image[i,j] < thresholdtop:
                image[i][j] = 172
            elif image[i][j] >= thresholdtop:
                image[i][j] = 255
    # print(image)
    return image

def OTSUClasses(image):
    ret,newImg = cv2.threshold(image,127,255,cv2.THRESH_OTSU)
    return newImg

def OTSUMultipleClasses(image):
    ThreshMiddle,newImg = cv2.threshold(image,127,255,cv2.THRESH_OTSU)
    img1 = newImgThresholdBottom(ThreshMiddle,histogram(image))
    img2 = newImgThresholdTop(ThreshMiddle,histogram(image))
    ThreshBottom,newimg1 = cv2.threshold(img1,127,255,cv2.THRESH_OTSU)
    ThreshTop,newimg2 = cv2.threshold(img2,127,255,cv2.THRESH_OTSU)
    newImage = SeparateImage(image.copy(),ThreshBottom,ThreshMiddle,ThreshTop)
    return newImage

def MeanShiftFunc(image): #Set Bandwidth Parameter
    image = cv2.resize(image,(75,75)) # resize image to not take to long
    ms = MeanShift(bandwidth=10) # Save to variable to not call MeanShift() every time
    height,width,channels = image.shape #Get the height and width of image
    imageLUV = cv2.cvtColor(image,cv2.COLOR_BGR2LUV) #Convert to LUV for easier with Mean Shift
    L = imageLUV[:][:][0] #All L Channel Elements
    U = imageLUV[:][:][1] #All U Channel Elements
    LU = np.empty((height*width,2)) #Create Empty array with 0
    for i in range(height):
        for j in range(width):
            LU[(i*width)+j][0] = imageLUV[i][j][0]
            LU[(i*width)+j][1] = imageLUV[i][j][1]

    ms.fit(LU)
    labels = ms.labels_

    # print("Number of Clusters:",len(np.unique(labels)))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    maxinum = int(max(labels))
    for i in range(len(labels)):
        labels[i] = int(labels[i]*(255/maxinum))
    # newImg = np.reshape(labels,(25,25))
    # newImg.astype(np.uint8)
    for i in range(height):
        for j in range(width):
            image[i][j] = labels[(i*width)+j]
    image = cv2.resize(image,(150,150))
    return image

#######################################################################################
#OTSU 2 classes DONE
# cv2.imshow('OTSU 2 Classes #1',np.hstack((OTSU2_1,OTSUClasses(OTSU2_1))))
# cv2.imshow('OTSU 2 Classes #2',np.hstack((OTSU2_2,OTSUClasses(OTSU2_2))))
# cv2.imshow('OTSU 2 Classes #3',np.hstack((OTSU2_3,OTSUClasses(OTSU2_3))))
#######################################################################################
#OTSU Multiple classes DONE
# cv2.imshow('OTSU 2 Multiple Classes #1',np.hstack((OTSUMult_1,OTSUMultipleClasses(OTSUMult_1))))
# cv2.imshow('OTSU 2 Multiple Classes #2',np.hstack((OTSUMult_2,OTSUMultipleClasses(OTSUMult_2))))
# cv2.imshow('OTSU 2 Multiple Classes #3',np.hstack((OTSUMult_3,OTSUMultipleClasses(OTSUMult_3))))
# cv2.imshow('OTSU 2 Multiple Classes #4',np.hstack((OTSUMult_4,OTSUMultipleClasses(OTSUMult_4))))
#######################################################################################
#Mean Shift
# cv2.imshow('Mean Shift #1 Mean Shift applied',MeanShiftFunc(meanshiftimg1.copy()))
# cv2.imshow('Mean Shift #1',meanshiftimg1)
# cv2.imshow('Mean Shift #2 Mean Shift applied',MeanShiftFunc(meanshiftimg2.copy()))
# cv2.imshow('Mean Shift #2',meanshiftimg2)
# cv2.imshow('Mean Shift #3 Mean Shift applied',MeanShiftFunc(meanshiftimg3.copy()))
# cv2.imshow('Mean Shift #3',meanshiftimg3)
# cv2.imshow('Mean Shift #4 Mean Shift applied',MeanShiftFunc(meanshiftimg4.copy()))
# cv2.imshow('Mean Shift #4',meanshiftimg4)
# cv2.imshow('Mean Shift #5 Mean Shift applied',MeanShiftFunc(meanshiftimg5.copy()))
# cv2.imshow('Mean Shift #5',meanshiftimg5)
#######################################################################################

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
