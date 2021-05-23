import cv2
import numpy as np
from math import sqrt
from math import factorial
from math import atan
from math import isnan
from math import floor
from math import degrees
from random import randint

img = cv2.imread('blocks_L-150x150.png') # Original Photo
imgResize1 = cv2.resize(img,(300,300)) # Original Photo Scaled up to 300x300
imgResize2 = cv2.resize(img,(500,500)) # Original Photo Scaled up to 500x500
imgFlip1 = cv2.flip(img,-1) #Rotate 180 or flip both
imgFlip2 = cv2.flip(img,0) #Flip on horizontal axis
imgFlip3 = cv2.flip(img,1) #Flip on vertical axis
imgResize1Flip1 = cv2.resize(imgFlip1,(300,300))
imgResize2Flip1 = cv2.resize(imgFlip1,(500,500))
imgResize1Flip2 = cv2.resize(imgFlip2,(300,300))
imgResize2Flip2 = cv2.resize(imgFlip2,(500,500))
imgResize1Flip3 = cv2.resize(imgFlip3,(300,300))
imgResize2Flip3 = cv2.resize(imgFlip3,(500,500))

def DoG(image,kexp,octave):
    arr = []
    for i in range(octave):
        arr.append([])
    k = sqrt(2)
    sigma = 1.6
    exp = 0
    counter = 0
    for j in range(octave):
        height,width,channels = image.shape
        for i in range(1,kexp):
            exp += 1
            a = cv2.GaussianBlur(image,(15,15),((k**exp)*sigma))
            b = cv2.GaussianBlur(image,(15,15),((k**(exp-1))*sigma))
            arr[j].append(a-b)
        image = cv2.resize(image,(int(height/2),int(width/2)))
        exp = 0
        counter += 1
        exp = counter*2
    return arr

def rescale(array):
    for i in range(1,len(array)):
        for j in range(len(array[i])):
            for k in range(len(array[i][j])):
                # print(array[i][j][k]*100) #Works
                array[i][j][k] = (array[i][j][k] * (2**i))
    return array

def peakDetect(array):
    # Create list to return with all feature keyPoints
    finalArr = []
    # Add sublists for each octave
    for i in range(len(array)):
        finalArr.append([])
    #Turn all images to grayscale
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j] = cv2.cvtColor(array[i][j],cv2.COLOR_BGR2GRAY)
    for i in range(len(array)):
        lengthArray = len(array[i])
        for j in range(1,lengthArray-1):
            highImg = array[i][j+1]
            medImg  = array[i][j]
            lowImg  = array[i][j-1]
            height,width = medImg.shape
            for x in range(1,height-1):
                for y in range(1,width-1):
                    ThreeByThree = []
                    for a in range(-1,2):
                        for b in range(-1,2):
                            ThreeByThree.append(lowImg[x+a][y+b])
                            ThreeByThree.append(medImg[x+a][y+b])
                            ThreeByThree.append(highImg[x+a][y+b])
                    mid = ThreeByThree[13]
                    ThreeByThree.pop(13)
                    if mid > max(ThreeByThree):
                        finalArr[i].append([x,y])
                    if mid < min(ThreeByThree):
                        finalArr[i].append([x,y])
    finalArr = rescale(finalArr)
    return finalArr

def HessianMatrix(image,array):
    imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gausBlur = cv2.GaussianBlur(imageGray,(3,3),cv2.BORDER_DEFAULT)
    Ixy = cv2.Sobel(cv2.Sobel(gausBlur,-1,1,1),-1,1,1)
    Ixx = cv2.Sobel(cv2.Sobel(gausBlur,-1,1,0),-1,1,0)
    Iyy = cv2.Sobel(cv2.Sobel(gausBlur,-1,0,1),-1,0,1)
    popList = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            x = array[i][j][0]
            y = array[i][j][1]
            hessian = np.matrix(([Ixx[x][y],Ixy[x][y]],[Ixy[x][y],Iyy[x][y]]))
            eigenVal = np.linalg.eigvals(hessian)
            trace = np.trace(hessian)
            det = np.linalg.det(hessian)
            Req = trace**2/det
            eigen = eigenVal[0]/eigenVal[1]
            if Req < (eigen + 1)**2/eigen:
                popList.append([i,j])
    for i in range(len(popList)-1,0,-1):
        x = popList[i][0]
        y = popList[i][1]
        print(array[x][y])
        array[x].pop(y)
    return array


def drawCircles(image,array):
    for i in range(len(array)):
        b = randint(0,255)
        g = randint(0,255)
        r = randint(0,255)
        color = (b,g,r)
        radius = (i+1)*10
        for j in range(len(array[i])):
            cv2.circle(image,(array[i][j][0],array[i][j][1]),radius,color,1)
    return image

def orienAssign(image,array):
    L = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)
    L = cv2.cvtColor(L,cv2.COLOR_BGR2GRAY)
    orientationArr = []
    counter = 0
    for i in range(len(array)):
        for j in range(len(array[i])):
            orientationArr.append([])
            x = array[i][j][0]
            y = array[i][j][1]
            for a in range(-1,2):
                for b in range(-1,2):
                    x += a
                    y += b
                    magnitude = sqrt(abs(((L[x+1,y]-L[x-1,y])**2)+((L[x,y+1]-L[x,y-1])**2)))
                    angle = degrees(atan(((L[x,y+1])-(L[x,y-1]))/((L[x+1,y])-(L[x-1,y]))))*4
                    orientationArr[counter].append([angle,magnitude])
            counter += 1

    for i in range(len(orientationArr)-1,0,-1): #For loop to get rid of Nan
        for j in range(len(orientationArr[i])):
            if isnan(orientationArr[i][j][0]):
                orientationArr[i][j][0] = 0
    for i in range(len(orientationArr)):
        histArr = []
        for j in range(36):
            histArr.append([])
        for j in range(len(orientationArr[i])):
            orientationPrime = (orientationArr[i][j][0])
            if 0 <= orientationPrime <= 10:
                histArr[0].append(orientationArr[i][j][1])
            elif 11 <= orientationPrime <= 20:
                histArr[1].append(orientationArr[i][j][1])
            elif 21 <= orientationPrime <= 30:
                histArr[2].append(orientationArr[i][j][1])
            elif 31 <= orientationPrime <= 40:
                histArr[3].append(orientationArr[i][j][1])
            elif 41 <= orientationPrime <= 50:
                histArr[4].append(orientationArr[i][j][1])
            elif 51 <= orientationPrime <= 60:
                histArr[5].append(orientationArr[i][j][1])
            elif 61 <= orientationPrime <= 70:
                histArr[6].append(orientationArr[i][j][1])
            elif 71 <= orientationPrime <= 80:
                histArr[7].append(orientationArr[i][j][1])
            elif 81 <= orientationPrime <= 90:
                histArr[8].append(orientationArr[i][j][1])
            elif 91 <= orientationPrime <= 100:
                histArr[9].append(orientationArr[i][j][1])
            elif 101 <= orientationPrime <= 110:
                histArr[10].append(orientationArr[i][j][1])
            elif 111 <= orientationPrime <= 120:
                histArr[11].append(orientationArr[i][j][1])
            elif 121 <= orientationPrime <= 130:
                histArr[12].append(orientationArr[i][j][1])
            elif 131 <= orientationPrime <= 140:
                histArr[13].append(orientationArr[i][j][1])
            elif 141 <= orientationPrime <= 150:
                histArr[14].append(orientationArr[i][j][1])
            elif 151 <= orientationPrime <= 160:
                histArr[15].append(orientationArr[i][j][1])
            elif 161 <= orientationPrime <= 170:
                histArr[16].append(orientationArr[i][j][1])
            elif 171 <= orientationPrime <= 180:
                histArr[17].append(orientationArr[i][j][1])
            elif 181 <= orientationPrime <= 190:
                histArr[18].append(orientationArr[i][j][1])
            elif 191 <= orientationPrime <= 200:
                histArr[19].append(orientationArr[i][j][1])
            elif 201 <= orientationPrime <= 210:
                histArr[20].append(orientationArr[i][j][1])
            elif 211 <= orientationPrime <= 220:
                histArr[21].append(orientationArr[i][j][1])
            elif 221 <= orientationPrime <= 230:
                histArr[22].append(orientationArr[i][j][1])
            elif 231 <= orientationPrime <= 240:
                histArr[23].append(orientationArr[i][j][1])
            elif 241 <= orientationPrime <= 250:
                histArr[24].append(orientationArr[i][j][1])
            elif 251 <= orientationPrime <= 260:
                histArr[25].append(orientationArr[i][j][1])
            elif 261 <= orientationPrime <= 270:
                histArr[26].append(orientationArr[i][j][1])
            elif 271 <= orientationPrime <= 280:
                histArr[27].append(orientationArr[i][j][1])
            elif 281 <= orientationPrime <= 290:
                histArr[28].append(orientationArr[i][j][1])
            elif 291 <= orientationPrime <= 300:
                histArr[29].append(orientationArr[i][j][1])
            elif 301 <= orientationPrime <= 310:
                histArr[30].append(orientationArr[i][j][1])
            elif 311 <= orientationPrime <= 320:
                histArr[31].append(orientationArr[i][j][1])
            elif 321 <= orientationPrime <= 330:
                histArr[32].append(orientationArr[i][j][1])
            elif 331 <= orientationPrime <= 340:
                histArr[33].append(orientationArr[i][j][1])
            elif 341 <= orientationPrime <= 350:
                histArr[34].append(orientationArr[i][j][1])
            elif 351 <= orientationPrime <= 360:
                histArr[35].append(orientationArr[i][j][1])

        for j in range(len(histArr)):
            histArr[j] = sum(histArr[j])

        print('Angle: ',histArr.index(max(histArr))*10) #Angle for arrow to point
        print('Magnitude: ',max(histArr))
        print('\n')

    return 0

def SIFT(image,kexp,octaves):
    x = peakDetect(DoG(image,kexp,octaves))
    x = HessianMatrix(image,x)
    orienAssign(image,x)
    y = drawCircles(image,x)
    cv2.imshow('Circles',y)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()



def main():
    #Original Image SIFT
    # SIFT(img,5,4)

    # #Resized Images
    # SIFT(imgResize1,5,4)
    # SIFT(imgResize2,5,4)
    #
    # #Flipped Images
    # SIFT(imgFlip1,5,4)
    # SIFT(imgFlip2,5,4)
    # SIFT(imgFlip3,5,4)
    #
    # #Resize & Flip
    # SIFT(imgResize1Flip1,5,4)
    # SIFT(imgResize2Flip1,5,4)
    # SIFT(imgResize1Flip2,5,4)
    # SIFT(imgResize2Flip2,5,4)
    # SIFT(imgResize1Flip3,5,4)
    # SIFT(imgResize2Flip3,5,4)

if __name__ == '__main__':
    main()
