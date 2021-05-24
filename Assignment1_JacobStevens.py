import cv2
import numpy as np

cap = cv2.VideoCapture('Data/Assignment-1/barriers.avi')
screenWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screenHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameCounter = 0 # counts frame to loop video
TotalFrameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT) #All frames in video counted
fps = round(1000/cap.get(cv2.CAP_PROP_FPS)) #Frames per second
fourcc = cv2.VideoWriter_fourcc(*'XVID')

def onChangeBrightness(brightness):
    return(brightness)

def onChangeContrast(contrast):
    return(contrast)

#Create Window for Slider Bars and bars
editedVideo = cv2.namedWindow("Edited Video") #Create window called "Edited Video"
cv2.createTrackbar('Brightness','Edited Video',50,255,onChangeBrightness) #Create Slider Bar
cv2.createTrackbar('Contrast','Edited Video',100,200,onChangeContrast) #Create Slider Bar

while(cap.isOpened):

    ret,frame = cap.read()
    gray1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Turn Video 1 Grayscale
    gray2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Turn Video 2 Grayscale Edited Video

    #Looping Video Section
    frameCounter += 1 #Increase frameCounter for every frame
    if frameCounter == TotalFrameCount: #Check when video reaches end
        cap.set(cv2.CAP_PROP_POS_FRAMES,0) #sets video back to start for loop
        frameCounter = 0 #Resets framecounter to start from begining

    #SliderBars for Video 2
    currentContrast = ((cv2.getTrackbarPos('Contrast','Edited Video'))/100) #Grabs the change in contrast and divides by 100 as contrast cannot go below 0 and above 1
    currentBrightness = cv2.getTrackbarPos('Brightness', 'Edited Video') #Grabs the change in brightness and adds it to frame
    gray2 = cv2.addWeighted(gray2,currentContrast,gray2,0,currentBrightness)

    #Histogram for Video 1 Section
    histogramWindow1 = np.zeros((256,256,3)) #Black Window Containing histogram lines
    histogram = cv2.calcHist([gray1],[0],None,[256],[0,256]) #Create histogram on Grayscale Video
    histogram = np.int32(np.around(histogram,0)) #np.around() rounds to specific decimal place
    for x,y in enumerate(histogram): #Loop to put what type of pixel goes where
        cv2.line(histogramWindow1,(x,256),(x,256 - y),(0,255,0)) #Create lines for the enumeration

    #Histogram for Video 2 Section
    histogramWindow2 = np.zeros((256,256,3)) #Black Window Containing histogram lines
    histogram = cv2.calcHist([gray2],[0],None,[256],[0,256]) #Create histogram on Grayscale Video
    histogram = np.int32(np.around(histogram,0)) #np.around() rounds to specific decimal place
    for x,y in enumerate(histogram): #Loop to put what type of pixel goes where
        cv2.line(histogramWindow2,(x,256),(x,256 - y),(0,255,0)) #Create lines for the enumeration

    #Display all windows
    cv2.imshow('Video 1',gray1) #Display Grayscale Video 1
    cv2.imshow('Video 2',gray2) #Display Grayscale Video 2
    cv2.imshow('Video 1 Histogram',histogramWindow1) #Window for Histogram for Video 1
    cv2.imshow('Video 2 Histogram',histogramWindow2) #Window for Histogram for Video 2

    #Exit Playback
    if cv2.waitKey(fps) == 27: #If ESC key hit, end video playback
        break

    #Saving Video
    if cv2.waitKey(fps) == ord('s'): #If s key hit, Save Current video
        setContrast = currentContrast
        setBrightness = currentBrightness
        cv2.destroyAllWindows()
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        output = cv2.VideoWriter('OutputVideo.avi',fourcc, 50, (screenWidth,screenHeight))
        while(cap.isOpened()):
            ret,frame = cap.read()
            output.write(cv2.addWeighted(frame,setContrast,frame,0,setBrightness))
        output.release()
        break
        cv2.destroyAllWindows()

#Release Resources
cap.release()
cv2.destroyAllWindows()
