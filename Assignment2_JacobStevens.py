import cv2
import numpy as np

img1 = cv2.imread('dog.bmp')
img2 = cv2.imread('bicycle.bmp')
img3 = cv2.imread('edge_L-150x150.png')
img4 = cv2.imread('south_L-150x150.png')

def boxFilter(frame,alpha):
    kernel = np.ones((alpha,alpha))/(alpha*alpha)
    output = cv2.filter2D(frame,-1,kernel)
    return output

def boxFilterOpenCV(frame,alpha):
    return cv2.boxFilter(frame,-1,(alpha,alpha),cv2.BORDER_DEFAULT)

def sobelFilterXAxis(frame,kernel):
    if kernel == 3:
        kernel = np.array(([-1,0,1],
                           [-2,0,2],
                           [-1,0,1]))
        output = cv2.filter2D(frame,-1,kernel)
        return output

    if kernel == 7:
        kernel = np.array(([3,2,1,0,-1,-2,-3],
                           [4,3,2,0,-2,-3,-4],
                           [5,4,3,0,-3,-4,-5],
                           [6,5,4,0,-4,-5,-6],
                           [5,4,3,0,-3,-4,-5],
                           [4,3,2,0,-2,-3,-4],
                           [3,2,1,0,-1,-2,-3]))
        output = cv2.filter2D(frame,-1,kernel)
        return output

def sobelFilterYAxis(frame,kernel):
    if kernel == 3:
        kernel = np.array(([-1,-2,-1],
                           [0,0,0],
                           [1,2,1]))
        output = cv2.filter2D(frame,-1,kernel)
        return output
    if kernel == 7:
        kernel = np.array(([-3,-4,-5,-6,-5,-4,-3],
                           [-2,-3,-4,-5,-4,-3,-2],
                           [-1,-2,-3,-4,-3,-2,-1],
                           [0,0,0,0,0,0,0],
                           [1,2,3,4,3,2,1],
                           [2,3,4,5,4,3,2],
                           [3,4,5,6,5,4,3]))
        output = cv2.filter2D(frame,-1,kernel)
        return output

def sobelFilterXAndYAxis(frame,ksize):
    X = sobelFilterXAxis(frame,ksize)
    Y = sobelFilterYAxis(frame,ksize)
    output = cv2.bitwise_or(X,Y)
    return output

def sobelFilterXAndYAxisOpenCV(frame,alpha):
    return cv2.Sobel(frame,-1,1,1,ksize=alpha)

def gaussianBlurOpenCV(frame,alpha):
    return cv2.GaussianBlur(frame,(alpha,alpha),cv2.BORDER_DEFAULT)

#######################################################################################
# # Box Filter
#     # Dog
cv2.imshow("Box Filter 3x3 Dog",np.hstack((img1,boxFilter(img1,3))))
# cv2.imshow("Box Filter 7x7 Dog",np.hstack((img1,boxFilter(img1,7))))
#
#     # Bicycle
# cv2.imshow("Box Filter 3x3 Bicycle",np.hstack((img2,boxFilter(img2,3))))
# cv2.imshow("Box Filter 7x7 Bicycle",np.hstack((img2,boxFilter(img2,7))))
#
#     # edge_L
# cv2.imshow("Box Filter 3x3 edge_L",np.hstack((img3,boxFilter(img3,3))))
# cv2.imshow("Box Filter 7x7 edge_L",np.hstack((img3,boxFilter(img3,7))))
#
#     # south_L
# cv2.imshow("Box Filter 3x3 south_L",np.hstack((img4,boxFilter(img4,3))))
# cv2.imshow("Box Filter 7x7 south_L",np.hstack((img4,boxFilter(img4,7))))

#######################################################################################

# # Box Filter OpenCV
#     # Dog
# cv2.imshow("Box Filter OpenCV 3x3 Dog", np.hstack((img1,boxFilterOpenCV(img1,3))))
# cv2.imshow("Box Filter OpenCV 7x7 Dog", np.hstack((img1,boxFilterOpenCV(img1,7))))
#
#     # Bicycle
# cv2.imshow("Box Filter OpenCV 3x3 Bicycle", np.hstack((img2,boxFilterOpenCV(img2,3))))
# cv2.imshow("Box Filter OpenCV 7x7 Bicycle", np.hstack((img2,boxFilterOpenCV(img2,7))))
#
#     # edge_L
cv2.imshow("Box Filter OpenCV 3x3 edge_L", np.hstack((img3,boxFilterOpenCV(img3,3))))
# cv2.imshow("Box Filter OpenCV 7x7 edge_L", np.hstack((img3,boxFilterOpenCV(img3,7))))
#
#     # south_L
# cv2.imshow("Box Filter OpenCV 3x3 south_L", np.hstack((img4,boxFilterOpenCV(img4,3))))
# cv2.imshow("Box Filter OpenCV 7x7 south_L", np.hstack((img4,boxFilterOpenCV(img4,7))))

#######################################################################################

# # Sobel Filter X Axis
#     # Dog
# cv2.imshow("Sobel Filter X Axis 3x3 Dog",np.hstack((img1,sobelFilterXAxis(img1,3))))
# cv2.imshow("Sobel Filter X Axis 7x7 Dog",np.hstack((img1,sobelFilterXAxis(img1,7))))
#
#     # Bicycle
# cv2.imshow("Sobel Filter X Axis 3x3 Bicycle",np.hstack((img2,sobelFilterXAxis(img2,3))))
# cv2.imshow("Sobel Filter X Axis 7x7 Bicycle",np.hstack((img2,sobelFilterXAxis(img2,7))))
#
#     # edge_L
# cv2.imshow("Sobel Filter X Axis 3x3 edge_L",np.hstack((img3,sobelFilterXAxis(img3,3))))
# cv2.imshow("Sobel Filter X Axis 7x7 edge_L",np.hstack((img3,sobelFilterXAxis(img3,7))))
#
#     # south_L
# cv2.imshow("Sobel Filter X Axis 3x3 south_L",np.hstack((img4,sobelFilterXAxis(img4,3))))
cv2.imshow("Sobel Filter X Axis 7x7 south_L",np.hstack((img4,sobelFilterXAxis(img4,7))))

#######################################################################################

# # Sobel Filter Y Axis
#     # Dog
# cv2.imshow("Sobel Filter Y Axis 3x3 Dog",np.hstack((img1,sobelFilterYAxis(img1,3))))
# cv2.imshow("Sobel Filter Y Axis 7x7 Dog",np.hstack((img1,sobelFilterYAxis(img1,7))))
#
#     # Bicycle
cv2.imshow("Sobel Filter Y Axis 3x3 Bicycle",np.hstack((img2,sobelFilterYAxis(img2,3))))
# cv2.imshow("Sobel Filter Y Axis 7x7 Bicycle",np.hstack((img2,sobelFilterYAxis(img2,7))))
#     # edge_L
# cv2.imshow("Sobel Filter Y Axis 3x3 edge_L",np.hstack((img3,sobelFilterYAxis(img3,3))))
# cv2.imshow("Sobel Filter Y Axis 7x7 edge_L",np.hstack((img3,sobelFilterYAxis(img3,7))))
#
#     # south_L
# cv2.imshow("Sobel Filter Y Axis 3x3 south_L",np.hstack((img4,sobelFilterYAxis(img4,3))))
# cv2.imshow("Sobel Filter Y Axis 7x7 south_L",np.hstack((img4,sobelFilterYAxis(img4,7))))

#######################################################################################

# # Sobel Filter X and Y Axis
#     # Dog
# cv2.imshow("Sobel Filter X and Y Axis 3x3 Dog",np.hstack((img1,sobelFilterXAndYAxis(img1,3))))
# cv2.imshow("Sobel Filter X and Y Axis 7x7 Dog",np.hstack((img1,sobelFilterXAndYAxis(img1,7))))
#
#     # Bicycle
# cv2.imshow("Sobel Filter X and Y Axis 3x3 Bicycle",np.hstack((img2,sobelFilterXAndYAxis(img2,3))))
# cv2.imshow("Sobel Filter X and Y Axis 7x7 Bicycle",np.hstack((img2,sobelFilterXAndYAxis(img2,7))))
#
#     # edge_L
# cv2.imshow("Sobel Filter X and Y Axis 3x3 edge_L",np.hstack((img3,sobelFilterXAndYAxis(img3,3))))
# cv2.imshow("Sobel Filter X and Y Axis 7x7 edge_L",np.hstack((img3,sobelFilterXAndYAxis(img3,7))))
#
#     # south_L
# cv2.imshow("Sobel Filter X and Y Axis 3x3 south_L",np.hstack((img4,sobelFilterXAndYAxis(img4,3))))
# cv2.imshow("Sobel Filter X and Y Axis 7x7 south_L",np.hstack((img4,sobelFilterXAndYAxis(img4,7))))

#######################################################################################

# # Sobel Filter on the X And Y Axis OpenCV
#     # Dog
# cv2.imshow("Sobel Filter 3x3 OpenCV Dog", np.hstack((img1,sobelFilterXAndYAxisOpenCV(img1,3))))
# cv2.imshow("Sobel Filter 7x7 OpenCV Dog", np.hstack((img1,sobelFilterXAndYAxisOpenCV(img1,7))))
#
#     # Bicycle
# cv2.imshow("Sobel Filter 3x3 OpenCV Bicycle", np.hstack((img2,sobelFilterXAndYAxisOpenCV(img2,3))))
# cv2.imshow("Sobel Filter 7x7 OpenCV Bicycle", np.hstack((img2,sobelFilterXAndYAxisOpenCV(img2,7))))
#
#     # edge_L
# cv2.imshow("Sobel Filter 3x3 OpenCV edge_L", np.hstack((img3,sobelFilterXAndYAxisOpenCV(img3,3))))
# cv2.imshow("Sobel Filter 7x7 OpenCV edge_L", np.hstack((img3,sobelFilterXAndYAxisOpenCV(img3,7))))
#
#     # south_L
# cv2.imshow("Sobel Filter 3x3 OpenCV south_L", np.hstack((img4,sobelFilterXAndYAxisOpenCV(img4,3))))
# cv2.imshow("Sobel Filter 7x7 OpenCV south_L", np.hstack((img4,sobelFilterXAndYAxisOpenCV(img4,7))))

#######################################################################################

# # Gaussian Blurs OpenCV
#     # Dog
# cv2.imshow("Gaussian Blur 3x3 Dog", np.hstack((img1,gaussianBlurOpenCV(img1,3))))
# cv2.imshow("Gaussian Blur 7x7 Dog", np.hstack((img1,gaussianBlurOpenCV(img1,7))))
#
#     # Bicycle
# cv2.imshow("Gaussian Blur 3x3 Bicycle", np.hstack((img2,gaussianBlurOpenCV(img2,3))))
# cv2.imshow("Gaussian Blur 7x7 Bicycle", np.hstack((img2,gaussianBlurOpenCV(img2,7))))
#
#     # edge_L
# cv2.imshow("Gaussian Blur 3x3 edge_L", np.hstack((img3,gaussianBlurOpenCV(img3,3))))
# cv2.imshow("Gaussian Blur 7x7 edge_L", np.hstack((img3,gaussianBlurOpenCV(img3,7))))
#
#     # south_L
# cv2.imshow("Gaussian Blur 3x3 south_L", np.hstack((img4,gaussianBlurOpenCV(img4,3))))
# cv2.imshow("Gaussian Blur 7x7 south_L", np.hstack((img4,gaussianBlurOpenCV(img4,7))))

#######################################################################################

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
