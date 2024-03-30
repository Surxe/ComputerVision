from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

gif_file = r"images\circles2.gif"
gif = cv.VideoCapture(gif_file)
ret, image = gif.read()
imageGray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

hist = cv.calcHist([image],[0],None,[255],[0,256])

gaussianImage = cv.GaussianBlur(imageGray, ksize=(5,5), sigmaX=0, sigmaY=0)
threshold,segmentedImage = cv.threshold(gaussianImage, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cannyImage = cv.Canny(segmentedImage, 100, 200, 5)

cv.imshow("image",image)
cv.imshow("image2",gaussianImage)
cv.imshow("image3",segmentedImage)
cv.imshow("image4",cannyImage)
plt.show()
# cv.waitKey(0) 

print("continue")

fig, axs = plt.subplots(1,1, figsize=(10,5))

axs.imshow(image[:,:,::-1])

def detectCircles(img,threshold,region,radius = None):
    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X = [x-m+R_max,x+m+R_max]                                           #Computing the extreme X values
            Y= [y-n+R_max,y+n+R_max]                                            #Computing the extreme Y values
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    return B[:,R_max:-R_max,R_max:-R_max]

def displayCircles(A):
    # ret, img = gif.read()
    # fig, axs = plt.subplots(1,1, fig)
    # plt.imshow(image)
    circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        axs.add_patch(circle[-1])
    plt.show()


detectedCircles = detectCircles(cannyImage, 13, 15, radius=[100,10])
displayCircles(detectedCircles)

# circle=plt.Circle((50,50), 10,color=(1,0,0), fill = False)

# axs[1].imshow(gaussianImage,"gray")
# axs[2].imshow(segmentedImage,"gray")
# axs[3].
    

# axs.add_patch(circle)

# cv.imshow("image",image)
# cv.imshow("image2",gaussianImage)
# cv.imshow("image3",segmentedImage)
# cv.imshow("image3",cannyImage)
# plt.show()
# cv.waitKey(0) 