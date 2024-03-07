import cv2 as cv
import numpy as np

# Load Image
image = cv.imread('bicycle.bmp')

def create_kernel(kernel_diameter):
  # Confirm diameter is a positive odd integer
  if (kernel_diameter%2!=1 or kernel_diameter<=0):
    raise ValueError('create_kernel() kernel_diameter must be a positive odd integer')
  
  return [[1] * kernel_diameter for _ in range(kernel_diameter)] #diameter of 3 = 3x3 kernel

# Convolution manually written
def convolution(image, kernel):
  kernel_diameter = len(kernel[0]) # get diameter of kernel
  kernel_radius = (kernel_diameter-1)//2 #i.e. 3x3 has radius of 1, the center pixel is not counted as part of the radius. 
  #// instead of / to ensure its outputted as an integer type

  # Get the dimensions of the image
  height, width, channels = image.shape

  # Set the size of new image, will have kernel_radius black pixels on each side
  average = np.zeros((height, width, channels), dtype=np.uint8) # dtype=np.uint8 seems to be necessary otherwise the image gets turned into intermittent pink pixels

  # Iterate through each pixel
  for y in range(height): 
    for x in range(width):
      for c in range(channels):
        if x<=kernel_radius or y<=kernel_radius or x>=(width-kernel_radius) or y>=(height-kernel_radius): #if in the border where kernel can't touch, fill as black
          average[y, x, c] = 0
          continue
        window = image[y-kernel_radius:y+kernel_radius+1, x-kernel_radius:x+kernel_radius+1, c]
        average[y, x, c] = np.sum(window * kernel)/kernel_diameter**2
  
  return average

# Apply a threshold to an image by outputting white if it does not meet the threshold, 
# black if it does. Threshold must range from [0-1]
def apply_threshold(image, threshold):
  #threshold refers to the % distance from min towards max, from 0 to 100% [0, 1]
  #threshold of .5 refers to the midpoint between min and max
  #threshold of 1 refers to the max value
  #threshold of 0 refers to the min value
  
  if threshold<0 or threshold>1:
    raise ValueError('apply_threshold() threshold must be in range [0,1]')
  
  # Get the dimensions of the image
  height, width, channels = image.shape

  # Set the size of new image, will have kernel_radius black pixels on each side
  thresholded_image = np.zeros((height, width, channels), dtype=np.uint8) # dtype=np.uint8 seems to be necessary otherwise the image gets turned into intermitten pink pixels
  channel_weights = [.144, .587, .299] #bgr 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue

  # Determine min/max for normalization
  min = np.inf
  max = -np.inf
  for y in range(height): 
    for x in range(width):
      for c in range(channels):
        if image[y, x, c]<min: #new min
          min = image[y, x, c] 
        elif image[y, x, c]>max: #new max
          max = image[y, x, c]

  # Set pixels to either 0 or 255 if it meets the threshold
  for y in range(height): 
    for x in range(width):
      # average value between channels
      sum = 0
      for c in range(channels):
        sum = sum + image[y, x, c]*channel_weights[c] #*channel_weights to convert to grayscale
      average = sum//3 #round down to stay as integer
      if average>(max-min)*threshold+min: #if the weighted average of all 3 channels meets the threshold
        thresholded_image[y, x, 0] = 0 #black if >thresh
        thresholded_image[y, x, 1] = 0 #black if >thresh
        thresholded_image[y, x, 2] = 0 #black if >thresh
      else:
        thresholded_image[y, x, 0] = 255 #white if <thresh
        thresholded_image[y, x, 1] = 255 #white if <thresh
        thresholded_image[y, x, 2] = 255 #white if <thresh
        
  return thresholded_image


# Create Windows
# Set dimensions for window separation
height, width, channel = image.shape # Get height/width quickly for window size and offset
height=height+30 # Add 28 pixels of height for the window tab size

# Original
cv.namedWindow('Original')
cv.moveWindow('Original', width*2, height*1)

# Box Filter
cv.namedWindow('BoxFilter')
cv.moveWindow('BoxFilter', width*0, height*0)

# Box Filter Manual
cv.namedWindow('BoxFilterManual')
cv.moveWindow('BoxFilterManual', width*0, height*1)

# Gaussian
cv.namedWindow('Gaussian')
cv.moveWindow('Gaussian', width*0, height*2)

# Sobel
cv.namedWindow('Sobelmanualxaxis')
cv.moveWindow('Sobelmanualxaxis', width*1, height*0)
cv.namedWindow('Sobelmanualyaxis')
cv.moveWindow('Sobelmanualyaxis', width*1, height*1)
cv.namedWindow('Sobelmanualxyaxis')
cv.moveWindow('Sobelmanualxyaxis', width*1, height*2)
cv.namedWindow('Sobelxyaxis')
cv.moveWindow('Sobelxyaxis', width*2, height*2)



# Create Filtered images
kernel_diameter = 5
# Original
image = image #no effect

# Box Filter
box = cv.boxFilter(image, -1, (kernel_diameter,kernel_diameter))

# Box Filter Manual
boxManual = convolution(image, create_kernel(kernel_diameter))

# Gauss
gauss = cv.GaussianBlur(image,(kernel_diameter,kernel_diameter),0)

# Sobel
# Set kernel and threshold
if kernel_diameter==3:
  vert_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] #vertical edge detector (product of deriv and gaussian filters)
  threshold = .2 #halved for xyaxis
elif kernel_diameter==5:
  vert_kernel = [[1,2,0,-2,-1],
                 [4,8,0,-8,-4],
                 [6,12,0,-12,-6],
                 [4,8,0,-8,-4],
                 [1,2,0,-2,-1]]
  threshold = .15 #halved for xyaxis
else:
  raise ValueError('Sobel manual kernel_diameter must be 3 or 5')
horiz_kernel = np.transpose(vert_kernel) #horizontal edge detector

sobelmanual_xaxis = apply_threshold(convolution(image, vert_kernel), threshold)
sobelmanual_yaxis = apply_threshold(convolution(image, horiz_kernel), threshold)
sobelmanual_xyaxis = apply_threshold((convolution(image, vert_kernel)**2 + convolution(image, horiz_kernel)**2)**.5, threshold)  #sqrt(x^2+y^2)

sobel_xyaxis = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert from bgr to grayscale automatically
sobel_xyaxis = cv.GaussianBlur(sobel_xyaxis,(kernel_diameter,kernel_diameter), sigmaX=0, sigmaY=0)
sobel_xyaxis = cv.Sobel(src=sobel_xyaxis, ddepth=cv.CV_64F, dx=1, dy=1, ksize=kernel_diameter) 


# Show Images
# Original
cv.imshow('Original', image)

# BoxFilter
cv.imshow('BoxFilter', box)
cv.imshow('BoxFilterManual', boxManual)

# Gaussian
cv.imshow('Gaussian', gauss)

# Sobel
cv.imshow('Sobelmanualxaxis', sobelmanual_xaxis)
cv.imshow('Sobelmanualyaxis', sobelmanual_yaxis)
cv.imshow('Sobelmanualxyaxis', sobelmanual_xyaxis)
cv.imshow('Sobelxyaxis', sobel_xyaxis)

# Wait to close
cv.waitKey(0)
cv.destroyAllWindows()

# https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
# https://indianaiproduction.com/blur-image-using-cv2-blur-cv2-boxfilter-opencv-python/
# https://learnopencv.com/edge-detection-using-opencv/#sobel-edge