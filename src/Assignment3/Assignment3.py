import cv2 as cv
import numpy as np

def create_histogram(cv_image, title, x, y):
  # Separate R G and B
  bgr_planes_image = cv.split(cv_image) 

  histSize = 256
  hist_range = (0, histSize) # the upper boundary is exclusive
  accumulate = False
  b_hist_image = cv.calcHist(bgr_planes_image, [0], None, [histSize], hist_range, accumulate=accumulate)
  g_hist_image = cv.calcHist(bgr_planes_image, [1], None, [histSize], hist_range, accumulate=accumulate)
  r_hist_image = cv.calcHist(bgr_planes_image, [2], None, [histSize], hist_range, accumulate=accumulate)

  hist_w = 410 # histogram width
  hist_h = 360 # histogram height
  bin_w = int(round( hist_w/histSize )) # bin width
  hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8) # 3 for r-g-b

  # Normalize
  cv.normalize(b_hist_image, b_hist_image, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
  cv.normalize(g_hist_image, g_hist_image, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
  cv.normalize(r_hist_image, r_hist_image, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

  # Create lines for each r-g-b
  for i in range(1, histSize):
    cv.line(hist_image, ( bin_w*(i-1), hist_h - int(b_hist_image[i-1]) ),
      ( bin_w*(i), hist_h - int(b_hist_image[i]) ),
      ( 255, 0, 0), thickness=2)
    cv.line(hist_image, ( bin_w*(i-1), hist_h - int(g_hist_image[i-1]) ),
      ( bin_w*(i), hist_h - int(g_hist_image[i]) ),
      ( 0, 255, 0), thickness=2)
    cv.line(hist_image, ( bin_w*(i-1), hist_h - int(r_hist_image[i-1]) ),
      ( bin_w*(i), hist_h - int(r_hist_image[i]) ),
      ( 0, 0, 255), thickness=2)

  # Create and move histograms
  cv.imshow(title, hist_image)
  cv.moveWindow(title, x, y)

def otsu_multiple_classes(image, num_classes):
  # Reshape the image into a 1D array
  reshaped_image = image.reshape((-1, 1))
  reshaped_image = np.float32(reshaped_image)

  # Define the criteria, number of clusters (classes), and apply KMeans
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  _, labels, centers = cv.kmeans(reshaped_image, num_classes, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

  # Convert back to 8 bit values
  centers = np.uint8(centers)

  # Flatten the labels array
  labels = labels.flatten()

  # Create the segmented image
  segmented_image = centers[labels.flatten()]

  # Reshape the segmented image back to the original shape
  segmented_image = segmented_image.reshape(image_gray.shape)

  return segmented_image

  # # Convert to grayscale
  # image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  # # Apply gaussian blur
  # image_blur = cv.GaussianBlur(image_gray, (5,5), 0)

  # # Apply Otsu's thresholding
  # ret, image_otsu = cv.threshold(image_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

  # # Apply k-means clustering
  # Z = image_otsu.reshape((-1,3))
  # Z = np.float32(Z)
  # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  # K = num_classes
  # ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

  # # Convert back to uint8
  # center = np.uint8(center)
  # res = center[label.flatten()]
  # image_kmeans = res.reshape((image.shape))

  #return image_kmeans

# Load Image
image = cv.imread('a3img2.png')
image = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR) #upscale by 2x factor
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Create Windows
# Set dimensions for window separation
#all images should be 150x150
height, width, channel = image.shape # Get height/width quickly for window size and offset
height=height+30 # Add 28 pixels of height for the window tab size

# Original
cv.namedWindow('ImgOriginal')
cv.moveWindow('ImgOriginal', width*0, height*0)

# Mean Shift
cv.namedWindow('ImgMeanShift')
cv.moveWindow('ImgMeanShift', width*1, height*0)

#Otsu with 2 classes (binarization)
cv.namedWindow('ImgOtsu2')
cv.moveWindow('ImgOtsu2', width*2, height*0)

#Otsu with 2+ classes
#Determine ideal number of classes
create_histogram(image, 'Image Histogram', 0, 410)

cv.namedWindow('ImgOtsu3')
cv.moveWindow('ImgOtsu3', width*3, height*0)

cv.namedWindow('ImgOtsu4')
cv.moveWindow('ImgOtsu4', width*4, height*0)





# Apply segmentation
# Apply mean shift algorithm
mean_shift = cv.pyrMeanShiftFiltering(image, 10, 30, 2)

# Apply otsu binarization
# Blur first
gaussian_blur = cv.GaussianBlur(image_gray,(5,5),0)

#2 classes
thresh,otsu2 = cv.threshold(gaussian_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#2+ classes
otsu3 = otsu_multiple_classes(gaussian_blur, 3)
otsu4 = otsu_multiple_classes(gaussian_blur, 4)





# Show Images
# Img1Original
cv.imshow('ImgOriginal', image)

# Img1MeanShift
cv.imshow('ImgMeanShift', mean_shift)

# Img1Otsu2
cv.imshow('ImgOtsu2', otsu2)

# Img1Otsu3
cv.imshow('ImgOtsu3', otsu3)

# Img1Otsu4
cv.imshow('ImgOtsu4', otsu4)


# Wait to close
cv.waitKey(0)
cv.destroyAllWindows()

#https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0

#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html