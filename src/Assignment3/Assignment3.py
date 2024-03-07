import cv2 as cv
import numpy as np
from skimage import data
from skimage.filters import threshold_multiotsu

global base_path 
base_path = "src/Assignment3/"

def create_histogram(cv_image, title, x, y, histSize):
  # Separate R G and B
  bgr_planes_image = cv.split(cv_image) 

  hist_range = (0, histSize) # the upper boundary is exclusive
  accumulate = False
  b_hist_image = cv.calcHist(bgr_planes_image, [0], None, [histSize], hist_range, accumulate=accumulate)
  g_hist_image = cv.calcHist(bgr_planes_image, [1], None, [histSize], hist_range, accumulate=accumulate)
  r_hist_image = cv.calcHist(bgr_planes_image, [2], None, [histSize], hist_range, accumulate=accumulate)

  hist_w = histSize # histogram width
  hist_h = histSize # histogram height + window tab size
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

image_num = [1,2,3,4]
for num in image_num:
  # Load Image
  image = cv.imread(base_path+'a3img'+str(num)+'.png')

  image = cv.resize(image, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR) #upscale by 1.5x factor
  image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  # Create Windows
  # Set dimensions for window separation
  #all images should be 150x150
  height, width, channel = image.shape # Get height/width quickly for window size and offset
  height=height+30 # Add 28 pixels of height for the window tab size

  # Original
  cv.namedWindow('Img'+str(num)+'Original')
  cv.moveWindow('Img'+str(num)+'Original', width*0, height*(num-1))

  # Mean Shift
  cv.namedWindow('Img'+str(num)+'MeanShift')
  cv.moveWindow('Img'+str(num)+'MeanShift', width*1, height*(num-1))

  #Otsu with 2 classes (binarization)
  cv.namedWindow('Img'+str(num)+'Otsu2')
  cv.moveWindow('Img'+str(num)+'Otsu2', width*2, height*(num-1))

  #Otsu with 2+ classes
  #Determine ideal number of classes
  create_histogram(image, 'Image'+str(num)+' Histogram', width*3, height*(num-1), width) #create histogram of widthxheight equal to image width at x=0 y=410

  cv.namedWindow('Img'+str(num)+'Otsu3')
  cv.moveWindow('Img'+str(num)+'Otsu3', width*4, height*(num-1))

  cv.namedWindow('Img'+str(num)+'Otsu4')
  cv.moveWindow('Img'+str(num)+'Otsu4', width*5, height*(num-1))





  # Apply segmentation
  # Apply mean shift algorithm
  mean_shift = cv.pyrMeanShiftFiltering(image, 10, 30, 2)

  # Apply otsu binarization
  # Blur first
  gaussian_blur = cv.GaussianBlur(image_gray,(5,5),0)

  #2 classes
  thresh,otsu2 = cv.threshold(gaussian_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

  #2+ classes
  thresholds3 = threshold_multiotsu(gaussian_blur, classes=3)
  regions3 = np.digitize(gaussian_blur, bins=thresholds3)
  otsu3 = (regions3 / (len(thresholds3) - 1) * 255).astype(np.uint8)

  thresholds4 = threshold_multiotsu(gaussian_blur, classes=4)
  regions4 = np.digitize(gaussian_blur, bins=thresholds4)
  otsu4 = (regions4 / (len(thresholds4) - 1) * 255).astype(np.uint8) #regions4.astype(np.uint8) * 255 

  
  # Show Images
  # Img1Original
  cv.imshow('Img'+str(num)+'Original', image)

  # Img1MeanShift
  cv.imshow('Img'+str(num)+'MeanShift', mean_shift)

  # Img1Otsu2
  cv.imshow('Img'+str(num)+'Otsu2', otsu2)

  # Img1Otsu3
  cv.imshow('Img'+str(num)+'Otsu3', otsu3)

  # Img1Otsu4
  cv.imshow('Img'+str(num)+'Otsu4', otsu4)


# Wait to close
cv.waitKey(0)
cv.destroyAllWindows()

#https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html