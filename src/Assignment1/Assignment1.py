import cv2 as cv
import numpy as np

# Global constants
default_value = 100  # Must be an integer, for sliders() default value to scale off of
new_preview = None

# Function to update the displayed images based on slider values
def update_images(*args):
  global new_preview  # global so it can save the previewed image when saved in main

  # Get the current slider values
  contrast = cv.getTrackbarPos('Contrast', 'Sliders') / default_value  # contrast multiplies
  brightness = cv.getTrackbarPos('Brightness', 'Sliders') - default_value  # brightness adds

  # Update the right side of the concatenated result
  new_preview = cv.convertScaleAbs(preview, alpha=contrast, beta=brightness)

  # Display the updated result
  cv.imshow('Preview', new_preview)

  # Calculate histogram for the preview image
  create_histogram(new_preview, 'Preview Histogram', 410, 410)
  
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

def start():
  # Load an image
  image = cv.imread('dog.bmp')
  global preview
  preview = image # Preview initially matches the image

  # Create windows for each image
  cv.namedWindow('Image')
  cv.moveWindow('Image', 0, 0)

  cv.namedWindow('Preview')
  cv.moveWindow('Preview', 410, 0)

  # Display the initial images
  cv.imshow('Image', image)
  cv.imshow('Preview', preview)

  # Create a separate window for sliders
  cv.namedWindow('Sliders', cv.WINDOW_NORMAL)
  cv.moveWindow('Sliders', 820, 0)

  # Create sliders for adjusting contrast and brightness
  cv.createTrackbar('Contrast', 'Sliders', default_value, default_value * 2, update_images)
  cv.createTrackbar('Brightness', 'Sliders', default_value, default_value * 2, update_images)

  # Create histograms to be displayed
  create_histogram(image, 'Image Histogram', 0, 410)
  create_histogram(preview, 'Preview Histogram', 410, 410) # this is updated on each slider call

  # Wait for a key press to save and close the windows
  key = cv.waitKey(0)

  # Check if the pressed key is 's'
  if key == ord('s'):
    # Save the preview image to the working directory
    cv.imwrite('dog.bmp', new_preview)

  cv.destroyAllWindows()
