import cv2 as cv
import numpy as np
import os
import json
import time
import math

global root_path
root_path = "src/Assignment4/"

# Shortcut to namedWindow, moveWindow, imshow
def createWindow(full_window_name, img, x, y):
    cv.namedWindow(full_window_name)
    cv.moveWindow(full_window_name, x, y)
    cv.imshow(full_window_name, img)

# Function to find local maximum within each window
def find_local_maxima(accumulator, window_radius=2):
    if window_radius%2!=0:
        raise ValueError("Window radius must be an even number")
    h, w, d = accumulator.shape
    window_diameter = window_radius*2+1  # Half of the window size #window diameter is window_radius*2+1
    local_maxima = np.zeros_like(accumulator)

    for y in range(window_radius, h - window_radius):
        for x in range(window_radius, w - window_radius):
            for r in range(d):
                if y - window_radius < 0 or y + window_radius >= h or x - window_radius < 0 or x + window_radius >= w:
                    continue
                window = accumulator[y - window_radius:y + window_radius + 1,
                                    x - window_radius:x + window_radius + 1, 
                                    r]
                max_value = np.max(window)
                if max_value == 0:
                    continue
                if accumulator[y, x, r] == max_value:
                    local_maxima[y, x, r] = max_value
                else:
                    local_maxima[y, x, r] = 0

    return local_maxima

def detect_circles_manual(image, minDist=20, canny_param1=30, canny_param2=50, votes_threshold=2, minRadius=0, maxRadius=0):
    # Apply Canny edge detection
    edges = cv.Canny(image, canny_param1, canny_param2)
    
    y_dim = edges.shape[0]
    x_dim = edges.shape[1]

    # Initialize an accumulator array to store votes for circle centers
    accumulator = np.zeros((y_dim, x_dim, maxRadius))
    
    # Iterate through each pixel in the edge image
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            # If this is an edge pixel
            if edges[y, x] > 0:
                for t in range(0, 360):
                    # Calculate gradient direction
                    #gradient_dir = np.arctan2(y, x)
                    
                    # Calculate possible circle centers
                    for r in range(minRadius, maxRadius):
                        a = int(x - r * np.cos(t))
                        b = int(y - r * np.sin(t))
                        
                        # Ensure the center is within the image bounds
                        if a >= 0 and a < x_dim and b >= 0 and b < y_dim:
                            accumulator[b, a, r] += 1  # Vote for this circle center

    #accumulator = find_local_maxima(accumulator)
    accumulator_thresholded = np.zeros_like(accumulator)
    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            for r in range(accumulator.shape[2]):
                if accumulator[y, x, r] >= votes_threshold:
                    accumulator_thresholded[y, x, r] = accumulator[y, x, r]
    
                            
    accumulator_localized = find_local_maxima(accumulator_thresholded, window_radius=10)

    circles = []
    for y in range(accumulator_localized.shape[0]):
        for x in range(accumulator_localized.shape[1]):
            for r in range(accumulator_localized.shape[2]):
                if accumulator_localized[y, x, r] >= votes_threshold:
                    if all(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) >= minDist for xc, yc, _, _ in circles):
                        circles.append((x, y, r, accumulator_localized[y, x, r]))

    return circles


#image_name: name of the image
#index: index of the image
def processImage(image_name, index):
    # Load image
    image_path = os.path.join(root_path,image_name)
    img = cv.imread(image_path)
    img = rescale(img, .65)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #ensure its grayscale

    # Get the height and width of the image
    height, width = img.shape[:2]
    height=height+30 #add 30 pixels for height of white window tab

    # Create window variables
    window_name = 'Image'+str(index)
    application_names = []
    application_images = []

    # Original
    application_names.append('Original')
    application_images.append(img)

    # Blur 
    imgBlur = cv.medianBlur(img, 7)

    # Canny
    canny_param1=255
    canny_param2=254
    imgCanny = cv.Canny(imgBlur, canny_param1, canny_param2)
    application_names.append('Canny')
    application_images.append(imgCanny)

    #Circles
    circle_hash = {}
    circle_hash['Circles'] = {}

    # cvHoughCircle
    start_time = time.time()
    imgcvHoughCircle = imgBlur.copy()
    circles = cv.HoughCircles(imgcvHoughCircle, cv.HOUGH_GRADIENT, 
                              dp=1, minDist=30, 
                              param1=200, param2=35, 
                              minRadius=10, maxRadius=55)
    end_time = time.time()
    print("Image", index, "Time taken for cvHoughCircle:", end_time-start_time)
    
    if circles is not None:
        circles = np.uint16(np.around(circles)) #convert types
        
        #draw circles on image and store circle information in text
        circle_hash['Circles']['cv'] = {}
        i=0 #tracks current circle
        for circle in circles[0,:]:
            i+=1
            circle_hash['Circles']['cv'][str(i)] = {}
            circle_hash['Circles']['cv'][str(i)]["x"] = int(circle[0])
            circle_hash['Circles']['cv'][str(i)]["y"] = int(circle[1])
            circle_hash['Circles']['cv'][str(i)]["radius"] = int(circle[2])
                
            cv.circle(imgcvHoughCircle, (circle[0], circle[1]), circle[2], (255,255,255), 3)

    application_names.append('cvHoughCircle')
    application_images.append(imgcvHoughCircle)

    #manualHoughCircle
    start_time = time.time()
    imgManualHoughCircle = cv.cvtColor(imgBlur, cv.COLOR_GRAY2RGB)
    circles = detect_circles_manual(imgBlur.copy(), 
                                    minDist=30, 
                                    canny_param1=canny_param1, canny_param2=canny_param2, 
                                    votes_threshold=105,
                                    minRadius=15, maxRadius=65)
    
    circle_hash['Circles']['manual'] = {}
    i=0
    
    for (x,y,r,votes) in circles:
        i+=1
        circle_hash['Circles']['manual'][str(i)] = {}
        circle_hash['Circles']['manual'][str(i)]["x"] = int(x)
        circle_hash['Circles']['manual'][str(i)]["y"] = int(y)
        circle_hash['Circles']['manual'][str(i)]["radius"] = int(r)
        circle_hash['Circles']['manual'][str(i)]["votes"] = int(votes)

        cv.circle(imgManualHoughCircle, (x, y), r, (255,0,0), 3)
        cv.circle(imgManualHoughCircle, (x, y), int(math.cbrt(int(votes))), (0, 0, 255), int(math.cbrt(int(votes))))

    end_time = time.time()
    print("Image", index, "Time taken for manualHoughCircle:", end_time-start_time)
    application_names.append('manualHoughCircle')
    application_images.append(imgManualHoughCircle)

    #write circle information to file
    output_file = os.path.join(root_path, image_name.replace(".png","_output.txt"))
    with open(output_file, 'w') as file:
        json_str = json.dumps(circle_hash, indent=4)
        file.write(json_str)

    # Create windows for each application of the image
    for application_index in range(len(application_names)):
        createWindow(window_name+application_names[application_index], application_images[application_index], width*application_index, height*(index-1))

# Rescale image, shortcut to cv.resize()
def rescale(img, scale=1):
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

if __name__=="__main__":  
    print("Assignment 4: Hough Circle Transform\nProgram initiating...")
    image_name_indexless = "circles.png" 
    for i in range(1,4):
        image_name = image_name_indexless.replace(".",str(i)+".") #circles.png to circles1.png
        print('Processing', image_name)
        processImage(image_name, i)

    print("Program complete")

    # Wait to close
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Program terminated")

#https://docs.opencv.org/4.x/d3/de5/tutorial_js_houghcircles.html
#https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    


    