import cv2 as cv
import numpy as np
import os
import json

global root_path
root_path = "src/Assignment4/"

# Shortcut to namedWindow, moveWindow, imshow
def createWindow(full_window_name, img, x, y):
    cv.namedWindow(full_window_name)
    cv.moveWindow(full_window_name, x, y)
    cv.imshow(full_window_name, img)

# def detect_circles_manual(image, dp=1, minDist=20, canny_param1=30, canny_param2=50, votes_threshold=2, minRadius=0, maxRadius=0):
#     """
#     Detect circles in a grayscale image using Hough Transform (manual implementation).

#     Args:
#         image: Grayscale input image.
#         dp: Inverse ratio of the accumulator resolution to the image resolution (default: 1).
#         minDist: Minimum distance between the centers of the detected circles (default: 20).
#         param1: Canny first method-specific parameter (default: 30).
#         param2: Canny second method-specific parameter (default: 50).
#         minRadius: Minimum circle radius (default: 0).
#         maxRadius: Maximum circle radius (default: 0).

#     Returns:
#         circles: A list of detected circles in the format (x, y, radius).
#     """
#     # Apply edge detection on the image
#     edges = cv.Canny(image, canny_param1, canny_param2)

#     # Apply Hough Transform to detect circles
#     circles = []
#     rows, cols = image.shape
#     r_min, r_max = minRadius, maxRadius
#     accumulator = np.zeros(image.shape)

#     for y in range(rows):
#         for x in range(cols):
#             if edges[y, x] != 0:
#                 for r in range(r_min, r_max + 1):
#                     for t in range(360):
#                         a = int(x - r * np.cos(np.radians(t)))
#                         b = int(y - r * np.sin(np.radians(t)))
#                         if 0 <= a < cols and 0 <= b < rows:
#                             if edges[b, a] == 0:
#                                 break

#                             accumulator[b, a] += 1
#                         if accumulator[b, a] > votes_threshold: #must meet threshold
#                             circles.append((b, a, r, accumulator[b, a]))
                    

    

    
#     # Filter circles based on minDist
#     # filtered_circles = []
#     # for circle in circles:
#     #     y, x, r, votes = circle

#     #     is_valid = True
#     #     for existing_circle in filtered_circles:
#     #         x0, y0, _, _ = existing_circle
#     #         if ((x - x0) ** 2 + (y - y0) ** 2) ** .5 < minDist:
#     #             is_valid = False
#     #             break
#     #     if is_valid:
#     #         filtered_circles.append(circle)

#     # Convert to the format (x, y, radius)
#     #filtered_circles = [(x, y, r) for (x, y, r, votes) in filtered_circles]

#     return circles


def detect_circles_manual2(image, minDist=20, canny_param1=30, canny_param2=50, votes_threshold=2, minRadius=0, maxRadius=0):
    # Apply Canny edge detection
    edges = cv.Canny(image, canny_param1, canny_param2)
    
    # Initialize an accumulator array to store votes for circle centers
    accumulator = np.zeros(image.shape[:2], dtype=np.uint16)
    
    # Iterate through each pixel in the edge image
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            # If this is an edge pixel
            if edges[y, x] > 0:
                # Calculate gradient direction
                gradient_dir = np.arctan2(y, x)
                
                # Calculate possible circle centers
                for r in range(minRadius, maxRadius):
                    a = int(x - r * np.cos(gradient_dir))
                    b = int(y - r * np.sin(gradient_dir))
                    
                    # Ensure the center is within the image bounds
                    if a >= 0 and a < image.shape[1] and b >= 0 and b < image.shape[0]:
                        accumulator[b, a] += 1  # Vote for this circle center
    
    # Find circles with enough votes
    circles = []
    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            if accumulator[y, x] >= votes_threshold:
                if all(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) >= minDist for xc, yc, _ in circles):
                    circles.append((x, y, minRadius))
    
    return circles


#image_name: name of the image
#index: index of the image
def processImage(image_name, index):
    # Load image
    image_path = os.path.join(root_path,image_name)
    img = cv.imread(image_path)
    img = rescale(img, 0.65)
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
    imgcvHoughCircle = imgBlur.copy()
    circles = cv.HoughCircles(imgcvHoughCircle, cv.HOUGH_GRADIENT, 
                              dp=1, minDist=30, 
                              param1=200, param2=35, 
                              minRadius=10, maxRadius=55)
    
    if circles is not None:
        circles = np.uint16(np.around(circles)) #convert types
        
        #draw circles on image and store circle information in text
        circle_hash['Circles']['cv'] = {}
        i=0 #tracks current circle
        for circle in circles[0,:]:
            i+=1
            circle_hash['Circles']['cv'][str(i)] = {}
            circle_hash['Circles']['cv'][str(i)]["x"] = str(circle[0])
            circle_hash['Circles']['cv'][str(i)]["y"] = str(circle[1])
            circle_hash['Circles']['cv'][str(i)]["radius"] = str(circle[2])
                
            cv.circle(imgcvHoughCircle, (circle[0], circle[1]), circle[2], (255,255,255), 3)

    application_names.append('cvHoughCircle')
    application_images.append(imgcvHoughCircle)

    # manualHoughCircle
    # imgManualHoughCircle = imgBlur.copy()
    # circles = detect_circles_manual2(imgManualHoughCircle, 
    #                                 minDist=30, 
    #                                 canny_param1=canny_param1, canny_param2=canny_param2, 
    #                                 votes_threshold=3,
    #                                 minRadius=15, maxRadius=65)
    # if circles:
    #     circle_hash['Circles']['manual'] = {}
    #     i=0
        
    #     for (x,y,r) in circles:
    #         i+=1
    #         circle_hash['Circles']['manual'][str(i)] = {}
    #         circle_hash['Circles']['manual'][str(i)]["x"] = x
    #         circle_hash['Circles']['manual'][str(i)]["y"] = y
    #         circle_hash['Circles']['manual'][str(i)]["radius"] = r
    #         #circle_hash['Circles']['manual'][str(i)]["votes"] = votes

    #         cv.circle(imgManualHoughCircle, (x, y), r, (255,255,255), 3)

    # print(circle_hash)

    # application_names.append('manualHoughCircle')
    # application_images.append(imgManualHoughCircle)

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
    image_name_indexless = "circles.png" 
    for i in range(1,4):
        image_name = image_name_indexless.replace(".",str(i)+".") #circles.png to circles1.png
        processImage(image_name, i)

    # Wait to close
    cv.waitKey(0)
    cv.destroyAllWindows()

#https://docs.opencv.org/4.x/d3/de5/tutorial_js_houghcircles.html
    