import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import P1Fcns as P1
import numpy as np

AllPics = os.listdir("test_images/")

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

for pic in AllPics:

    image = mpimg.imread('test_images/'+pic)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = P1.grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    blur_gray = P1.gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    # edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = P1.canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(100,imshape[0]),(440, 330), (560, 330), (950,imshape[0])]], dtype=np.int32)

    # Next we'll create a masked edges image using cv2.fillPoly()
    # mask = np.zeros_like(edges)
    # ignore_mask_color = 255
    # cv2.fillPoly(mask, vertices, ignore_mask_color)
    # masked_edges = cv2.bitwise_and(edges, mask)
    masked_edges = P1.region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 25    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 150   # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    '''lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            if np.absolute((x2-x1)/(y2-y1))>1.35:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Create a "color" binary image to combine with line image
    # color_edges = np.dstack((edges, edges, edges))
    # Draw the lines on the edge image
    # lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)'''

    line_image2 = P1.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # lines_origin = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    lines_origin2 = P1.weighted_img(image,line_image2)
    plt.imshow(lines_origin2)
    plt.show()
