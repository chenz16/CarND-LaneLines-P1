
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import P1Fcns as P1

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    gray = P1.grayscale(image)
    kernel_size = 5
    blur_gray = P1.gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 30
    high_threshold = 150
    edges = P1.canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(100,imshape[0]),(440, 330), (560, 330), (950,imshape[0])]], dtype=np.int32)

    # Next we'll create a masked edges image using cv2.fillPoly(
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
    line_image2 = P1.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines_origin2 = P1.weighted_img(image,line_image2)

    return lines_origin2
