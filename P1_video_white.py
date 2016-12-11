# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import process_image as PI


white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(PI.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
# %time white_clip.write_videofile(white_output, audio=False)
