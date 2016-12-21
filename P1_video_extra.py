# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import process_image as PI

extra_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
yellow_clip = clip2.fl_image(PI.process_image)
yellow_clip.write_videofile(extra_output, audio=False)
