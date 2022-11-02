import numpy as np
import cv2
import matplotlib.pyplot as plt
#import webcolors
import matplotlib.colors as mc
import time
from PIL import Image, ImageOps
#from keras.models import load_model
import pyzbar.pyzbar as pyzbar
# import tensorflow.python
from os import listdir
from os.path import isfile, join
from os.path import exists
import os
import sys
import argparse
import pytesseract
from djitellopy import Tello
from DetectShapes.detect_shapes_OpenCV import DetectUtils
from DetectShapes.detect_shapes_OpenCV import DetectScreenInFrame
import logging
import threading
from Main_Tello_Simon import TelloControl
import Communication
import tkinter
import matplotlib


dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\second_try\second_try4"
filename = "frame_0_Screen detected, screen mode is 1" + ".jpg"

# dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\first_try\first_try6"
# filename = "frame_0_Screen detected, screen mode is 2" + ".jpg"

# dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\DetectShapes\pics_from_1658831841451"
# filename = "frame383" + ".jpg"


dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\second_try\second_try11"
filename = "frame_3_Screen detected, screen mode is 2" + ".jpg"


full_path = os.path.join(dirpath, filename)
img = cv2.imread(full_path)
cv2.imshow("photo", img)  # optional display photo
cv2.waitKey(0)
debugTello = TelloControl(dirpath=dirpath, debug=True)
debugTello.ImageProcessing.original_image = img
# PlotCv2ImageWithPlt(self.ImageProcessing.screen, 'screen to analyze')
debugTello.ImageProcessing.DetectScreen(True)
#debugTello.FixPositionByScreenBB()
if debugTello.ImageProcessing.screen is not None:
    flag, value = debugTello.AnalyzePicture()
    # print(value)
    # fin_data = debugTello.Str2IntList(value)
    # fin_data.insert(0, 3)
    # print(fin_data)
    # print(flag)

