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
import Communication


class TelloControl:

    def __init__(self, dirpath=None, debug=True):
        self.tello = Tello()
        self.dirpath = dirpath
        self.ImageProcessing = DetectScreenInFrame()
        self.timing = 0
        self.status = "on ground"
        self.debug = debug
        self.getcloservalue = 40  # in centimeters, how far get closer to the silver screen
        self.searchingstep = 20  # in centimeters ' how far to go to each side in searching for screen square
        self.same_data_times = 2  # how many times must acquire the same data to accept final results
        self.max_try = 6  # how many times read the picture
        self.takeoffaddvalue = 130  # how many centimeters to add to tello in order to be on screen height
        self.movementscale = 0.1  # movement scale for BB fixing
        self.sleep_time_sec = 10  # sleep time between status logging
        self.Server = None

    def logging_function(self, sleep_time_sec=10):
        start_time = time.time()
        logging.info(f"logging start time {start_time}")
        while True:
            time.sleep(sleep_time_sec)
            logging.info(f"logging start time {time.time() - start_time}, Tello Control Status is {self.status} ")

    def DetectScreenOnPosition(self):
        self.ImageProcessing.original_image = self.TakePicture()
        self.ImageProcessing.DetectScreen(False)
        if self.ImageProcessing.screenmode != 0:
            txt = "Screen detected, screen mode is " + str(self.ImageProcessing.screenmode)
            result = True
        else:
            txt = "Screen not detected, screen mode is " + str(self.ImageProcessing.screenmode)
            result = False
        if self.debug:
            print(txt)
            logging.info(txt)
            self.SavePicture(self.ImageProcessing.original_image, txt)
        return result

    def TakePicture(self):
        frame_read = self.tello.get_frame_read()
        return frame_read.frame

    def SavePicture(self, cv2_frame, text_on_picture):
        filename = "frame_" + str(self.timing)+"_" + str(text_on_picture) +".jpg"
        full_name = os.path.join(self.dirpath, filename)
        #logging.info(f"Saving Picture {filename}")
        # cv2_frame = cv2.putText(img=cv2_frame, text=text_on_picture, org=(250, 50), fontScale= 1,
        #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
        cv2.imwrite(full_name, cv2_frame)
        self.timing += 1  # Update timing for next Picture

    def Start(self, server = None):
        self.tello.connect()
        self.tello.streamon()
        #x = threading.Thread(target=self.logging_function(sleep_time_sec=self.sleep_time_sec), daemon=True)
        #x.start()
        self.tello.takeoff()
        #self.tello.move_up(int(30))
        self.TelloMove("up", int(self.takeoffaddvalue))
        self.status = "on air - starting position"
        self.server = server
        self.SendDataToRobot([Communication.MessageType.Tello_Up.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])   # Tello Up

    def End(self):
        self.tello.land()
        self.tello.end()

    def Rotate(self, direction, angledeg):
        angl = int(angledeg)
        if direction == "clockwise":
            self.tello.rotate_clockwise(angl)
        if direction == "counter_clockwise":
            self.tello.rotate_counter_clockwise(angl)

    def TelloMove(self, direction, displacment_cm):

        if 20 >= displacment_cm >= 10:
            displacment_cm = 20
        elif 10 >= displacment_cm:
            return

        disp_cm = int(displacment_cm)
        if direction == "forward":
            self.tello.move_forward(disp_cm)
        elif direction == "back":
            self.tello.move_back(disp_cm)
        elif direction == "up":
            self.tello.move_up(int(disp_cm))
        elif direction == "down":
            self.tello.move_down(disp_cm)
        elif direction == "right":
            self.tello.move_right(disp_cm)
        elif direction == "left":
            self.tello.move_left(disp_cm)

    def AnalyzeScreen(self):
        self.status = "on air - screen analyzing"
        datalist = []
        moved_close = False
        #self.FixPositionByScreenBB()  # puts screen in the center of a frame by its bounding box
        print("here was Fix Position By screen")
        if self.debug:
            print("Fixing position by screen")
            logging.info("Fixing position by screen")
        success, data = self.AnalyzePicture()
        if not success:  # found screen, and read what is on it
            self.TelloMove("forward", self.getcloservalue)
            moved_close = True
            if self.debug:
                print("moved close")
                logging.info("moved close")
            self.DetectScreenOnPosition()
            success, data = self.AnalyzePicture()
        if success:
            datalist.append(data)
            while True:  # needs to get the same data couple of times
                    self.DetectScreenOnPosition()  # takes another picture
                    success, data = self.AnalyzePicture()  # Analyze new picture
                    if success:
                        same_data = 1
                        for dataitem in datalist:
                            if dataitem == data:
                                same_data += 1
                        if same_data >= self.same_data_times:
                            return True, data, moved_close
                    datalist.append(data)
                    if len(datalist) > self.max_try:
                        if self.debug:
                            print(f"{self.max_try} different times cannot get the same result ")
                            logging.info(f"{self.max_try} different times cannot get the same result ")
                        return False, data, moved_close
        else:
            if self.debug:
                print("cannot recognize what is on screen")
                logging.info("cannot recognize what is on screen")
            return False, data, moved_close

    def AnalyzePicture(self):
        result = []
        shapes, shapes_color, shapes_bb, numbers = self.ImageProcessing.DetectfeaturesInSilverScreen()
        # IF Detected shapes or Text Boxes Or Qr code Box
        IsTextQrCode = True
        for idx, color in enumerate(shapes_color):  # Determine whether all shapes in screen are black and white
            if abs(int(color[0]) - int(color[1])) + abs(int(color[0]) - int(color[2]) + abs(int(color[1]) - int(color[2]))) > 40:
                IsTextQrCode = False
                break
        if IsTextQrCode:  # IF QR Code or Text
            if self.debug:
                print(f"The Picture Contains Qr Code or Text")
                logging.info(f"The Picture Contains Qr Code or Text")
            Qrflag, value = DetectUtils.read_qr_code(self.ImageProcessing.screen)
            if Qrflag:
                if self.debug:
                    print("QR Code screen")
                    print(f"output string is {value}")
                    logging.info(f"QR Code screen ,output string is {value} ")
                return True, value
            for i in range(len(shapes)):  # searching again for QR Code in shapes
                # name = f"shape {i} "
                # if print_shapes_by_order:
                #     DetectUtils.PlotCv2ImageWithPlt(shapes[i], name)
                flag, value = DetectUtils.read_qr_code(shapes[i])
                if flag:
                    if self.debug:
                        print("QR Code screen")
                        print(f"output string is {value}")
                        logging.info(f"QR Code screen ,output string is {value} ")
                    return True, value
            # did not succeeded in finding QR code - trying to get text
            value = self.ImageProcessing.DetectWordsInSilverScreen()
            if self.debug:
                print("Text screen")
                print(f"output string is {value}")
                logging.info(f"Text screen, output string is {value}")
            if len(value) > 0:
                full_word_list = [None]*9
                for i in range(len(value)):
                    full_word_list[i] = value[i]
                return True, full_word_list
            else:
                return False, value

        else:  # Colored Shapes
            ordered_shapes, ordered_shapes_color, ordered_shapes_bb = self.ImageProcessing.OrderShapes(shapes, shapes_color, shapes_bb)
            list_ordered_colors = DetectUtils.PrepareReturnList(ordered_shapes_color, outputflag=1)
            if self.debug:
                print(f"output string is {list_ordered_colors}")
                logging.info(f"output string is {list_ordered_colors}")
            result = list_ordered_colors

        if len(result) > 0:
            return True, result
        else:
            return False, result

    def AnalyzePictureOLD(self):
        result = []
        shapes, shapes_color, shapes_bb, numbers = self.ImageProcessing.DetectfeaturesInSilverScreen()
        # IF Detected shapes or Text Boxes Or Qr code Box
        IsTextQrCode = True
        for idx, color in enumerate(shapes_color):  # Determine whether all shapes in screen are black and white
            if abs(int(color[0]) - int(color[1])) + abs(int(color[0]) - int(color[2]) + abs(int(color[1]) - int(color[2]))) > 40:
                IsTextQrCode = False
        if IsTextQrCode:  # IF QR Code or Text
            if self.debug:
                print(f"The Picture Contains Qr Code or Text")
                logging.info(f"The Picture Contains Qr Code or Text")
            for i in range(len(shapes)):
                # name = f"shape {i} "
                # if print_shapes_by_order:
                #     DetectUtils.PlotCv2ImageWithPlt(shapes[i], name)
                flag, value = DetectUtils.read_qr_code(shapes[i])
                if flag:
                    if self.debug:
                        print("QR Code screen")
                        print(f"output string is {value}")
                        logging.info(f"QR Code screen ,output string is {value} ")
                    result = value
                    break
                else:
                    result = self.ImageProcessing.DetectWordsInSilverScreen()
                    if self.debug:
                        print("Text screen")
                        print(f"output string is {result}")
                        logging.info(f"Text screen, output string is {result}")
                    break
            if len(shapes) == 0:
                 result = self.ImageProcessing.DetectWordsInSilverScreen()
                 if self.debug:
                    print("Text screen")
                    print(f"output string is {result}")
                    logging.info(f"Text screen, output string is {result}")
        else:  # Colored Shapes
            ordered_shapes, ordered_shapes_color, ordered_shapes_bb = self.ImageProcessing.OrderShapes(shapes, shapes_color, shapes_bb)
            list_ordered_colors = DetectUtils.PrepareReturnList(ordered_shapes_color, outputflag=0)
            if self.debug:
                print(f"output string is {list_ordered_colors}")
                logging.info(f"output string is {list_ordered_colors}")
            result = list_ordered_colors

        if len(result) > 0:
            return True, result
        else:
            return False, result

    def SendDataToRobot(self, data):
        if self.server is not None:
            self.server.Message2Send = Communication.Message2str(data)
            if self.debug:
                print(f"sending data to a robot {data}")
                logging.info(f"Message to send is {Communication.Message2str(data)}")
        else:
            if self.debug:
                print(f"sending data to a robot {data}")
                logging.info(f"no communication, Message to send is {Communication.Message2str(data)}")

        return

    def FixPositionByScreenBB(self):
        #  self.ImageProcessing.screen  #
        #   self.ImageProcessing.bb  #  (xc, yc, w, h)
        # cropped_image = self.original_image[yc:yc + h, xc:xc + w]
        displacement_y = self.ImageProcessing.screen.shape[0]/2 - self.ImageProcessing.bb[1] - self.ImageProcessing.bb[3]/2
        displacement_x = self.ImageProcessing.screen.shape[1]/2 - self.ImageProcessing.bb[0] - self.ImageProcessing.bb[2]/2

        if self.debug:
            print(f"FixPositionByScreenBB displacement y={displacement_y} displacement x={displacement_x} pixels")
            logging.info(f"FixPositionByScreenBB displacement y={displacement_y} displacement x={displacement_x} pixels")
        if displacement_y ==0 and displacement_x == 0:
            return
        if displacement_y >= 0:
            if self.debug:
                print(f"moving up")
                logging.info(f"moving up")
            self.TelloMove("up", int(abs(displacement_y)*self.movementscale))
        else:
            if self.debug:
                print(f"moving down")
                logging.info(f"moving down")
            self.TelloMove("down", int(abs(displacement_y) * self.movementscale))
        if displacement_x >= 0:
            if self.debug:
                print(f"moving left")
                logging.info(f"moving left")
            self.TelloMove("left", int(abs(displacement_x)*self.movementscale/2))
        else:
            if self.debug:
                print(f"moving right")
                logging.info(f"moving right")
            self.TelloMove("right", int(abs(displacement_x) * self.movementscale/2))

    def FindAndReadScreen(self):
        # takes picture and searches for screen
        movement_to_find_screen = [['up', self.searchingstep], ["right", self.searchingstep],
                                   ["down", self.searchingstep], ["down", self.searchingstep],
                                   ["left", self.searchingstep], ["left", self.searchingstep],
                                   ["up", self.searchingstep], ["up", self.searchingstep]]  # movement procedure to find screen
        if self.status != "sending data to a robot":
            self.status = "searching for a screen"
        if self.debug:
            logging.info(f"status is {self.status}")
        movement_list = []
        moved_close = False
        FoundScreen = False
        movement_idx = 0
        while (not FoundScreen) and (movement_idx < len(movement_to_find_screen)):
            if self.DetectScreenOnPosition():
                FoundScreen = True
                if self.ImageProcessing.screenmode == 2:  # if silver screen
                    if self.status != "sending data to a robot":
                        success, final_data, moved_close = self.AnalyzeScreen()
                        if success:
                            self.status = "sending data to a robot"
                            self.SendDataToRobot(final_data)
                        else:
                            self.FixPositionByScreenBB()
                            success, final_data, moved_close = self.AnalyzeScreen()
                            if success:
                                self.status = "sending data to a robot"
                                fin_data = self.Str2IntList(final_data)
                                fin_data.insert(0, 3)
                                self.SendDataToRobot(fin_data)
                    else:
                        print("here was Fix Position")
                        #self.FixPositionByScreenBB()
                else:
                    self.status = f"found non silver screen"
                    if self.debug:
                        print(self.status)
                        logging.info(self.status)
                    print("here was Fix Position")
                    #self.FixPositionByScreenBB()
            else:  # trying to take another picture to be sure that there is no screen
                if self.DetectScreenOnPosition():
                    FoundScreen = True
                    if self.ImageProcessing.screenmode == 2:  # if silver screen
                        if self.status != "sending data to a robot":
                            success, final_data, moved_close = self.AnalyzeScreen()
                            if success:
                                self.status = "sending data to a robot"
                                fin_data = self.Str2IntList(final_data)
                                fin_data.insert(0, 3)
                                self.SendDataToRobot(fin_data)
                            else:
                                print("here was Fix Position")
                                #self.FixPositionByScreenBB()
                                success, final_data, moved_close = self.AnalyzeScreen()
                                if success:
                                    self.status = "sending data to a robot"
                                    fin_data = self.Str2IntList(final_data)
                                    fin_data.insert(0, 3)
                                    self.SendDataToRobot(fin_data)
                        else:
                            print("here was Fix Position")
                            # self.FixPositionByScreenBB()
                    else:
                        self.status = f"found non silver screen"
                        if self.debug:
                            print(self.status)
                            logging.info(self.status)
                        print("here was Fix Position")
                        # self.FixPositionByScreenBB()
                else:
                    if self.debug:
                        print(f"searching for screen: displacement {movement_to_find_screen[movement_idx][0]} - {movement_to_find_screen[movement_idx][1]}")
                        logging.info(f"searching for screen: displacement {movement_to_find_screen[movement_idx][0]} - {movement_to_find_screen[movement_idx][1]}")
                    self.TelloMove(movement_to_find_screen[movement_idx][0], movement_to_find_screen[movement_idx][1])
                    movement_idx += 1
                    movement_list.append(movement_to_find_screen[movement_idx])
        if moved_close:
            self.TelloMove("back", self.getcloservalue)
        if not FoundScreen:
            if self.debug:
                print(f"didn't find screen")
                logging.info(f"didn't find screen")
            self.status = "did not find screen"
        if len(movement_list) >= 1:
            self.ReturnToOriginalPosition(movement_list)

    def FindAndReadScreenOLD_With_FIXPOSITIONBYSCREEN(self):
        # takes picture and searches for screen
        movement_to_find_screen = [['up', self.searchingstep], ["right", self.searchingstep],
                                   ["down", self.searchingstep], ["down", self.searchingstep],
                                   ["left", self.searchingstep], ["left", self.searchingstep],
                                   ["up", self.searchingstep], ["up", self.searchingstep]]  # movement procedure to find screen
        if self.status != "sending data to a robot":
            self.status = "searching for a screen"
        if self.debug:
            logging.info(f"status is {self.status}")
        movement_list = []
        moved_close = False
        FoundScreen = False
        movement_idx = 0
        while (not FoundScreen) and (movement_idx < len(movement_to_find_screen)):
            if self.DetectScreenOnPosition():
                FoundScreen = True
                if self.ImageProcessing.screenmode == 2:  # if silver screen
                    if self.status != "sending data to a robot":
                        success, final_data, moved_close = self.AnalyzeScreen()
                        if success:
                            self.status = "sending data to a robot"
                            self.SendDataToRobot(final_data)
                        else:
                            self.FixPositionByScreenBB()
                            success, final_data, moved_close = self.AnalyzeScreen()
                            if success:
                                self.status = "sending data to a robot"
                                fin_data = self.Str2IntList(final_data)
                                fin_data.insert(0, 3)
                                self.SendDataToRobot(fin_data)
                    else:
                        self.FixPositionByScreenBB()
                else:
                    self.status = f"found non silver screen"
                    if self.debug:
                        print(self.status)
                        logging.info(self.status)
                    self.FixPositionByScreenBB()
            else:
                if self.debug:
                    print(f"searching for screen: displacement {movement_to_find_screen[movement_idx][0]} - {movement_to_find_screen[movement_idx][1]}")
                    logging.info(f"searching for screen: displacement {movement_to_find_screen[movement_idx][0]} - {movement_to_find_screen[movement_idx][1]}")
                self.TelloMove(movement_to_find_screen[movement_idx][0], movement_to_find_screen[movement_idx][1])
                movement_idx += 1
                movement_list.append(movement_to_find_screen[movement_idx])
        if moved_close:
            self.TelloMove("back", self.getcloservalue)
        if not FoundScreen:
            if self.debug:
                print(f"didn't find screen")
                logging.info(f"didn't find screen")
            self.status = "did not find screen"
        if len(movement_list) >= 1:
            self.ReturnToOriginalPosition(movement_list)

    def ReturnToOriginalPosition(self, movement_list):

        param = len(movement_list)

        if param == 1:
            self.TelloMove("down", self.searchingstep)
        if param == 2:
            self.TelloMove("left", self.searchingstep)
            self.TelloMove("down", self.searchingstep)
        if param == 3:
            self.TelloMove("left", self.searchingstep)
        if param == 4:
            self.TelloMove("left", self.searchingstep)
            self.TelloMove("up", self.searchingstep)
        if param == 5:
            self.TelloMove("up", self.searchingstep)
        if param == 6:
            self.TelloMove("right", self.searchingstep)
            self.TelloMove("up", self.searchingstep)
        if param == 7:
            self.TelloMove("right", self.searchingstep)
        if param == 8:
            self.TelloMove("right", self.searchingstep)
            self.TelloMove("down", self.searchingstep)
        else:
            return
        return

    def Str2IntList(self, stringlist):
        intlist = [stringlist[0]]
        if not stringlist[0]:  # if Shape
            for string in stringlist[1:]:
                if string == "Triangle" or string == "triangle":
                    intlist.append(Communication.Shape.Triangle.value)
                elif string == "Rectangle" or string == "rectangle":
                    intlist.append(Communication.Shape.Rectangle.value)
                elif string == "Square" or string == "square":
                    intlist.append(Communication.Shape.Square.value)
                elif string == "Rhombus" or string == "rhombus":
                    intlist.append(Communication.Shape.Rhombus.value)
                elif string == "Pentagon" or string == "pentagon":
                    intlist.append(Communication.Shape.Pentagon.value)
                elif string == "Octagon" or string == "octagon":
                    intlist.append(Communication.Shape.Octagon.value)
                elif string == "Star" or string == "star":
                    intlist.append(Communication.Shape.Star.value)
                elif string == "Circle" or string == "circle":
                    intlist.append(Communication.Shape.Circle.value)
                elif string == None:
                    intlist.append(0)
                else:
                    intlist.append(Communication.Shape.Unknown.value)
                    if self.debug:
                        print(f"Could not match string {string} to Shape Class")
        else:
            for string in stringlist[1:]:
                if string == "Green" or string == "green":
                    intlist.append(Communication.Color.Green.value)
                elif string == "Orange" or string == "orange":
                    intlist.append(Communication.Color.Orange.value)
                elif string == "Pink" or string == "pink":
                    intlist.append(Communication.Color.Pink.value)
                elif string == "Red" or string == "red":
                    intlist.append(Communication.Color.Red.value)
                elif string == "Purple" or string == "purple":
                    intlist.append(Communication.Color.Purple.value)
                elif string == "Yellow" or string == "yellow":
                    intlist.append(Communication.Color.Yellow.value)
                elif string == "Brown" or string == "brown":
                    intlist.append(Communication.Color.Brown.value)
                elif string == "Blue" or string == "blue":
                    intlist.append(Communication.Color.Blue.value)
                elif string == None:
                    intlist.append(0)
                else:
                    intlist.append(Communication.Color.Unknown.value)
                    if self.debug:
                        print(f"Could not match string {string} to Color Class")
        return intlist



def main():

    full_ditpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\second_try\second_try11"
    MainTello = TelloControl(dirpath=full_ditpath, debug=True)
    logging.basicConfig(level=logging.INFO, filename=MainTello.dirpath + '\\sample.log')
    IScommunication = False

    # Starting Tello with or Without Communication
    if IScommunication:
        ip = "192.168.0.1"
        port = 8080
        Server = Communication.CServer(ip, port)
        threadbox = threading.Thread(target=Server.SendReciveThread, args=())
        threadbox.setDaemon(True)
        threadbox.start()
        MainTello.Start(server=Server)
    else:
        MainTello.Start()


    # Main Loop with status update
    while True:
        MainTello.FindAndReadScreen()
        if MainTello.status == "sending data to a robot":
            if IScommunication:
                if MainTello.Server.Message:
                    if MainTello.Server.Message[0] == str(Communication.MessageType.DoneSequence.value):
                        print(f'message recieved {MainTello.Server.Message}')
                        logging.info(f"got Message Done sequence from Robot while seeing silver screen")
                        MainTello.Server.Message = None
                        MainTello.status = "found non silver screen"
                        MainTello.ImageProcessing.screenmode = 4   # Act as if there were a Red screen
            else:
                time.sleep(3)  # wait and take another picture to ensure silver screen statue
                if MainTello.debug:
                    print(f"sending data to the robot wait and take another picture to ensure silver screen")
                    logging.info(f"sending data to the robot wait and take another picture to ensure silver screen")
        if MainTello.status == "found non silver screen":
            if MainTello.debug:
                print(f"Tello status {MainTello.status} and screen mode is {MainTello.ImageProcessing.screenmode}")
                logging.info(f"Tello status {MainTello.status} and screen mode is {MainTello.ImageProcessing.screenmode}")
            if MainTello.ImageProcessing.screenmode == 1:  # black screen
                if MainTello.debug:
                    print(f"Tello status rotate clockwise ")
                    logging.info(f"Tello status rotate clockwise ")
                MainTello.Rotate("clockwise", 90)
            if MainTello.ImageProcessing.screenmode == 4:  # red screen
                if IScommunication:
                    MainTello.SendDataToRobot([Communication.MessageType.RedScreen.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                if MainTello.debug:
                    print(f"wait for the screen to get back to silver")
                    logging.info(f"wait for the screen to get back to silver ")
                time.sleep(3)  # wait for the screen to get back to silver
            if MainTello.ImageProcessing.screenmode == 3:  # green screen
                if MainTello.debug:
                    print(f"wait and rotate after green screen")
                    logging.info(f"wait and rotate after green screen")
                time.sleep(3)  # wait for some screen to became silver
                MainTello.Rotate("clockwise", 90)
        if MainTello.status == "did not find screen":
            if MainTello.debug:
                print(f"did not find screen, breaking and landing")
                logging.info(f"did not find screen, breaking and landing")
            break

    MainTello.End()


if __name__ == '__main__':
    main()

