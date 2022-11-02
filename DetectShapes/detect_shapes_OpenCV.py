import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
#import webcolors
import matplotlib.colors as mc
import time
from PIL import Image, ImageOps
#from keras.models import load_model
import pyzbar.pyzbar as pyzbar
#import tensorflow.python
from os import listdir
from os.path import isfile, join
from os.path import exists
import os
import sys
import argparse
import pytesseract
import Communication
import torch
import yolov5.helpers

class DetectUtils:

    @staticmethod
    def PlotCv2ImageWithPlt(image, title='No Name'):
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_img, interpolation='none')
        plt.title(title)
        plt.show()

    @staticmethod
    def LoadFrame(filename, dirpath):
        full_path = os.path.join(dirpath, filename)
        return cv2.imread(full_path)

    @staticmethod
    def FindContoursGryFrame(GrayImage, ColorImage):
        dst = cv2.Canny(GrayImage, 0, 150)
        blured = cv2.blur(dst, (5, 5), 0)
        img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        shapes_order = []

        for i in range(len(contours)):
            contour = contours[i]
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1] - 5
            (xc, yc, w, h) = cv2.boundingRect(contour)
            color_tuple = DetectUtils.GetBBLocationColor([xc, yc, w, h], cv2.cvtColor(ColorImage, cv2.COLOR_BGR2RGB))
            #if (w > 15) and (h > 15) and (yc > 10) and (xc > 10):
            if (hierarchy[0][i][3] != -1) and (hierarchy[0][i][2] == -1):
                parent_contour = contours[hierarchy[0][i][3]]
                (xcp, ycp, wp, hp) = cv2.boundingRect(parent_contour)
                parent_approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                color_tuple_parent = DetectUtils.GetBBLocationColor([xcp, ycp, wp, hp], cv2.cvtColor(ColorImage, cv2.COLOR_BGR2RGB))
                if color_tuple == color_tuple_parent and len(parent_approx) == 4:
                    parent_aspectRatio = float(wp) / hp
                    if 0.8 <= parent_aspectRatio < 1.8:
                        shapes_order.append(DetectUtils.RecognizeShape(approx=approx, xc=xc, yc=yc, w=w, h=h, color_tuple=color_tuple))

        return shapes_order

    @staticmethod
    def RecognizeShape(approx, xc, yc, w, h, color_tuple):
        if len(approx) == 3:
            # cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shape = [1, xc, yc, w, h, color_tuple]
            # print(f"triangle x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            # print(aspectRatio);
            if 0.95 <= aspectRatio < 1.05:
                # cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                shape = ([2, xc, yc, w, h, color_tuple])
                # print(f"square x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
            else:
                # cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                shape = ([3, xc, yc, w, h, color_tuple])
                # print(f"rectangle x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        elif len(approx) == 5:
            # cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shape = ([4, xc, yc, w, h, color_tuple])
            # print(f"pentagon x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        elif len(approx) == 6:
            # cv2.putText(img, "hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shape = ([5, xc, yc, w, h, color_tuple])
            # print(f"hexagon x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        elif len(approx) == 10:
            # cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shape = ([6, xc, yc, w, h, color_tuple])
            # print(f"star x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        elif len(approx) == 12:
            # cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shape = ([7, xc, yc, w, h, color_tuple])
            # print(f"star x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        else:
            # cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shape = ([8, xc, yc, w, h, color_tuple])
            # print(f"circle x-{xc:4}, y-{yc:4}, width={w:3}, height={h:3}")
        return shape

    # @staticmethod
    # def GetColorName(rgb_triplet):
    #     min_colours = {}
    #     for name, key in mc.CSS4_COLORS.items():
    #         r_c, g_c, b_c = webcolors.hex_to_rgb(key)
    #         rd = (r_c - rgb_triplet[0]) ** 2
    #         gd = (g_c - rgb_triplet[1]) ** 2
    #         bd = (b_c - rgb_triplet[2]) ** 2
    #         min_colours[(rd + gd + bd)] = name
    #     return min_colours[min(min_colours.keys())]

    @staticmethod
    def GetBBLocationColor(locationbb, cv2_img):
        xc, yc, w, h = locationbb
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return tuple(img[int(yc+h/2), int(xc+w/2)])

    @staticmethod
    def ShapesOrderByBoundingBox(shapes_bb, image_size):
        first_row = []
        second_row = []
        for idx, shape in enumerate(shapes_bb):
            if shape[1] <= image_size[1] / 4.5:
                first_row.append([idx, shape])
            else:
                second_row.append([idx, shape])
        first_row_sorted = sorted(first_row, key=lambda i: i[1][0])
        second_row_sorted = sorted(second_row, key=lambda i: i[1][0])
        list_of_orders = first_row_sorted+second_row_sorted
        final_orders_list = []
        for val in list_of_orders:
            final_orders_list.append(val[0])
        return final_orders_list

    @staticmethod
    def CropImageByScreen(full_image):
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(gray, 0, 150)
        blured = cv2.blur(dst, (5, 5), 0)
        results = []
        img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        Contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        i = 0
        for contour in Contours:
            (xc, yc, w, h) = cv2.boundingRect(contour)
            if (w > 200) and (h > 200) and (yc > 10) and (xc > 10):
                i += 1
                print(f"xc {xc}, yc {yc}, w {w}, h {h}")
                cropped_image = full_image[yc:yc + h, xc:xc + w]
                results.append(cropped_image)
        return results

    # @staticmethod
    # def DetectShapesOnCroppedFrame(cropped_frame):
    #     imgGry = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    #     contours = DetectUtils.FindContoursGryFrame(imgGry, cropped_frame)
    #     sorted_contours = DetectUtils.SortContours(contours, cropped_frame)
    #     return sorted_contours

    # @staticmethod
    # def PrintOrderWithColors(sorted_contours):
    #     # print shapes order
    #     for ordered_shape in sorted_contours:
    #         if ordered_shape[0] == 1:
    #             print('triangle')
    #         elif ordered_shape[0] == 2:
    #             print('square')
    #         elif ordered_shape[0] == 3:
    #             print('rectangle')
    #         elif ordered_shape[0] == 4:
    #             print('pentagon')
    #         elif ordered_shape[0] == 5:
    #             print('hexagon')
    #         elif ordered_shape[0] == 6:
    #             print('star')
    #         elif ordered_shape[0] == 7:
    #             print('david star')
    #         elif ordered_shape[0] == 8:
    #             print('circle')
    #         print(f"with {DetectUtils.GetColorName(ordered_shape[5])} color")

    @staticmethod
    def get_dominant_color2(cv2_img, bb=None, palette_size=16):
        # Resize image to speed up processing
        color_coverted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_coverted)
        # pil_img.show()
        img = pil_img.copy()
        if bb is not None:
            img = pil_img.crop((bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]))
        img.thumbnail((100, 100))
        # Reduce colors (uses k-means internally)
        paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)
        # Find the color that occurs most often
        palette = paletted.getpalette()
        color_counts = sorted(paletted.getcolors(), reverse=True)
        palette_index = color_counts[0][1]
        dominant_color = palette[palette_index * 3:palette_index * 3 + 3]
        return dominant_color

    @staticmethod
    def DistanceBetweenTwoTuples(tuple1, tuple2):
        xx = (tuple1[0] - tuple2[0]) ** 2
        yy = (tuple1[1] - tuple2[1]) ** 2
        zz = (tuple1[2] - tuple2[2]) ** 2
        return np.sqrt(xx + yy + zz)

    @staticmethod
    def RecognizeDigit(cv2_img, model):
        color_coverted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        grayImage = cv2.cvtColor(color_coverted, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        pil_img = Image.fromarray(blackAndWhiteImage)
        # resize image to 28x28 pixels
        img = pil_img.resize((28, 28))
        # convert rgb to grayscale
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = np.array(img)
        # reshaping to support our model input and normalizing
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        # predicting the class
        res = model.predict([img])[0]
        return np.argmax(res), max(res)

    @staticmethod
    def read_qr_code(cv2_img):
        """read the QR code.
        Args:
            cv2_img (np.array in cv2 format): image
        Returns:
            qr (string): Value from QR code
        """
        colors_list = ['green', 'orange', 'pink', 'brown', 'red', 'yellow', 'blue', 'purple']
        shapes_list = ["circle", "octagon", "pentagon", "rectangle", "rhombus", "square", "star", "triangle"]
        try:
            flag = True
            detections = pyzbar.decode(cv2_img, symbols=[pyzbar.ZBarSymbol.QRCODE])
            fullstring = detections[0].data
            string_repr = fullstring.decode('utf-8')
            str_list = string_repr.split("\n")
            del str_list[-1]
            result_string_list = [None]*9
            if str_list[0] in colors_list:
                result_string_list[0] = 1
            elif str_list[0] in shapes_list:
                result_string_list[0] = 0
            for idx in range(len(str_list)):
                result_string_list[idx+1] = str_list[idx]
            return flag, result_string_list
        except:
            return False, False


    @staticmethod
    def extractImages(pathIn, pathOut):
        count = 0
        vidcap = cv2.VideoCapture(pathIn)
        success , image = vidcap.read()
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line \n",
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            filename = r"\\frame" + str(count) + ".jpg"
            cv2.imwrite( pathOut + filename , image)
            count = count + 1

    @staticmethod
    def RecognizeColor(rgb_color_tuple):
        # (RED, GREEN, BLUE)
        norm_color = np.array(rgb_color_tuple)/max(rgb_color_tuple)
        if norm_color[1] == 1:
            return 0, "Green"
        if norm_color[2] == 1:
            if norm_color[0] < norm_color[1]:
                return 1, "Blue"
            else:
                return 2, "Purple"
        if norm_color[2] > 3*norm_color[1]:
            return 3, "Pink"
        if norm_color[1]+norm_color[2] <= 0.3*norm_color[0] :
            return 4, "Red"
        if (norm_color[1]+norm_color[2] >= 0.6*norm_color[0]) and (0.5*norm_color[2] <= norm_color[1] <= 1.5*norm_color[2]):
            return 5, "Brown"
        if norm_color[1] >= 0.65 and norm_color[1] >= 3*norm_color[2]:
            return 6, "Yellow"
        if norm_color[1] < 0.65 and norm_color[1] >= 3*norm_color[2]:
            return 7, "Orange"
        else:
            return 8, "Unrecognized_Color"

    @staticmethod
    def PrepareReturnList(orderedlist, outputflag=0):
        return_list = [None]*9
        return_list[0] = outputflag
        if outputflag:
            for idx, item in enumerate(orderedlist):
                if idx+1 < 9:
                    return_list[idx+1] = (DetectUtils.RecognizeColor(item)[1])
            return return_list

    @staticmethod
    def ColorOrShape(listofwords, colors_list, shapes_list):
        #colors_list = ['green', 'orange', 'pink', 'brown', 'red', 'yellow', 'blue', 'purple']
        #shapes_list = ["circle", "octagon", "pentagon", "rectangle", "rhombus", "square", "star", "triangle"]
        for word in listofwords:
            if word in colors_list:
                return 1
            if word in shapes_list:
                return 0
        return None

    @staticmethod
    def RecognizeShapesFromListWords(listofwords):
        colors_list = ['green', 'orange', 'pink', 'brown', 'red', 'yellow', 'blue', 'purple']
        shapes_list = ["circle", "octagon", "pentagon", "rectangle", "rhombus", "square", "star", "triangle"]
        colorshape = DetectUtils.ColorOrShape(listofwords, colors_list, shapes_list)
        if colorshape is None:
            return None
        elif colorshape == 1:
            finallist = DetectUtils.PickClosestWords(listofwords, colors_list)
            finallist.insert(0, 1)
            return finallist
        elif colorshape == 0:
            finallist = DetectUtils.PickClosestWords(listofwords, shapes_list)
            finallist.insert(0, 0)
            return finallist
        return None

    @staticmethod
    def PickClosestWords(listofwords, wordsdictionary):
        result_list = []
        for word in listofwords:
            if word in wordsdictionary:
                result_list.append(word)
            else:
                IsClose, DictionaryWord = DetectUtils.CloseByOneLetter(word, wordsdictionary)
                if IsClose:
                    result_list.append(DictionaryWord)
        return result_list

    @staticmethod
    def CloseByOneLetterOLD(word, wordsdictionary):
        wordletterslist = [*word]
        for dictionaryword in wordsdictionary:
            dictionarywordletterslist = [*dictionaryword]
            if len(wordletterslist) == len(dictionarywordletterslist):
                strike = 0
                for letteridx in range(len(wordletterslist)):
                    if wordletterslist[letteridx] != dictionarywordletterslist[letteridx]:
                        strike += 1
                if strike <= 1:
                    return 1, dictionaryword
        return 0, None

    @staticmethod
    def CloseByOneLetter(word, wordsdictionary):
        wordletterslist = [*word]
        for dictionaryword in wordsdictionary:
            dictionarywordletterslist = [*dictionaryword]
            if len(wordletterslist) == len(
                    dictionarywordletterslist):  # if all the letters are there but one letter is incorrect
                strike = 0
                for letteridx in range(len(wordletterslist)):
                    if wordletterslist[letteridx] != dictionarywordletterslist[letteridx]:
                        strike += 1
                if strike <= 1:
                    return 1, dictionaryword
            elif len(wordletterslist) == len(dictionarywordletterslist) - 1:  # if one letter is missing in read word
                strike = 0
                dictidx = 0
                wordidx = 0
                while dictidx < len(dictionarywordletterslist):
                    if wordletterslist[wordidx] == dictionarywordletterslist[dictidx]:
                        dictidx += 1
                        wordidx += 1
                    else:
                        dictidx += 1
                        strike += 1
                if strike <= 1:
                    return 1, dictionaryword

        return 0, None

    @staticmethod
    def RemoveDuplicateShapes(ordered_shapes, ordered_shapes_color, ordered_shapes_bb):
        ordered_shapesn = []
        ordered_shapes_colorn = []
        ordered_shapes_bbn = []
        for idx, shapebb in enumerate(ordered_shapes_bb):
            if idx > 0:
                if DetectUtils.DistanceBetweenTwoBB(shapebb, ordered_shapes_bbn[-1]):
                    ordered_shapesn.append(ordered_shapes[idx])
                    ordered_shapes_colorn.append(ordered_shapes_color[idx])
                    ordered_shapes_bbn.append(ordered_shapes_bb[idx])
            else:
                ordered_shapesn.append(ordered_shapes[idx])
                ordered_shapes_colorn.append(ordered_shapes_color[idx])
                ordered_shapes_bbn.append(ordered_shapes_bb[idx])

        return ordered_shapesn, ordered_shapes_colorn, ordered_shapes_bbn

    @staticmethod
    def DistanceBetweenTwoBB(boundingbox1, boundingbox2):
        (xc1, yc1, w1, h1) = boundingbox1
        (xc2, yc2, w2, h2) = boundingbox2
        center_bb1 = (int(xc1 + w1/2), int(yc1 + h1/2))
        center_bb2 = (int(xc2 + w2/2), int(yc2 + h2/2))
        if (math.pow((center_bb2[0] - center_bb1[0]),2) + math.pow((center_bb2[1] - center_bb1[1]),2) > math.pow(w1/2,2) + math.pow(h1/2,2)):
            return True
        else:
            return False


class DetectScreenInFrame:

    def __init__(self, filename=None, dirpath=None):
        self.filename = filename
        self.dirpath = dirpath
        self.original_image = None
        self.screenmode = 0  # 0 - no screen , 1 - black screen, 2- silver screen, 3- green screen, 4- red screen
        self.screen = None
        self.bb = None
        self.model = yolov5.helpers.load_model(model_path=r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Yolo5_Simon\Repo\yolov5l.pt")

    def DetectScreen(self, print_screens=False):
        if self.filename is not None:
            self.original_image = DetectUtils.LoadFrame(self.filename, self.dirpath)
        self.screenmode = 0
        results = self.model(self.original_image)
        results.print()
        cropped_image = None
        (xc, yc, w, h) = (0, 0, 0, 0)
        prediction = results.pandas().xyxy[0]
        for i in range(len(prediction)):
            # cropped_image = cv2.imshow(f"Cropped screen with class {prediction.iloc[i]['class']}",
            #                            img[prediction.iloc[i].ymin.astype(int):prediction.iloc[i].ymax.astype(int),
            #                            prediction.iloc[i].xmin.astype(int):prediction.iloc[i].xmax.astype(int)])
            if prediction.iloc[i]['class'].astype(int) in [62, 74, 63]:  # TV Clock Laptop
                print("Yolo5 recognized screen")
                print(f"we Have TV with confidence of {prediction.confidence.iloc[i] * 100}%")
                print(f"x min = {prediction.iloc[i].xmin} , xmax = {prediction.iloc[i].xmax}")
                print(f"y min = {prediction.iloc[i].ymin}, ymax = {prediction.iloc[i].ymax}")
                cropped_image = self.original_image[prediction.iloc[i].ymin.astype(int):prediction.iloc[i].ymax.astype(int), prediction.iloc[i].xmin.astype(int):prediction.iloc[i].xmax.astype(int)]
                xc = prediction.iloc[i].xmin.astype(int)
                yc = prediction.iloc[i].ymin.astype(int)
                w = prediction.iloc[i].xmax.astype(int) - prediction.iloc[i].xmin.astype(int)
                h = prediction.iloc[i].ymax.astype(int) - prediction.iloc[i].ymin.astype(int)

        # color_tuple = get_dominant_color(file_full_path, bb)
        if cropped_image is not None:
            color_tuple = DetectUtils.get_dominant_color2(cropped_image)
            #DetectUtils.PlotCv2ImageWithPlt(cropped_image, "screen image")
            #print(color_tuple)
            #print(DetectUtils.GetColorName(color_tuple))
            if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (0, 0, 0)) <= 60:
                print("black screen")
                self.screenmode = 1
                print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (0, 0, 0))}")
                if print_screens:
                    cv2.imshow("black screen image", cropped_image)
                    cv2.waitKey(0)
                    # DetectUtils.PlotCv2ImageWithPlt(self.original_image, "original image ")
                    # DetectUtils.PlotCv2ImageWithPlt(cropped_image, "black screen image")
                self.screen = cropped_image
                self.bb = (xc, yc, w, h)
            if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (170, 170, 170)) <= 150:
                print("silver screen")
                self.screenmode = 2
                print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (170, 170, 170))}")
                if print_screens:
                    cv2.imshow("silver screen image", cropped_image)
                    cv2.waitKey(0)
                self.screen = cropped_image
                self.bb = (xc, yc, w, h)
            if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (200, 0, 0)) <= 60:
                print("red screen")
                self.screenmode = 4
                print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (200, 0, 0))}")
                if print_screens:
                    cv2.imshow("red screen image", cropped_image)
                    cv2.waitKey(0)
                self.screen = cropped_image
                self.bb = (xc, yc, w, h)
            if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (20, 100, 20)) <= 60:
                print("green screen")
                self.screenmode = 3
                print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (20, 70, 20))}")
                if print_screens:
                    cv2.imshow("green screen image", cropped_image)
                    cv2.waitKey(0)
                self.screen = cropped_image
                self.bb = (xc, yc, w, h)
        else:
            self.DetectScreenOLD(print_screens=print_screens)

    def DetectScreenOLD(self, print_screens=False):
        if self.filename is not None:
            self.original_image = DetectUtils.LoadFrame(self.filename, self.dirpath)
        self.screenmode = 0
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(gray, 0, 150)
        blured = cv2.blur(dst, (5, 5), 0)
        img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        Contours, imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        i = 0
        #print(f"number of detected contours is: {len(Contours)}")
        for contour in Contours:
            (xc, yc, w, h) = cv2.boundingRect(contour)

            # cropped_image = self.original_image[yc:yc + h, xc:xc + w]
            # DetectUtils.PlotCv2ImageWithPlt(cropped_image, "screen image")
            # print(f"xc {xc}, yc {yc}, w {w}, h {h}")
            if (w > 200) and (h > 200) and (yc > 1) and (xc > 1):
                cropped_image = self.original_image[yc:yc + h, xc:xc + w]
                #DetectUtils.PlotCv2ImageWithPlt(cropped_image, "screen image")
                if (1.8 * h > w > 1.3 * h) and (xc + w < 1279) and (yc + h < 719):
                    i += 1
                    #print(f"xc {xc}, yc {yc}, w {w}, h {h}")
                    # cropped_image = self.original_image[yc:yc + h, xc:xc + w]
                    # DetectUtils.PlotCv2ImageWithPlt(cropped_image, "screen image")
                    if i == 1:
                        bb = (0, 0, 0, 0)
                    print(np.abs(xc + yc + w + h - bb[0] - bb[1] - bb[2] - bb[3]))
                    if np.abs(xc + yc + w + h - bb[0] - bb[1] - bb[2] - bb[3]) > 150:
                        bb = (xc, yc, w, h)
                        # color_tuple = get_dominant_color(file_full_path, bb)
                        color_tuple = DetectUtils.get_dominant_color2(cropped_image)
                        #DetectUtils.PlotCv2ImageWithPlt(cropped_image, "screen image")
                        #print(color_tuple)
                        #print(DetectUtils.GetColorName(color_tuple))
                        if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (0, 0, 0)) <= 60:
                            print("black screen")
                            self.screenmode = 1
                            print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (0, 0, 0))}")
                            if print_screens:
                                cv2.imshow("black screen image", cropped_image)
                                cv2.waitKey(0)
                            self.screen = cropped_image
                            self.bb = (xc, yc, w, h)
                        if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (170, 170, 170)) <= 150:
                            print("silver screen")
                            self.screenmode = 2
                            print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (170, 170, 170))}")
                            if print_screens:
                                cv2.imshow("silver screen image", cropped_image)
                                cv2.waitKey(0)
                            self.screen = cropped_image
                            self.bb = (xc, yc, w, h)
                        if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (200, 0, 0)) <= 60:
                            print("red screen")
                            self.screenmode = 4
                            print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (200, 0, 0))}")
                            if print_screens:
                                cv2.imshow("red screen image", cropped_image)
                                cv2.waitKey(0)
                            self.screen = cropped_image
                            self.bb = (xc, yc, w, h)
                        if DetectUtils.DistanceBetweenTwoTuples(color_tuple, (20, 100, 20)) <= 60:
                            print("green screen")
                            self.screenmode = 3
                            print(f"distance is {DetectUtils.DistanceBetweenTwoTuples(color_tuple, (20, 70, 20))}")
                            if print_screens:
                                cv2.imshow("green screen image", cropped_image)
                                cv2.waitKey(0)
                            self.screen = cropped_image
                            self.bb = (xc, yc, w, h)

    def DetectfeaturesInSilverScreen(self):
        if self.screen is not None:
            GrayImage = cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY)
            dst = cv2.Canny(GrayImage, 0, 150)
            blured = cv2.blur(dst, (5, 5), 0)
            img_thresh = cv2.adaptiveThreshold(blured, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            shapes_order = []
            cropped_shapes = []
            cropped_shapes_hierarchy = []
            cropped_shapes_index = []
            cropped_shapes_contour = []
            filtered_duplicate_shapes_index = []
            screen_dominant_color = DetectUtils.get_dominant_color2(self.screen)
            print("Shapes Prints")
            for i in range(len(contours)):
                contour = contours[i]
                (xc, yc, w, h) = cv2.boundingRect(contour)
                #color_tuple = DetectUtils.GetBBLocationColor([xc, yc, w, h], cv2.cvtColor(self.screen, cv2.COLOR_BGR2RGB))
                aspect_ratio = w / h
                if 2 >= aspect_ratio >= 0.5:
                    #DetectUtils.PlotCv2ImageWithPlt(self.screen[yc:yc + h, xc:xc + w], "before first filter")
                    if (100 > w >= 20 and 100 > h >= 20) and hierarchy[0][i][3] != -1:
                        #DetectUtils.PlotCv2ImageWithPlt(self.screen[yc:yc + h, xc:xc + w], "before second filter")
                        if (hierarchy[0][i][3] not in cropped_shapes_index) and (hierarchy[0][i][3] not in filtered_duplicate_shapes_index):
                            cropped_shape = self.screen[yc:yc + h, xc:xc + w]
                            cropped_shapes_contour.append(contour)
                            cropped_shapes.append(cropped_shape)
                            cropped_shapes_index.append(i)
                            cropped_shapes_hierarchy.append(hierarchy[0][i])
                        else:
                            filtered_duplicate_shapes_index.append(i)
            # Filter To shapes and numbers
            filtered_cropped_shapes = []
            filtered_cropped_numbers = []
            filtered_cropped_shapes_bb = []
            filtered_cropped_shapes_color = []
            for i in range(len(cropped_shapes)):
                #DetectUtils.PlotCv2ImageWithPlt(cropped_shapes[i], 'number or shape')
                (xc, yc, w, h) = cv2.boundingRect(cropped_shapes_contour[i])
                if w / h > 0.95: # 0.8
                    if yc >= 0.6*self.screen.shape[0]:
                        filtered_cropped_numbers.append(cropped_shapes[i])
                    else:
                        filtered_cropped_shapes.append(cropped_shapes[i])
                        filtered_cropped_shapes_bb.append((xc, yc, w, h))
                        filtered_cropped_shapes_color.append(DetectUtils.GetBBLocationColor([xc, yc, w, h], self.screen))
                        #print(f"dominant color is {DetectUtils.GetColorName(DetectUtils.get_dominant_color2(cropped_shapes[i]))} wit tuple {DetectUtils.get_dominant_color2(cropped_shapes[i])}")
                        #print(f"center color is {DetectUtils.GetColorName(DetectUtils.GetBBLocationColor([xc, yc, w, h], self.screen))} with tuple {DetectUtils.GetBBLocationColor([xc, yc, w, h], self.screen)}")
                else:
                    filtered_cropped_numbers.append(cropped_shapes[i])
            return filtered_cropped_shapes, filtered_cropped_shapes_color, filtered_cropped_shapes_bb, filtered_cropped_numbers
        else:
            return [], [], [], []

    def DetectWordsInSilverScreenOLD(self):
        if self.screen is not None:
            GrayImage = cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY)
            dst = cv2.Canny(GrayImage, 0, 150)
            blured = cv2.blur(dst, (5, 5), 0)
            img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            shapes_order = []
            cropped_shapes = []
            cropped_shapes_hierarchy = []
            cropped_shapes_index = []
            cropped_shapes_contour = []

            for i in range(len(contours)):
                contour = contours[i]
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                #x = approx.ravel()[0]
                #y = approx.ravel()[1] - 5
                (xc, yc, w, h) = cv2.boundingRect(contour)
                #color_tuple = DetectUtils.GetBBLocationColor([xc, yc, w, h], cv2.cvtColor(self.screen, cv2.COLOR_BGR2RGB))
                aspect_ratio = w / h
                if 2 >= aspect_ratio >= 0.5:
                    if ((w >= 15 and h >= 15) ):    #and hierarchy[0][i][2] != -1 and hierarchy[0][i][3] != -1):
                        cropped_shape = self.screen[yc:yc + h, xc:xc + w]
                        DetectUtils.PlotCv2ImageWithPlt(cropped_shape)
                        cropped_shapes_contour.append(contour)
                        cropped_shapes.append(cropped_shape)
                        cropped_shapes_index.append(i)
                        cropped_shapes_hierarchy.append(hierarchy[0][i])
            print(f"first filter shapes {len(cropped_shapes)}")
            return 0, 0, 0, 0
        else:
            return None, None, None, None

    def DetectWordsInSilverScreen(self):
        smallimg = cv2.resize(self.screen, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gry = cv2.cvtColor(smallimg, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 28)
        bnt = cv2.bitwise_not(thr)
        txt = pytesseract.image_to_string(bnt, config="--psm 6")
        str_list = txt.split()
        final_list = DetectUtils.RecognizeShapesFromListWords(str_list)
        return final_list

    def OrderShapes(self, shapes, shapes_color, shapes_bb):
        image_size = self.screen.shape
        ordered_shapes = []
        ordered_shapes_color = []
        ordered_shapes_bb = []
        shapes_order = DetectUtils.ShapesOrderByBoundingBox(shapes_bb, image_size)
        for place in shapes_order:
            #DetectUtils.PlotCv2ImageWithPlt(shapes[place], 'shape')
            ordered_shapes.append(shapes[place])
            ordered_shapes_color.append(shapes_color[place])
            ordered_shapes_bb.append(shapes_bb[place])
        # Removing Duplicates (shapes that are inside other shapes
        ordered_shapes_nd, ordered_shapes_color_nd, ordered_shapes_bb_nd = DetectUtils.RemoveDuplicateShapes(ordered_shapes, ordered_shapes_color, ordered_shapes_bb)
        return ordered_shapes_nd, ordered_shapes_color_nd, ordered_shapes_bb_nd




def main():

    ###Detect Screen with Shapes
    # filename = r"frame60.jpg"
    # dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja2-master\Simon_Resources\pics_from_tello_pov"

    dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja2-master\DetectShapes\pics_from_1658831841451"

##  Frames for shapes
    #filename = r"frame146.jpg"
    #filename = r"frame384.jpg"
    #filename = r"frame431.jpg"
    #filename = r"frame572.jpg"
    #filename = r"frame686.jpg"
##  frame for QR- CODE
    #filename = r"frame625.jpg"
    filename = r"frame640.jpg"

## Frames With Text
    #filename = r"frame231.jpg"

    # dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja2-master\Simon_Resources\pics_from_tello_pov"
    # filename = r"frame162.jpg"


    print_screen = True
    print_shapes_by_order = True
    detect_number = False
    print_number = False

    # if detect_number:
    #     model = load_model('mnist.h5')
    start = time.time()
    ScreenFrame = DetectScreenInFrame(filename=filename, dirpath=dirpath)
    ScreenFrame.DetectScreen(print_screens=print_screen)


    shapes, shapes_color, shapes_bb, numbers = ScreenFrame.DetectfeaturesInSilverScreen()
    print(f"total time for screen features detection = {time.time() - start}")
    if ScreenFrame.screenmode == 2:  # if screen is silver

        if len(shapes) >= 0:

            # IF Detected shapes or Text Boxes Or Qr code Box
            IsTextQrCode = True
            for idx, color in enumerate(shapes_color):
                if abs(int(color[0]) + int(color[1]) - 2 * int(color[2])) > 40:
                    IsTextQrCode = False

            # IF QR Code of Text
            if IsTextQrCode:
                print(f"The Picture Contains Qr Code or Text")
                for i in range(len(shapes)):
                    name = f"shape {i} "
                    if print_shapes_by_order:
                        DetectUtils.PlotCv2ImageWithPlt(shapes[i], name)
                    flag, value = DetectUtils.read_qr_code(shapes[i])
                    if flag:
                        print("QR Code screen")
                        print(f"output string is {value}")
                        break
                    else:
                        print("Text screen")
                        print(f"output string is {ScreenFrame.DetectWordsInSilverScreen()}")
                        break
                if len(shapes) == 0:
                    print("Text screen")
                    print(f"output string is {ScreenFrame.DetectWordsInSilverScreen()}")

            # IF Shapes
            else:
                ordered_shapes, ordered_shapes_color, ordered_shapes_bb = ScreenFrame.OrderShapes(shapes, shapes_color,
                                                                                                  shapes_bb)
                list_ordered_colors = DetectUtils.PrepareReturnList(ordered_shapes_color, outputflag=0)
                print(f"output string is {list_ordered_colors}")
                if print_shapes_by_order:
                    for i in range(len(ordered_shapes)):
                        name = f"shape {i} with color {DetectUtils.RecognizeColor(ordered_shapes_color[i])[1]} "
                        DetectUtils.PlotCv2ImageWithPlt(ordered_shapes[i], name)

        print(f"total time with number recognition = {time.time() - start} seconds")

if __name__ == '__main__':
    main()

