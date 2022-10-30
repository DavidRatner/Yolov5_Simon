import torch
import cv2
import os


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5n - yolov5x6, custom

dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\second_try\second_try4"
filename = "frame_0_Screen detected, screen mode is 1" + ".jpg"

dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\second_try\second_try11"
filename = "frame_6_Screen detected, screen mode is 2" + ".jpg"


# From Pics
# dirpath = r"C:\Users\davidra\Desktop\NINJA_RAFAEL\Ninja_Symon2\tello_pics"
# filename = "frame_23438466" + ".jpg"

full_path = os.path.join(dirpath, filename)
img = cv2.imread(full_path)
cv2.imshow("photresults = {Detections: 1} image 1/1: 720x960 1 tv\nSpeed: 0.0ms pre-process, 317.9ms inference, 0.0ms NMS per image at shape (1, 3, 480, 640)o", img)  # optional display photo
cv2.waitKey(0)

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
prediction = results.pandas().xyxy[0]

# labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
print(f"class is {prediction}")
for i in range(len(prediction)):
    # cropped_image = cv2.imshow(f"Cropped screen with class {prediction.iloc[i]['class']}",
    #                            img[prediction.iloc[i].ymin.astype(int):prediction.iloc[i].ymax.astype(int),
    #                            prediction.iloc[i].xmin.astype(int):prediction.iloc[i].xmax.astype(int)])
    if prediction.iloc[i]['class'].astype(int) in [62, 74, 63]: #== 62 or prediction.iloc[i]['class'].astype(int) == 74 or prediction.iloc[i]['class'].astype(int) == 63:
        print(f"we Have TV with confidence of {prediction.confidence.iloc[i] * 100}%")
        print(f"x min = {prediction.iloc[i].xmin} , xmax = {prediction.iloc[i].xmax}")
        print(f"y min = {prediction.iloc[i].ymin}, ymax = {prediction.iloc[i].ymax}")
        cropped_image = cv2.imshow(f"Cropped screen with class {prediction.iloc[i]['class']}", img[prediction.iloc[i].ymin.astype(int):prediction.iloc[i].ymax.astype(int), prediction.iloc[i].xmin.astype(int):prediction.iloc[i].xmax.astype(int)])
    cv2.waitKey(0)

print("ee")