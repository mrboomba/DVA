import cv2
import sys
import os
import math
from PIL import Image
for filename in os.listdir(sys.argv[1]):
    os.chdir(sys.argv[1])
    print(filename)
    name = filename.split(".")[0];
    if filename == sys.argv[2]:
        os.mkdir("../result/"+name)
        vidcap = cv2.VideoCapture(sys.argv[1]+filename);
        frameRate = vidcap.get(5) #frame rat
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            frameId = vidcap.get(1)
            success,image = vidcap.read()
            if (success != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                print('Read a new frame: ', success)
                cv2.imwrite("../result/"+name+"/%d.png" % count, image)     # save frame as JPEG file
                img = Image.open("../result/"+name+"/%d.png" % count)
                w, h = img.size
                area = ((w/2)-621,(h/2)-187.5 ,(w/2)+621 , (h/2)+187.5)
                cropped_img = img.crop(area)
                cropped_img.save("../result/"+name+"/%d.png" % count)
                count += 1

