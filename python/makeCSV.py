import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.image as mpimg
from tqdm import tqdm
import math
import sys
import csv

os.chdir(sys.argv[1])
filename = sys.argv[2]
code = filename.split(".")[0]
# change these depending on your file names / paths
TEST_GROUND_TRUTH_JSON_PATH = './data/drive.json' # change this to the test ground truth

TEST_IMG_PATH = './test/test_IMG/'+code 
DRIVE_TEST_CSV_PATH = './test/driving_test.csv'
TEST_PREDICT_PATH = './test/test_predict/'+code 

WEIGHTS = 'weights.h5'
EVAL_SAMPLE_SIZE = 100 # Number of samples to evaluate to compute MSE



### Preprocessing helpers
def preprocess_image(image):
    image_cropped = image[100:440, :-90] # -> (380, 550, 3)
    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
    return image


def preprocess_image_valid_from_path(image_path, speed):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img, speed

from model import nvidia_model
from opticalHelpers import opticalFlowDenseDim3
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam

N_img_height = 66
N_img_width = 220
N_img_channels = 3


model = nvidia_model()
model.load_weights(WEIGHTS)

COUNT = 0
data = pd.read_csv(DRIVE_TEST_CSV_PATH)
mydata = []
time = 0
for idx in tqdm(range(1, len(data) - 1)):
    row_now = data.iloc[[idx]].reset_index()
    row_prev = data.iloc[[idx - 1]].reset_index()
    row_next = data.iloc[[idx + 1]].reset_index()

    # Find the 3 respective times to determine frame order (current -> next)

    time_now = row_now['time'].values[0]
    time_prev = row_prev['time'].values[0]
    time_next = row_next['time'].values[0]

    if time_now - time_prev > 0 and 0.0000001 < time_now - time_prev < 0.58: # 0.578111 is highest diff i have seen
        # in this case row_prev is x1 and row_now is x2
        row1 = row_prev
        row2 = row_now

    elif time_next - time_now > 0 and 0.0000001 < time_next - time_now < 0.58:
        # in this case row_now is x1 and row_next is x2
        row1 = row_now
        row2 = row_next

    img_2 = cv2.imread(row2['image_path'].values[0])
    high_res_2 = np.copy(img_2)
    high_res_2 = cv2.cvtColor(high_res_2, cv2.COLOR_BGR2RGB)

    x1, y1 = preprocess_image_valid_from_path(row1['image_path'].values[0], row1['speed'].values[0])
    x2, y2 = preprocess_image_valid_from_path(row2['image_path'].values[0], row2['speed'].values[0])

    img_diff = opticalFlowDenseDim3(x1, x2)
    img_diff_reshaped = img_diff.reshape(1, img_diff.shape[0], img_diff.shape[1], img_diff.shape[2])
    prediction = model.predict(img_diff_reshaped)
    error = abs(prediction - y2)

    predict_path = os.path.join(TEST_PREDICT_PATH, str(idx) + '.jpg')
                                   
    # overwrite the prediction of y2 onto image x2
    # save overwritten image x2 to new directory ./data/predict
    

                                   
    # Make a copy 
    x2_copy = high_res_2
    
    # to write new image via openCV
    offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    x2_copy = cv2.resize(x2_copy, (640, 480), interpolation = cv2.INTER_AREA)
    cv2.putText(x2_copy,'pred: ' + str(prediction[0][0])[:5],(5,offset), font, 1,(66,220,224),1,cv2.LINE_AA)

    if math.floor(time_now) != time :
        time = math.floor(time_now)
        mydata.append([time,str(prediction[0][0])[:5]])
    
    
    
    #cv2.putText(x2_copy,'truth: ' + str(y2)[:5],(5,offset * 2), font, 1,(0,20,255),1,cv2.LINE_AA)
    #cv2.putText(x2_copy, 'error: ' + str(error[0][0])[:5], (5, offset*3),font, 1, (255, 0, 0),1, cv2.LINE_AA)
    
    # convert back to BGR for writing
    x2_copy = cv2.cvtColor(x2_copy, cv2.COLOR_RGB2BGR)
    COUNT += 1
    cv2.imwrite(predict_path, x2_copy)
myFile = open('../result/' + code + '/' + code + '_vel.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(["frame", "velocity"])
    for x in mydata:
        writer.writerow(x)
    
print('done creating test predictions')

# from moviepy.editor import VideoFileClip
# from moviepy.editor import ImageSequenceClip
# import glob
# import os

# images = [TEST_PREDICT_PATH + str(i+1) + '.jpg' for i in range(0, COUNT - 1)]
# clip = ImageSequenceClip(images, fps=11.7552)
# clip.write_videofile("movie-vTest2.mp4", fps = 11.7552)
# print('done creating video')





