import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import csv
os.chdir(sys.argv[1])
img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer


DIR = sys.argv[2]
mydata = []
for i, ret in enumerate(os.walk('../result/' + DIR)):
    print(ret)
    for i, filename in enumerate(ret[2]):
        if "rb" in filename:
            result = predict(ret[0] + '/' + filename)
            mydata.append([result, int(filename.split("_")[0])])
        else:
            continue

myFile = open('../result/' + DIR + '/' + DIR + '.csv', 'w')
mydata.sort(key=lambda x: x[1])
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(["direction", "frame"])
    for x in mydata:
        writer.writerow(x)

print("Writing complete")
