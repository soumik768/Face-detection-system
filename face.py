import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

from skimage import io
from sklearn.cross_validation import train_test_split


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, Flatten

DatasetPath = []
imageData = []
imageLabels = []
imageDataFin = []


for i in os.listdir("yalefaces"):
    DatasetPath.append(os.path.join("yalefaces", i))

for i in DatasetPath:
    imgRead = io.imread(i,as_grey=True)
    imageData.append(imgRead)
    
    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
    imageLabels.append(labelRead)


faceDetectClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
for i in imageData:
    facePoints = faceDetectClassifier.detectMultiScale(i)
    x,y = facePoints[0][:2]
    cropped = i[y: y + 150, x: x + 150]
    imageDataFin.append(cropped)


c = np.array(imageDataFin)
X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.8, random_state = 20)
X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np_utils.to_categorical(y_train, 15)
Y_test = np_utils.to_categorical(y_test, 15)

X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1],X_train.shape[2]).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1], X_test.shape[2]).astype('float32')


X_train /= 255
X_test /= 255


model = Sequential()

model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(1, size, size), subsample=(1, 1) ))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(p=0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(p=0.2))
model.add(Dense(15, activation='softmax', name='predictions'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, nb_epoch=10000)

predict = model.predict(X_test)

print(predict)

