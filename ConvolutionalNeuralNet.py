import os
import cv2
import dlib
import random
import pickle
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Code to create Pickle file for training
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# SMALLER DATASET
directory = r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Project II\IEA - Project II - Group 6\Databases'

categories = ['Old', 'Young']
dataIndecises = []

# Loop through all the images of old and young people in Databases Folder
for category in categories:
    path = os.path.join(directory, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        passport = cv2.imread(imgpath)

        gray = cv2.cvtColor(passport, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predictor(gray, face)
            facial_index = (landmarks.part(9).y - landmarks.part(28).y) / (
                    landmarks.part(16).x - landmarks.part(2).x)
            mandibular_index = (landmarks.part(9).y - landmarks.part(65).y) / (
                    landmarks.part(13).x - landmarks.part(5).x)
            intercanthal_index = (landmarks.part(43).x - landmarks.part(40).x) / (
                    landmarks.part(46).x - landmarks.part(37).x)
            orbital_width_index = (landmarks.part(46).x - landmarks.part(43).x) / (
                    landmarks.part(43).x - landmarks.part(40).x)
            eye_fissure_index = (landmarks.part(47).y - landmarks.part(45).y) / (
                    landmarks.part(46).x - landmarks.part(43).x)
            nasal_index = (landmarks.part(36).x - landmarks.part(32).x) / (
                    landmarks.part(34).y - landmarks.part(28).y)
            vermilion_height_index = (landmarks.part(65).y - landmarks.part(52).y) / (
                    landmarks.part(58).y - landmarks.part(65).y)
            mouth_face_width_index = (landmarks.part(55).x - landmarks.part(49).x) / (
                    landmarks.part(16).x - landmarks.part(2).x)

            dataIndecises.append([[facial_index, mandibular_index, intercanthal_index, orbital_width_index,
                                   eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index],
                                  label])

            # dataInd = np.array(dataIndecises).flatten()

        # Write the data into a pickle file before training
# print(dataIndecises)
# for dat in dataIndecises:
#
#   print(dat)
#   print("\n")

pick_in1 = open('dataIndecisesCNN.pickle', 'wb')
pickle.dump(dataIndecises, pick_in1)
pick_in1.close()

# load the trained pickle file
pick = open("dataIndecisesCNN.pickle", "rb")
data = pickle.load(pick)
pick.close()

# Split the elements in data into features and labels

random.shuffle(data)
features = []
labels = []
for feature, label in data:
    features.append(feature)
    labels.append(label)

size = len(feature)

# Split the data into train (70%) and test data (30%)
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

# reshape training and testing features lists based on number of features selected by the user
xtrain = np.reshape(xtrain, (-1, size, 1, 1))
xtest = np.reshape(xtest, (-1, size, 1, 1))

# convert to tensors
xtrain = tf.convert_to_tensor(xtrain, dtype=tf.float32)
xtest = tf.convert_to_tensor(xtest)
ytrain = tf.convert_to_tensor(ytrain)
ytest = tf.convert_to_tensor(ytest)

# define the CNN Sequential Model

model = Sequential()

model.add(Conv2D(64, (3,1), input_shape = xtrain.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(xtrain, ytrain, batch_size=1, epochs=19, validation_data=(xtest, ytest))

model.save('CNN_Ratios.model')
# code to predict on new image
image = "103964655-authorphoto.brianwong.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

passport = cv2.imread(image)

gray = cv2.cvtColor(passport, cv2.COLOR_BGR2GRAY)
dataIndex = []
faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    landmarks = predictor(gray, face)
    facial_index = (landmarks.part(9).y - landmarks.part(28).y) / (
            landmarks.part(16).x - landmarks.part(2).x)
    mandibular_index = (landmarks.part(9).y - landmarks.part(65).y) / (
            landmarks.part(13).x - landmarks.part(5).x)
    intercanthal_index = (landmarks.part(43).x - landmarks.part(40).x) / (
            landmarks.part(46).x - landmarks.part(37).x)
    orbital_width_index = (landmarks.part(46).x - landmarks.part(43).x) / (
            landmarks.part(43).x - landmarks.part(40).x)
    eye_fissure_index = (landmarks.part(47).y - landmarks.part(45).y) / (
            landmarks.part(46).x - landmarks.part(43).x)
    nasal_index = (landmarks.part(36).x - landmarks.part(32).x) / (
            landmarks.part(34).y - landmarks.part(28).y)
    vermilion_height_index = (landmarks.part(65).y - landmarks.part(52).y) / (
            landmarks.part(58).y - landmarks.part(65).y)
    mouth_face_width_index = (landmarks.part(55).x - landmarks.part(49).x) / (
            landmarks.part(16).x - landmarks.part(2).x)

    dataIndex.append([facial_index, mandibular_index, intercanthal_index, orbital_width_index,
                      eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index])
dataIndex = np.reshape(dataIndex, (-1, size, 1, 1))

# replace size by 8 for pre trained model
# for user trained model size = nb of ratios

model = tf.keras.models.load_model('CNN_Ratios.model')
prediction = model.predict(dataIndex)
print("Prediction is: ", categories[int(prediction[0][0])])