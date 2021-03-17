import cv2
import dlib
import os
import pickle
import numpy as np
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#SMALLER DATASET
directory = r'C:\Users\sara6\OneDrive\Documents\UserData'

categories = ['Old', 'Young']
dataIndecises = []


    #Loop through all the images of old and young people in Databases Folder
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
                                  eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index], label])

                # dataInd = np.array(dataIndecises).flatten()

        # Write the data into a pickle file before training
print(len(dataIndecises))
for item in dataIndecises:
    print(item)
# for dat in dataIndecises:
#
#   print(dat)
#   print("\n")

pick_in1 = open('dataIndecises.pickle', 'wb')
pickle.dump(dataIndecises, pick_in1)
pick_in1.close()