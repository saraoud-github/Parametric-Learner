import platform
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QColor, QDropEvent
from ProjectIIGUI import Ui_MainWindow
from ui_functionsII import UIFunctions
from PyQt5 import QtGui
from ProjectIIGUI import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import sys
import os
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil, random, glob, cv2, dlib, os, pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pickle
from PIL import Image as im
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import random, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf


class MainWindow(QMainWindow):
   # GUI Main Window constructor
        def __init__(self):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

            # MOVE WINDOW
            def moveWindow(event):
                # Restore before move
                if UIFunctions.returnStatus(self) == 1:
                    UIFunctions.maximize_restore(self)

                # IF LEFT CLICK MOVE WINDOW
                if event.buttons() == Qt.LeftButton:
                    self.move(self.pos() + event.globalPos() - self.dragPos)
                    self.dragPos = event.globalPos()
                    event.accept()

            # SET TITLE BAR
            self.ui.Title_Bar.mouseMoveEvent = moveWindow

            #Connect the "Predict" button to the testNewImage function
            #self.ui.btn_predict.clicked.connect(self.testNewImage)

            #Connect the "Next" button to the second page of the widget
            self.ui.next_btn1.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
            self.ui.next_btn1.clicked.connect(self.show_img)

            # Connect the "Next" button to the third page of the widget
            self.ui.next_btn1_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))



            # Connect the "Back" button to the first page of the widget
            self.ui.next_btn1_6.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))

            # Connect the "Back" button to the first page of the widget
            self.ui.next_btn1_7.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))


            # Connect the "Predict" button to the fourth page of the widget
            self.ui.next_btn1_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_4))

            #Connect the slider to the text box above it to display a number
            self.ui.horizontalSlider.valueChanged.connect(self.number_changed)

            #Divide the number training data entered by user and store them in a local directory
            self.ui.next_btn1_2.clicked.connect(self.create_dir)
            self.ui.next_btn1_2.clicked.connect(self.show_img)

            #Connect the predict button if a pre-trained model is chosen
            self.ui.next_btn1_3.clicked.connect(self.pre_trained)

            #Connect the train button to a function which starts training the selected model
            self.ui.next_btn1_4.clicked.connect(self.train_model)
            self.ui.next_btn1_4.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_5))



            ## ==> SET UI DEFINITIONS
            UIFunctions.uiDefinitions(self)
            # SHOW ==> MAIN WINDOW
            self.show()

            # APP EVENTS


        def train_model(self):

            if self.ui.radioButton_4.isChecked():
                pick_in = open('dataIndecises.pickle', 'rb')

                # Load the pickle file into data variable
                data = pickle.load(pick_in)
                pick_in.close()
                random.shuffle(data)
                features = []
                labels = []

                # Split the elements in data into features and labels
                for feature, label in data:
                    features.append(feature)
                    labels.append(label)

                # Split the data into train (70%) and test data (30%)
                xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

                # Get an instance of the model
                # Parameter tuning: find the number of neighbors that results in best predictions

                accuracies = []
                models = []
                for k in range(1, int(self.ui.lineEdit_2.text())):
                    knn_model = KNeighborsClassifier(n_neighbors=k)
                    # Train the model
                    knn_model.fit(xtrain, ytrain)
                    # Predict from the test data and compare with actual labels
                    predictions = knn_model.predict(xtest)
                    # accuracy = np.mean(predictions==ytest)
                    accuracies.append(np.mean(predictions == ytest))
                    models.append(knn_model)

                for k in range(1, 11):
                    print("Accuracy for k: ", k)
                    print(accuracies[k - 1])

                # Plotting accuracies
                # plt.plot(range(1,11),accuracies)
                # plt.ylabel("Accuracies")
                # plt.xlabel("# of Neighbours")
                # plt.title("Accuracies")
                # plt.grid()
                # plt.show()

                # Get accuracies as percentages
                percentages = []
                for i in range(1, 11):
                    percentages.append(100 * accuracies[i - 1])

                # Find maximum value and corresponding K index
                maximum = max(percentages)
                print("Max percentage: ", maximum)
                print("K-value: ", percentages.index(maximum) + 1)
                print("K-value: ", accuracies.index(max(accuracies)) + 1)

                index = percentages.index(maximum) + 1
                print(percentages[index - 1])

                plt.plot(range(1, 11), percentages)
                plt.ylabel("Accuracies Percentages")
                plt.xlabel("# of Neighbours")
                plt.title("Percentages")
                plt.grid()
                plt.savefig('knnplot.jpg')
                knnplot = Image.open('knnplot.jpg')
                new_knn_plot = knnplot.resize(510,110)
                new_knn_plot.save('knnplot.jpg')
                plt.show()


                self.ui.label_17.setPixmap(QPixmap('knnplot.jpg'))

                optimized_model = models[index - 1]
                # Save the model in 'model.sav' folder
                pick = open('knn_model.sav', 'wb')
                pickle.dump(knn_model, pick)
                pick.close()


            if self.ui.radioButton_5.isChecked():
                pick_in = open('dataIndecises.pickle', 'rb')

                # Load the pickle file into data variable
                data = pickle.load(pick_in)
                pick_in.close()
                random.shuffle(data)
                features = []
                labels = []

                # Split the elements in data into features and labels
                for feature, label in data:
                    features.append(feature)
                    labels.append(label)

                # Split the data into train (70%) and test data (30%)
                xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

                decision_trees_model = tree.DecisionTreeClassifier()
                decision_trees_model.fit(xtrain, ytrain)
                prediction = decision_trees_model.predict(xtest)

                self.ui.label_17.setText(classification_report(ytest, prediction))

                print("depth: ", decision_trees_model.get_depth())
                print("prediction", prediction)
                # print("Testing accuracy ", score)
                # print("Numpy accuracy ", np.mean(ytest == prediction))

                # Saves the model in 'model.sav' folder
                pick = open('decision_trees_model.sav', 'wb')
                pickle.dump(decision_trees_model, pick)
                pick.close()

            if self.ui.radioButton_6.isChecked():
                # Read the pickle file containing the labeled data
                pick_in = open('dataIndecises.pickle', 'rb')
                # Load the pickle file into data variable
                data = pickle.load(pick_in)
                pick_in.close()

                # Shuffle the data
                random.shuffle(data)
                features = []
                labels = []

                # Split the elements in data into features and labels
                for feature, label in data:
                    features.append(feature)
                    labels.append(label)

                # Split the data into train (70%) and test data (30%)
                xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

                # Define a parameter grid for the SVM model
                param_grid = {'C': [0.1, 1, 10, 100, 1000],
                              'gamma': [0.01, 0.001, 0.0001],
                              'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}

                # Define the SVM model
                svc = svm.SVC(probability=True)
                # Chooses the best parameters from param_grid for the SVM model
                model = GridSearchCV(svc, param_grid, cv=3)
                # Trains the model on the specified training data
                model.fit(xtrain, ytrain)

                # Saves the model in 'model_svm.sav' folder
                pick = open('model_svm.sav', 'wb')
                pickle.dump(model, pick)
                pick.close()
                print("svm")

                # Testing phase: predict and store the predictions of the testing data in model_predictions
                model_predictions = model.predict(xtest)
                # Print out a classification report for the model that includes: precision, accuracy, f-value, and recall
                self.ui.label_17.setText(classification_report(ytest, model_predictions))

            if self.ui.radioButton_7.isChecked():
                # load the trained pickle file
                pick = open("dataIndecises.pickle", "rb")
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
                model = tf.keras.models.load_model('CNN_Ratios.model')
                prediction = model.predict(dataIndex)
                #self.ui.label_16.setText(model.compile(metrics = ['accuracy']))
                print("cnn")



            if self.ui.radioButton_8.isChecked():
                print("random")
                # Read the pickle file containing the labeled data
                pick_in = open('dataIndecises.pickle', 'rb')
                # Load the pickle file into data variable
                data = pickle.load(pick_in)
                pick_in.close()

                # Shuffle the data
                random.shuffle(data)
                dataInd = []
                labels = []
                # Split the elements in data into features and labels
                for ind, label in data:
                    dataInd.append(ind)
                    labels.append(label)

                #
                # print(dataInd)
                # print(labels)

                X_train, X_test, y_train, y_test = train_test_split(dataInd, labels, test_size=0.1, random_state=0)

                # Feature Scaling

                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)

                classifier = RandomForestClassifier(n_estimators=20, random_state=0)
                classifier.fit(X_train, y_train)

                # Saves the model in 'model_forest.sav' folder
                pick = open('model_forest.sav', 'wb')
                pickle.dump(model, pick)
                pick.close()
                y_pred = classifier.predict(X_test)

                plot_confusion_matrix(classifier, X_test, y_test, values_format='d', display_labels=["old", "young"])
                plt.show()
                self.ui.label_17.setText(classification_report(y_test, y_pred))
                # print(accuracy_score(y_test, y_pred.round(), normalize=True))

            if self.ui.radioButton.isChecked():

                def get_dataset():
                    # Read the pickle file containing the labeled data
                    pick_in = open('dataIndecises.pickle', 'rb')
                    # Load the pickle file into data variable
                    data = pickle.load(pick_in)
                    pick_in.close()

                    # Shuffle the data
                    random.shuffle(data)
                    dataInd = []
                    labels = []
                    # Split the elements in data into features and labels
                    for ind, label in data:
                        dataInd.append(ind)
                        labels.append(label)

                    return dataInd, labels

                # define the base models
                level0 = list()
                if self.ui.checkBox_11.isChecked():
                    level0.append(('knn', KNeighborsClassifier()))
                if self.ui.checkBox_12.isChecked():
                    level0.append(('cart', DecisionTreeClassifier()))
                if self.ui.checkBox_13.isChecked():
                    level0.append(('svm', SVC()))
                if self.ui.checkBox_14.isChecked():
                    level0.append(('lr', LogisticRegression()))
                if self.ui.checkBox_15.isChecked():
                    level0.append(('bayes', GaussianNB()))
                # define meta learner model
                level1 = LogisticRegression()

                # define the stacking ensemble
                model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

                # fit the model on all available data
                dataInd, labels = get_dataset()
                xtrain, xtest, ytrain, ytest = train_test_split(dataInd, labels, test_size=0.1, random_state=1,
                                                                stratify=labels)

                model.fit(xtrain, ytrain)
                # Saves the model in 'model_stacking.sav' folder
                pick = open('model_stacking.sav', 'wb')
                pickle.dump(model, pick)
                pick.close()

                # Testing phase: predict and store the predictions of the testing data in model_predictions
                model_predictions = model.predict(xtest)
                # Print out a classification report for the model that includes: precision, accuracy, f-value, and recall
                print("stacking")
                self.ui.label_17.setText(classification_report(ytest, model_predictions))





        def pre_trained(self):

            if self.ui.comboBox.currentText() == 'K-Nearest Neighbor':
                # Open the trained model
                pick = open('knn_premodel_8features.sav', 'rb')
                model = pickle.load(pick)
                pick.close()

                path = self.ui.lineEdit.text()

                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

                data= []
                categories = ['Old', 'Young']

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
                faces = detector(gray)
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()

                    facial_index, mandibular_index, intercanthal_index, orbital_width_index, \
                    eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index = facial_landmarks(face, gray, predictor)

                    data.append([facial_index, mandibular_index, intercanthal_index, orbital_width_index,
                                           eye_fissure_index, nasal_index, vermilion_height_index,
                                           mouth_face_width_index])

                    pred = model.predict(data)
                    self.ui.label_2.setText(categories[pred[0]])
                    cv2.imwrite('orgimg.jpg', img)
                    self.ui.label_5.setPixmap(QPixmap('orgimg.jpg'))

            if self.ui.comboBox.currentText() == 'Decision Trees':

                # Open the trained model
                pick = open('decision_trees_premodel.sav', 'rb')
                model = pickle.load(pick)
                pick.close()

                categories = ['Old', 'Young']

                path = self.ui.lineEdit.text()

                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

                data = []

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
                faces = detector(gray)
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()

                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
                    faces = detector(gray)
                    for face in faces:
                        x1 = face.left()
                        y1 = face.top()
                        x2 = face.right()
                        y2 = face.bottom()

                        facial_index, mandibular_index, intercanthal_index, orbital_width_index, \
                        eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index = facial_landmarks(
                            face, gray, predictor)

                        data.append([facial_index, mandibular_index, intercanthal_index, orbital_width_index,
                                     eye_fissure_index, nasal_index, vermilion_height_index,
                                     mouth_face_width_index])

                        pred = model.predict(data)
                        self.ui.label_2.setText(categories[pred[0]])
                        cv2.imwrite('orgimg.jpg', img)
                        self.ui.label_5.setPixmap(QPixmap('orgimg.jpg'))

            if self.ui.comboBox.currentText() == 'Support Vector Machines':
                path = self.ui.lineEdit.text()

                face_cascade = cv2.CascadeClassifier(
                    'haarcascade_frontalface_default.xml')  # the trained face detection model found in OpenCV

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale

                faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # existing function to detect the faces
                print("Number of Faces Found = {0}".format(len(faces)))

                # Open the trained model
                pick = open('model.sav', 'rb')
                model = pickle.load(pick)
                pick.close()

                if len(faces) != 0:  # if the detector was able find faces in the image
                    for i, face in enumerate(faces):  # iterate over these faces to classify each
                        faces = [face]
                        print('Face Number: ', i + 1)

                        for (
                        x, y, w, h) in faces:  # x,y,w and h make up the 4 courners of the face's bounding rectangle
                            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),
                                                2)  # draw the bounding rectangle on the image
                            roi_gray = gray[y:y + h, x:x + w]
                            roi_color = img[y:y + h, x:x + w]
                            cropimg = img[y:y + h, x:x + w]  # crops the image to have the face only
                            cropimg = cv2.resize(cropimg, (
                                128, 128))  # resize the array of the image to have the same size used for training
                            cropimg = Image.fromarray(cropimg)  # creates an image from the array object
                            cropimg = cropimg.resize(
                                (128, 128))  # resize the shape of the image to the image size previously used
                            cropimg.save('orgimg.jpg')
                            image = cv2.imread('orgimg.jpg')
                            org_resized = cv2.resize(image, (381, 311))
                            cv2.imwrite('orgimg.jpg', org_resized)
                            plt.imshow(cropimg, cmap='gray')
                            plt.show()

                            enhancer = ImageEnhance.Sharpness(cropimg)
                            img_sh = enhancer.enhance(5)
                            sharpened = np.asarray(img_sh)
                            edges = cv2.Canny(sharpened, 128, 128)
                            image = np.array(edges).flatten()  # 1D array of image == 16 384
                            image1 = image.reshape(1, 16384)
                            # image = image.resize(16384)
                            # image1= cv2.resize(image,(128,128))
                            prediction = model.predict(image1)



                            person = image.reshape(128, 128)  # making the 1D array 2D

                            newimg = im.fromarray(person)
                            newimg.save('edgeimg.jpg')
                            img = cv2.imread('edgeimg.jpg')
                            resized = cv2.resize(img, (381, 311))
                            cv2.imwrite('edgeimg.jpg', resized)

                            plt.imshow(person, cmap='gray')
                            # plt.show()

                categories = ['Old', 'Young']
                q = model.predict_proba(image1)
                print(prediction)

                # Print out the prediction onto the second window
                self.ui.label_2.setText((str)(categories[prediction[0]]))
                # Print out the original image with the face detected onto the second window
                self.ui.label_5.setPixmap(QPixmap('orgimg.jpg'))

            if self.ui.comboBox.currentText() == 'Convolutional Neural Network':

                # code to predict on new image
                categories = ['Old', 'Young']
                path = self.ui.lineEdit.text()

                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

                passport = cv2.imread(path)

                gray = cv2.cvtColor(passport, cv2.COLOR_BGR2GRAY)
                dataIndex = []
                faces = detector(gray)
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    facial_index, mandibular_index, intercanthal_index, orbital_width_index, \
                    eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index = facial_landmarks(
                        face, gray, predictor)

                    dataIndex.append([facial_index, mandibular_index, intercanthal_index, orbital_width_index,
                                      eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index])
                dataIndex = np.reshape(dataIndex, (-1, size, 1, 1))

                # replace size by 8 for pre trained model
                # for user trained model size = nb of ratios

                model = tf.keras.models.load_model('CNN_Ratios.model')
                prediction = model.predict(dataIndex)
                print("Prediction is: ", categories[int(prediction[0])])



        def mousePressEvent(self, event):
            self.dragPos = event.globalPos()


        def show_img(self):
            path = self.ui.lineEdit.text()
            img_path = cv2.imread(path)
            cv2.imwrite('orgimg.jpg', img_path)
            self.ui.label_9.setPixmap(QPixmap('orgimg.jpg'))
            self.ui.label_6.setPixmap(QPixmap('orgimg.jpg'))


        def number_changed(self):
            new_value = str(self.ui.horizontalSlider.value())
            self.ui.label_15.setText(new_value)

        def create_dir(self):

            src_folder = r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Project II\IEA - Project II - Group 6\Databases'
            dest = r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Project II\IEA - Project II - Group 6'
            os.mkdir(os.path.join(dest, r'UserData'))


            categories = ['Old', 'Young']
            to_be_moved = []

            #Get number of training data from user
            training_data = (int(self.ui.label_15.text()))//2
            # Loop through all the images of old and young people in Databases Folder
            for category in categories:
                path = os.path.join(src_folder, category)
                to_be_moved = random.sample(glob.glob(path + "\*.jpg"), training_data)
                os.mkdir(os.path.join(r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Project II\IEA - Project II - Group 6\UserData', category))


                for img in to_be_moved:
                    shutil.copy(img, os.path.join(r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Project II\IEA - Project II - Group 6\UserData', category))

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            # SMALLER DATASET
            directory = r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Project II\IEA - Project II - Group 6\UserData'

            dataIndecises = []
            count = 0


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

                        facial_index, mandibular_index, intercanthal_index, orbital_width_index, eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index = facial_landmarks(face, gray, predictor)

                        count += 1

                            # Handle the Anthropometric model radiobutton
                        if self.ui.radioButton_2.isChecked():
                            #print("here")

                            dataIndecises.append(
                                [[facial_index, mandibular_index, intercanthal_index, orbital_width_index,
                                  eye_fissure_index, nasal_index, vermilion_height_index,
                                  mouth_face_width_index],
                                 label])


                        if self.ui.checkBox.isChecked():

                            dataIndecises.append([[facial_index], label])

                        if self.ui.checkBox_4.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(mandibular_index)
                            else:
                                dataIndecises.append([[mandibular_index], label])

                        if self.ui.checkBox_3.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(intercanthal_index)
                            else:
                                dataIndecises.append([[mandibular_index], label])

                        if self.ui.checkBox_9.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(orbital_width_index)
                            else:
                                dataIndecises.append([[mandibular_index], label])


                        if self.ui.checkBox_5.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(eye_fissure_index)
                            else:
                                dataIndecises.append([[mandibular_index], label])


                        if self.ui.checkBox_2.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(nasal_index)
                            else:
                                dataIndecises.append([[mandibular_index], label])


                        if self.ui.checkBox_8.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(vermilion_height_index)
                            else:
                                dataIndecises.append([[ver], label])

                        if self.ui.checkBox_6.isChecked():

                            if len(dataIndecises) != 0:
                                dataIndecises[count-1][0].append(mouth_face_width_index)
                            else:
                                dataIndecises.append([[mouth_face_width_index], label])





        # Write the data into a pickle file before training

            pick_in1 = open('dataIndecises.pickle', 'wb')
            pickle.dump(dataIndecises, pick_in1)
            pick_in1.close()

            # for itemss in dataIndecises:
            #     print(itemss)
            # print("here")

def facial_landmarks(face, gray, predictor):
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
   return facial_index, mandibular_index, intercanthal_index, orbital_width_index, eye_fissure_index, nasal_index, vermilion_height_index, mouth_face_width_index


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())