from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import random
import matplotlib.pyplot as plt
import  numpy as np
# Define the directory and categories and get the saved anthropometric model features from saved pickle file
directory = r"C:\Users\myria\OneDrive\Desktop\Small Dataset"
categories = ['Old', 'Young']
pick_in = open('dataIndecises (1).pickle', 'rb')

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

model = KNeighborsClassifier(n_neighbors=7)
model.fit(xtrain,ytrain)

# Save the default model with 5 neighbours in pickle folder
# pick = open('knn_fixmodel_8features.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()

 #Opens and reads the model
pick = open('knn_fixmodel_8features.sav', 'rb')
model = pickle.load(pick)
pick.close()

 #Testing phase: predict and store the predictions of the testing data in model_predictions
model_predictions = model.predict(xtest)

#Another method used to calculate the accuarcy
accuracy = np.mean(model_predictions==ytest)

categories = ['Old', 'Young'];
print('Accuracy: ', accuracy);
print('Prediction is: ', categories[model_predictions[0]]);

   #Plot an edge-detected image from the training set
person = np.reshape(xtest[0], (-1,128,128,1))
# plt.subplot(121),plt.imshow(passport,cmap = 'gray')
plt.subplot(122),plt.imshow(person,cmap = 'gray')
plt.show()


