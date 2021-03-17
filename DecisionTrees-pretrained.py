import pickle
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#Define the directory and categories and get the saved anthropometric model features from saved pickle file
directory = r"C:\Users\myria\OneDrive\Desktop\Small Dataset"
categories = ['Old', 'Young']
pick_in = open('dataIndecises (1).pickle', 'rb')

#Load the pickle file into data variable
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
features = []
labels = []


#Split the elements in data into features and labels
for feature, label in data:
    features.append(feature)
    labels.append(label)

#Split the data into train (70%) and test data (30%)
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

decision_trees_model=tree.DecisionTreeClassifier()
decision_trees_model.fit(xtrain,ytrain)
prediction=decision_trees_model.predict(xtest)
score=decision_trees_model.score(xtest,ytest)


print("depth: " , decision_trees_model.get_depth())
print("prediction", prediction)
print("Testing accuracy " ,score )
print("Numpy accuracy " , np.mean(ytest==prediction))

#Saves the model in 'model.sav' folder
# pick = open('decision_trees_model.sav', 'wb')
# pickle.dump(decision_trees_model, pick)
# pick.close()


   #Opens and reads the model
pick = open('decision_trees_model.sav', 'rb')
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