import pickle
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Define the directory and categories and get the saved anthropometric model features from saved pickle file
#directory = r"C:\Users\myria\OneDrive\Desktop\Small Dataset"
categories = ['Old', 'Young']
pick_in = open('dataIndecises.pickle', 'rb')

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

print(classification_report(ytest, prediction))
print("depth: " , decision_trees_model.get_depth())
print("prediction", prediction)
print("Testing accuracy " ,score )
print("Numpy accuracy " , np.mean(ytest==prediction))

#Saves the model in 'model.sav' folder
pick = open('decision_trees_model.sav', 'wb')
pickle.dump(decision_trees_model, pick)
pick.close()