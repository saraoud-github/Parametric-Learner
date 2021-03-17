import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
import random, pickle
import matplotlib.pyplot as plt

#Read the pickle file containing the labeled data
pick_in = open('dataIndecises.pickle', 'rb')
   #Load the pickle file into data variable
data = pickle.load(pick_in)
pick_in.close()

   #Shuffle the data
random.shuffle(data)
dataInd =[]
labels = []
   #Split the elements in data into features and labels
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
y_pred =classifier.predict(X_test)

plot_confusion_matrix(classifier, X_test, y_test,values_format='d', display_labels=["old", "young"])
plt.show()
print(classification_report(y_test,y_pred))
#print(accuracy_score(y_test, y_pred.round(), normalize=True))