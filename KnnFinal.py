from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import random
import numpy as np
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


#Get an instance of the model
#Parameter tuning: find the number of neighbors that results in best predictions

accuracies=[]
models=[]
for k in range(1,11):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    #Train the model
    knn_model.fit(xtrain,ytrain)
    #Predict from the test data and compare with actual labels
    predictions = knn_model.predict(xtest)
    #accuracy = np.mean(predictions==ytest)
    accuracies.append(np.mean(predictions==ytest))
    models.append(knn_model)

for k in range(1,11):
    print("Accuracy for k: " ,k)
    print(accuracies[k-1])

# Plotting accuracies
# plt.plot(range(1,11),accuracies)
# plt.ylabel("Accuracies")
# plt.xlabel("# of Neighbours")
# plt.title("Accuracies")
# plt.grid()
# plt.show()

#Get accuracies as percentages
percentages=[]
for i in range(1,11):
        percentages.append(100*accuracies[i-1])

#Find maximum value and corresponding K index
maximum = max(percentages)
print("Max percentage: ", maximum)
print("K-value: ", percentages.index(maximum)+1)
print("K-value: ", accuracies.index(max(accuracies))+1)

index = percentages.index(maximum)+1
print(percentages[index-1])

plt.plot(range(1,11),percentages)
plt.ylabel("Accuracies Percentages")
plt.xlabel("# of Neighbours")
plt.title("Percentages")
plt.grid()
plt.show()

optimized_model = models[index-1]
#Save the model in 'model.sav' folder
pick = open('knn_model.sav', 'wb')
pickle.dump(knn_model, pick)
pick.close()
