import random, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# get the dataset
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
#print(get_dataset())

# define the base models
level0 = list()
level0.append(('lr', LogisticRegression()))
level0.append(('knn', KNeighborsClassifier()))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC()))
level0.append(('bayes', GaussianNB()))

# define meta learner model
level1 = LogisticRegression()

# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# fit the model on all available data
dataInd, labels = get_dataset()
xtrain, xtest, ytrain, ytest = train_test_split(dataInd, labels, test_size=0.1, random_state=1, stratify=labels)


model.fit(xtrain, ytrain)
# make a prediction for one example

#Testing phase: predict and store the predictions of the testing data in model_predictions
model_predictions = model.predict(xtest)
  #Print out a classification report for the model that includes: precision, accuracy, f-value, and recall
print(classification_report(ytest, model_predictions))
