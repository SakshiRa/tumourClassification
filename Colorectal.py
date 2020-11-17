# Classification of cancer diagnosis
# importing the libraries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# importing the dataset
dataset = pd.read_csv('Data.csv',low_memory=False)

X = dataset.iloc[:, 3:8].values
Y = dataset.iloc[:, 8].values

print(dataset.head())

print("Tumour data set dimensions : {}".format(dataset.shape))

print(dataset.groupby('label').size())

dataset.isnull().sum()
dataset.isna().sum()

dataframe = pd.DataFrame(Y)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Decision Tree Algorithm

classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train, Y_train)
dt_score = classifier.score(X_test, Y_test)
print("Decision Tree:", dt_score * 100)

# Fitting the Logistic Regression Algorithm to the Training Set

classifier = LogisticRegression(solver="lbfgs", random_state=0)
classifier.fit(X_train, Y_train)
# Use score method to get accuracy of model
lg_score = classifier.score(X_test, Y_test)
print("logistic regression:", lg_score * 100)

# predicting the Test set results
Y_pred = classifier.predict(X_test)

# Creating the confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
c = (cm[0, 0] + cm[1, 1])
print(cm)