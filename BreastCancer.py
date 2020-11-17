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
dataset = pd.read_csv('Cancer_data.csv')
X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 31].values

print(dataset.head())

print("Cancer data set dimensions : {}".format(dataset.shape))

dataset.groupby('diagnosis').size()

# Visualization of data
hist = dataset.groupby('diagnosis').hist(figsize=(12, 12))
plt.savefig("plot.pdf")

dataset.isnull().sum()
dataset.isna().sum()

dataframe = pd.DataFrame(Y)
# Encoding categorical data values

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the Logistic Regression Algorithm to the Training Set

classifier = LogisticRegression(solver="lbfgs", random_state=0)
classifier.fit(X_train, Y_train)
# Use score method to get accuracy of model
lg_score = classifier.score(X_test, Y_test)
print("logistic regression:", lg_score * 100)
# 95.8 Accuracy

# Fitting K-NN Algorithm

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)
knn_score = classifier.score(X_test, Y_test)
print("K-NN:", knn_score * 100)
# 95.1 Acuracy

# Fitting SVM

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, Y_train)
svm_score = classifier.score(X_test, Y_test)
print("SVM:", svm_score * 100)
# 97.2 Acuracy

# Fitting K-SVM

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)
ksvm_score = classifier.score(X_test, Y_test)
print("K-VSM:", ksvm_score * 100)
# 96.5 Acuracy

# Fitting Naive_Bayes

classifier = GaussianNB()
classifier.fit(X_train, Y_train)
nb_score = classifier.score(X_test, Y_test)
print("Naive Bayes:", nb_score * 100)
# 91.6 Acuracy

# Fitting Decision Tree Algorithm

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
dt_score = classifier.score(X_test, Y_test)
print("Decision Tree:", dt_score * 100)
# 95.8 Acuracy

# Fitting Random Forest Classification Algorithm

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
rf_score = classifier.score(X_test, Y_test)
print("Random Forest:", rf_score * 100)
# 98.6 Acuracy

# predicting the Test set results
Y_pred = classifier.predict(X_test)

# Creating the confusion Matrix

cm = confusion_matrix(Y_test, Y_pred)
c = (cm[0, 0] + cm[1, 1])
print(c)
