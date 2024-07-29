#ml_hw1_1_a_iris_datasetwith3classes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
-------------------------
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)

# Train a random classifier for comparison
dummy = DummyClassifier(strategy="uniform")
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)

accuracy_gnb, accuracy_dummy

------------------
#1.b:
# Train a dummy classifier that always predicts class 1 (for example)
dummy_constant = DummyClassifier(strategy="constant", constant=1)
dummy_constant.fit(X_train, y_train)
y_pred_constant = dummy_constant.predict(X_test)
accuracy_constant = accuracy_score(y_test, y_pred_constant)

# Calculate error
error_constant = 1 - accuracy_constant

# Calculate theoretical upper bound for M classes
M = len(set(y))
upper_bound_error = (M - 1) / M

error_constant, upper_bound_error
--------------------
