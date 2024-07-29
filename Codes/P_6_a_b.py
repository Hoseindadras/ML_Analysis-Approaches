#6_a_b
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Pre-processing
data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Custom train-test split
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

X = data.drop(columns='diagnosis')
y = data['diagnosis']
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Naive Bayes Implementation
prior_probabilities = y_train.value_counts(normalize=True)
class_mean = X_train.groupby(y_train).mean()
class_std = X_train.groupby(y_train).std()

def gaussian_pdf(x, mean, std):
    exponent = np.exp(- ((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def naive_bayes_predict(X_row):
    probabilities = {}
    for class_label in prior_probabilities.index:
        probabilities[class_label] = prior_probabilities[class_label]
        for feature in X.columns:
            probabilities[class_label] *= gaussian_pdf(X_row[feature], class_mean.loc[class_label, feature], class_std.loc[class_label, feature])
    return pd.Series(probabilities).idxmax()

y_pred = X_test.apply(naive_bayes_predict, axis=1)

# Evaluation
def confusion_matrix(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_true, y_pred):
    TP, _, FP, _ = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP)

def recall(y_true, y_pred):
    TP, _, _, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN)

acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)
TP, TN, FP, FN = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("Confusion Matrix: TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
