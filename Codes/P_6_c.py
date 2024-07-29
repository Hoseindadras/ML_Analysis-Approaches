#ml_6_c
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Data Loading
data_solution = pd.read_csv('/mnt/data/data.csv')

# 2. Data Preprocessing
# Extract features and target
X_solution = data_solution.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
y_solution = data_solution['diagnosis'].map({'M': 1, 'B': 0})  # Convert diagnosis to numeric (Malignant: 1, Benign: 0)

# Splitting the data into training and testing sets
X_train_solution, X_test_solution, y_train_solution, y_test_solution = train_test_split(X_solution, y_solution, test_size=0.2, random_state=42)

# Normalizing the data
scaler_solution = StandardScaler()
X_train_solution = scaler_solution.fit_transform(X_train_solution)
X_test_solution = scaler_solution.transform(X_test_solution)

# 3. Training the Naive Bayes Model
gnb_solution = GaussianNB()
gnb_solution.fit(X_train_solution, y_train_solution)

# 4. Model Evaluation
y_pred_solution = gnb_solution.predict(X_test_solution)
acc_solution = accuracy_score(y_test_solution, y_pred_solution)
prec_solution = precision_score(y_test_solution, y_pred_solution)
rec_solution = recall_score(y_test_solution, y_pred_solution)
conf_matrix_solution = confusion_matrix(y_test_solution, y_pred_solution)

# 5. Display Results
acc_solution, prec_solution, rec_solution, conf_matrix_solution
