# Importing necessary libraries for data manipulation and machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# File path where the baseball data is stored
file_path = "C:/Users/cedel/Downloads/baseball.xlsx"

# Loading the dataset into a pandas DataFrame
data = pd.read_excel(file_path)

# Displaying the first few rows of the dataset for initial inspection
print("First few rows of the dataset:")
print(data.head())

# Extracting features (independent variables) and target variable (dependent variable)
features = data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
target = data['Playoffs']

# Splitting the dataset into training and testing sets
# 80% of the data will be used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Creating a Random Forest Classifier model with 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model using the training set
model.fit(X_train, y_train)

# Predicting the target variable (Playoffs) for the testing set
predictions = model.predict(X_test)

# Evaluating the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the model:", accuracy)

# Providing explanations for each line of code:

# Line 4: Importing necessary libraries for data manipulation (pandas) and machine learning (scikit-learn).
# Line 7: Specifying the file path where the baseball data is stored.
# Line 10: Loading the dataset from the specified file path into a pandas DataFrame.
# Line 14: Printing the first few rows of the dataset for initial inspection.
# Line 17-18: Extracting the features (independent variables) and the target variable (dependent variable) from the dataset.
# Line 22: Splitting the dataset into training and testing sets using an 80-20 split.
# Line 26-27: Creating a Random Forest Classifier model with 100 decision trees.
# Line 31: Training the model using the training set.
# Line 35: Predicting the target variable (Playoffs) for the testing set using the trained model.
# Line 39: Evaluating the accuracy of the model by comparing the predicted values with the actual values in the testing set.

# It's important for students to understand each step of data preprocessing, model creation, training, prediction, and evaluation to grasp the fundamentals of programming and data analysis.
