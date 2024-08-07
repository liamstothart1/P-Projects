#Question 1 
import pandas  as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from src_1 import utils
import seaborn as sns  
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve


# load data into dataframe (1)
df = pd.read_csv('data/pulsar.csv')
print(df.head())

# Split the data into feature and class (2)
X = df.drop('class', axis=1)
y = df['class']
print('STD&Mean for Fmatrix', X.describe().loc[['mean', 'std']])

# standardize the features (3)
scaler = StandardScaler()
X_s = scaler.fit_transform(X) # X_scaled data

print('scaled mean:', X_s.mean(axis=0)) #preview data to check scaling
print('sacled std:', X_s.std(axis=0))

# How many values (4)
print(f'There are {X_s.shape[0]} values in the classes')
print(f'There are {X_s.shape[1]} classes')

# Question 2 

from RFC_default import rfc_default

rfc_default(X_s, y, 0.5)

# Question 3 


# a dictionary of parameter names to search, and the values to try out
param_grid = {
    'min_samples_leaf': np.arange(2,8,2),
#    'n_estimators': np.arange(61, 65, 2),
#    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth' : np.arange(2,6, 2),
    'min_samples_split': np.arange(2, 8, 2)
}

# make a GridSearchCV instance
grid = GridSearchCV(RandomForestClassifier(), param_grid)

#fit the grid
grid.fit(X_s, y);

best = grid.best_params_
print(best)

#Plot with optimised params (2)
model1 = RandomForestClassifier(n_estimators=63, criterion = 'gini', max_depth = 4 ,
                                min_samples_split = 6, bootstrap = False, min_samples_leaf = 2)

# Train the model on the entire dataset
model1.fit(X_s, y)

# # Predict on the test set
y_pred = model1.predict(X_test)

# Calc the metrics (3)
accuracy2 = accuracy_score(y_test, y_pred)
precision2 = precision_score(y_test, y_pred, pos_label=0)
recall2 = recall_score(y_test, y_pred, pos_label=0)
f1_2 = f1_score(y_test, y_pred, pos_label=0)

mat = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(9, 9))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('true label')
plt.ylabel('predicted label');

# compare data from default and optimised model
print(f' Accuracy ratio is {accuracy2 / accuracy}')
print(f' Precision ratio is {precision2 / precision}')
print(f' Recall ratio is {((recall2 / recall)-1)*100} more accurate')
print(f' F1 ratio is {f1_2 / f1}')

#Question 4
from Learn_Curve import learning_curve_plot

rfc_class = RandomForestClassifier(n_estimators=63, criterion = 'gini', max_depth = 4 ,
                                min_samples_split = 6, bootstrap = False, min_samples_leaf = 2)

learning_curve_plot(X_s, y, rfc_class ,15)
