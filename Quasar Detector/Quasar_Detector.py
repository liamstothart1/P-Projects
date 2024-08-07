import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from NeuralN import NeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from nose.tools import assert_almost_equal
from data_engineer import histo_feature_plot
from data_engineer import feature_standardisation_test


# Load data
df = pd.read_csv('data/SDSS_galaxies.csv')

# Calculate colour bands 
df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']
df['r-i'] = df['r'] - df['i']
df['i-z'] = df['i'] - df['z']

# Set target string 
df['target'] = (df['specClass'] == 'QSO').astype(int)
# we want to check that the results are binary 
df['target'].head()

# split into Features and target
features = ['u-g', 'g-r', 'r-i', 'i-z']
target = 'target'

# Standardise features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


histo_feature_plot(df[features], ['u-g', 'g-r', 'r-i', 'i-z'])
print(df[features].head())

feature_standardisation_test(df[features], features)
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Instantiate the network
nn = NeuralNetwork([4, 16, 10, 2], activation_function=['relu'])

# Correcting the one-hot encoding to use resampled labels for training
y_train_nn = np.eye(2)[y_train_smote]

# Correcting the one-hot encoding for the test labels
y_test_nn = np.eye(2)[y_test]  # Convert to one-hot for the neural network

# Training the network with resampled data
nn.SGD(X_train_smote.values, y_train_nn, X_test.values, y_test_nn, epochs=80, eta=0.001)
from NeuralN import validation_curve

# define variables for validation curve from the test and train scores from the neural network

train_data = nn.train_scores
test_data = nn.test_scores


validation_curve(train_data, test_data)
from RFC_default import rfc_default
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# a dictionary of parameter names to search, and the values to try out
param_grid = {
    'n_estimators': np.arange(9, 15, 3),
    'max_depth' : np.arange(9,15, 3),
    'min_samples_split': np.arange(5, 8, 1)
}


# make a GridSearchCV instance
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

# fit it
grid.fit(X_train, y_train)

# Get the best parameters
best = grid.best_params_

# Test performance 
model1 = RandomForestClassifier(**best)
model1.fit(X_train, y_train)

predict = model1.predict(X_test)
accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict, average='weighted')
recall = recall_score(y_test, predict, average='weighted')
f1 = f1_score(y_test, predict, average='weighted')

# confusion matrix
conf_matrix = confusion_matrix(y_test, predict)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

plt.show()

print('Accuracy = ', accuracy)
print('Precision = ', precision)
print('Recall = ', recall)
print('F1 Score = ', f1)
print('Best Parameters = ', best)
