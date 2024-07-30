# This repo contains all the code of Diabetes Prediction Model

# Import Dependencies
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('diabetes.csv')

dataset.head()
dataset.shape
dataset.describe()  

print( dataset['Outcome'].value_counts() )

# Printing Mean of Outcome
dataset.groupby('Outcome').mean()

diabetes_data = dataset.drop( columns = 'Outcome', axis = 1 )
target = dataset['Outcome' ]

diabetes_data.head()
target.head()

# Count Plot
sns.countplot(x = 'Outcome',hue = 'Outcome', data = dataset)
# Pairplot 
sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()

# Standardizing data
scaler = StandardScaler()
scaler.fit( diabetes_data )
diabetes_data.standardized = scaler.transform(diabetes_data)
