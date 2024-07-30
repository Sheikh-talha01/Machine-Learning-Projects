# This repo contains all the code of Diabetes Prediction Model
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
