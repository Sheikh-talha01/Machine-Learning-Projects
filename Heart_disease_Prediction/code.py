# Heart Disease Prediction

# Import Dependies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Importing Dataset
dataset = pd.read_csv('heart.csv ')

dataset.head()
dataset.tail()

# Statistical Approch
dataset.info()
dataset.describe()
dataset.shape
dataset['target'].value_counts()
dataset.groupby('target').mean()
dataset.isnull().sum()

# CountPlot
sns.countplot( x = dataset['target'], hue = dataset['target'], data = dataset)

# Heatplot
sns.heatmap( data = dataset )

# Pairplot
sns.pairplot( data = dataset )

# BarPlot
sns.barplot( x = dataset['target'], y = dataset['age'] , data = dataset )

# Splitting dataset into data and label
x = dataset.drop( columns = 'target', axis = 1)
y = dataset['target']

x.head()

# Training Testing Data and Testing Data
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, stratify = y, random_state = 42 )
print( x.shape, x_train.shape, x_test.shape )

model = LogisticRegression()
model.fit( x_train, y_train )

x_train_prediction = model.predict( x_train )
train_accuracy_score = accuracy_score( x_train_prediction, y_train )
print( 'Accuracy Score of Training Data is: ', train_accuracy_score )

x_test_prediction = model.predict( x_test )
test_accuracy_score = accuracy_score( x_test_prediction, y_test )
print( 'Accuracy Score of Testing Data is: ', test_accuracy_score )

# Prediction System
input_data = ( 52,1,0,125,212,0,1,168,0,1,2,2,3 )
data_as_an_array = np.asarray( input_data )
reshape_input_data = data_as_an_array.reshape( 1, -1)
prediction = model.predict( reshape_input_data )
print( prediction )

if( prediction[0] == 0 ):
    print('The Person does not have Heart Disease' )
else:
    print('The Person has Heart Disease')

