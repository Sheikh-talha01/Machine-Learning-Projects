# This repo includes all the code 

# Import Dependenciess
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Import Dataset
dataset_ = pd.read_csv( 'sonar data.csv ', header = None)

dataset_.head()
dataset_.tail()
dataset = pd.DataFrame(dataset_)
dataset.head()

# Checking Null values
dataset.isnull().sum()
dataset.shape

dataset[60].value_counts()
dataset.groupby(60).mean()

dataset.describe()
dataset.info()

# Separating data and labels
x = dataset.drop(columns = 60, axis = 1 )
y = dataset[60]
x.head()
y.head()

# Training and Testing Data
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.1, stratify = y, random_state = 1 )
print( x.shape, x_train.shape, x_test.shape )

# Model Training
model = LogisticRegression()
model.fit( x_train, y_train )

# Checking Accuracy Score on Training Data
x_train_prediction = model.predict( x_train )
training_data_accuracy = accuracy_score( x_train_prediction , y_train )
print( "Accuracy Score of Training data is:", training_data_accuracy )

# Checking Accuracy Score on Testing Data


x_test_prediction = model.predict( x_test )
testing_data_accuracy = accuracy_score( x_test_prediction , y_test )
print( "Accuracy Score of Testing data is:", testing_data_accuracy )

# Predicting System
input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032,)
# input data as numpy array
input_data_as_array = np.asarray( input_data)
# reshaping 
input_data_reshape = input_data_as_array.reshape ( 1,-1)
# Prediction
prediction = model.predict(input_data_reshape)
print( prediction)

if ( prediction[0] == 'M' ):
   print( "Object is Mine" )
else:
  print( "Object is Rock" )

