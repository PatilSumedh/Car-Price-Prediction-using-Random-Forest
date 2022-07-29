#importing required libraries

import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import r2_score

#Load Dataset

dataset = pd.read_csv('dataset.csv')
dataset = dataset.drop(['car_ID'],axis=1)
dataset

#Summarize Dataset

print(dataset.shape)
print(dataset.head(5))

#Splitting Dataset into X & Y
#This X contains Both Numerical & Text Data

Xdata = dataset.drop('price',axis='columns')
numericalCols = Xdata.select_dtypes(exclude=['object']).columns
X = Xdata[numericalCols]
X

Y = dataset['price']
Y

#Scaling the Independent Variables (Features)

cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X

#Splitting Dataset into Train & Test

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

#Training using Random Forest

clf = RandomForestRegressor()
clf.fit(X_train, Y_train)

ypred = clf.predict(X_test)

r2score = r2_score(Y_test,ypred)
print("R2Score",r2score*100)

import numpy as np

r_data = (1.743470,-1.690772,-0.426521,-0.844782,-2.020417,-0.014566,0.074449,0.519071,-1.839377,-0.288349,0.174483,-0.262960,-0.646553,-0.546059)
#input_data = tuple(map(float,input("Enter Data:").split(',')))

input_arr = np.asarray(r_data)
#print(input_data)

#reshape the array as we are predicting the output for one instance
reshape_data = input_arr.reshape(1,-1)

#final prediction 

prediction = clf.predict(reshape_data)

print(prediction) 
