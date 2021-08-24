import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')

dataset= pd.read_csv(r'C:\Users\hp\Desktop\ML2\HeartDisease.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:, 2].values

#training testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#fitting K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors = 5 , metric ='minkowski', p=2)
classifier.fit(X_train,Y_train)

#predicting the results
Y_pred = classifier.predict(X_test)

#knn confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#forest classifier
from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_jobs=100)
rfr.fit(X_train,Y_train)
pred_rfr=rfr.predict(X_test)

#Linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
l_pred=reg.predict(X_test)

#dataset of x axis
X = np.linspace(0, 10)
fig, ax = plt.subplots()

for n in range(-20,30,10):
    ax.plot(X, np.cos(X) + np.random.randn(50) + n)

ax.set_title("Original-Data-On-X")

plt.show()

#dataset of y axis
Y = np.linspace(0, 10)
fig, ax = plt.subplots()

for n in range(-20,30,10):
    ax.plot(Y, np.cos(Y) + np.random.randn(50) + n)

ax.set_title("Original-Data-On-Y")

plt.show()

#best way of plotting
fig,ax =plt.subplots(figsize=(15,5))
a,=ax.plot(Y_test,color='red',lw=2,ls='--')
b,=ax.plot(Y_pred,color='brown',lw=2,ls='--')
c,=ax.plot(pred_rfr,color='blue',lw=2,ls='--')
d,=ax.plot(l_pred,color='green',lw=2,ls='--')
plt.legend([a,b,c,d],['Original','Knn','RandomForest','linear'])
plt.xlabel('HeartDisease-Peryear')
plt.ylabel('Patients')
plt.title('HeartDisease-dataset') 





