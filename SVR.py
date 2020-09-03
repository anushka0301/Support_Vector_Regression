#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
Y=dataset.iloc[:, 2:3].values

# Splitting the dataset into Training and Test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)


#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X, Y)

#Predicting a new result
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing SVR results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truff or Bluff(SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()
