# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:50:34 2024

@author: balra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

dataset=pd.read_csv(r"C:\Users\balra\Desktop\2024\kc_house_data.csv\kc_house_data.csv")
spaces=dataset['sqft_living15']
price=dataset['price']

x=np.array(spaces).reshape(-1,1)
y=np.array(price)

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3 , random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

pred=regressor.predict(x_test)


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('values for training dataset')
plt.xlabel('Spaces')
plt.ylabel('Price')
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('values for Test dataset')
plt.xlabel('Spaces')
plt.ylabel('Price')
plt.show()
