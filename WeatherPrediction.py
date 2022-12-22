import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#% matplotlib inline

weather = pd.read_csv('C:/Users/91967/OneDrive/Desktop/Machinelearning/weather.csv')
print(weather.shape)
print(weather.describe())

weather.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

plt.figure(figsize=(10,10))
plt.tight_layout()
sns.histplot(weather['MaxTemp'])
plt.show()

#Datasplicing
x = weather['MinTemp'].values.reshape(-1,1)
y = weather['MaxTemp'].values.reshape(-1,1)

X_train , X_test , Y_train , Y_test= train_test_split(x,y,test_size=0.2,random_state=0)


regressor= LinearRegression()
regressor.fit(X_train,Y_train) #taining the algorithm

print('Intercept:',regressor.intercept_)
print('Coefficient:',regressor.coef_)

Y_pred=regressor.predict(X_test)

df=pd.DataFrame({'Actual':Y_test.flatten(),'Predicted':Y_pred.flatten()})
print(df)

df1= df.head(25)
df1.plot(kind='bar', figsize=(10,10))
plt.grid(which='major', linestyle='-',linewidth=0.5,color='green')
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.show()

plt.scatter(X_test,Y_test,color='gray')
plt.plot(X_test,Y_pred,color='red',linewidth=2)
plt.show()


print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(Y_test,Y_pred))
print('Root Mean Squared Error:',np.sqrt( metrics.mean_squared_error(Y_test,Y_pred)))
