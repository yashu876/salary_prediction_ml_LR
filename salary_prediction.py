import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

d = pd.read_csv("Salairy_Data(1).csv")
d = d.dropna()
d = pd.get_dummies(d,drop_first=True)
d.head(100)
d.info()
d.describe()
d.corr()

x = d.drop("Salary",axis=1)
y = d["Salary"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)

plt.scatter(y_test,pred)
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.title("Actual value vs Predicted value")
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()])
plt.show()

mse = mean_squared_error(y_test,pred)
rsme = np.sqrt(mse)
r2 = r2_score(y_test,pred)
print("mse:",mse)
print("rsme:",rsme)
print("r2:",r2)


