# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing Data
data = pd.read_csv("http://bit.ly/w-data")

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  

# Splitting Train and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Fitting Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1, 1)
regressor.fit(X_train, y_train)

# Predicting the Test set results
X_test = X_test.reshape(-1,1)
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Hours of Study vs Score (TRAIN)")
plt.xlabel("Hours of Study"), plt.ylabel("Score")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Hours of Study vs Score (TEST)")
plt.xlabel("Hours of Study"), plt.ylabel("Score")
plt.show()

print("Train Set score:", regressor.score(X_train, y_train))
print("Test Set score:", regressor.score(X_test, y_pred))
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

# Predict Score for 9.25 hrs of study
print("Predicted score for 9.25 hours of studying is:", regressor.predict([[9.25]]))


