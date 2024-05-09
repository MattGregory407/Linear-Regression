import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle

# Reading in data from student .csv file

data = pd.read_csv("student-mat.csv", sep=";")

# Trimming data to necesssary data points

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Establishing prediciton(s)

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Splitting off 10% of data for testing purposes

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Finding best fit line

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

linear.score(x_test, y_test)

acc = linear.score(x_test, y_test)

print(acc)

print("Co:  \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])