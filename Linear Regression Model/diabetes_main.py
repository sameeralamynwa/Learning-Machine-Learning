import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.keys())

# It contains information about all the variables or attributes (age, sex, body mass index, etc) and other relevant information
# print(diabetes.DESCR)

# A matrix containing all the values for the attributes of all the patients. To calculate the number of attributes we can print len(diabetes.data)
# print(diabetes.data)

# Accessing the values of the column with index 2 in the form of a matrix
diabetes_x1 = diabetes.data[:, np.newaxis, 2]

# print(diabetes_x1)

# Will be used for training our model (Last 30)
diabetes_x1_train = diabetes_x1[:-30]

# Will be used for testing our model (First 30)
diabetes_x1_test = diabetes_x1[-30:]

# Slicing should be the same as that of diabetes_x1_train
diabetes_y1_train = diabetes.target[:-30]

# Slicing should be the same as that of diabetes_x1_test
diabetes_y1_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

# Fitting the curve into a line (Model Prepared)
model.fit(diabetes_x1_train, diabetes_y1_train)

# Predicting the value (Testing of our model)
diabetes_y1_predicted = model.predict(diabetes_x1_test)

# Chosing an appropriate metric to calculate the deviation of our prediction from the original data
print("Mean squared error is ", mean_squared_error(diabetes_y1_test, diabetes_y1_predicted))

# Slope of the line that fitted the curve
# print("Weight or Slope :", model.coef_)

# Intercept of the line that fitted the curve
# print("Intercept :", model.intercept_)

# Plotting the curve using matplotlib.pyplot

# Scatter plot (Use of dots to represent the values)
plt.scatter(diabetes_x1_test, diabetes_y1_test)

# Plotting a line that passes almost in between all the scattered points reducing the errors.
plt.plot(diabetes_x1_test, diabetes_y1_predicted)

plt.show()

# Till now we had only considered one attribute that was at index 2. So our line was y(x1) = w0 + w1 * x1
# If we were to consider two attributes, our line would look like y(x1, x2) = w0 + w1 * x1 + w2 * x2
# Similarly we can take any number of attributes, we will observe that as we increase the number of attributes our
# mean squared error will keep on decreasing.

# To get the corresponding values of w1, w2, w3 .. n. We can find using the mode.coef_.
# w0 is the intercept which can be found using the model.intercept_.

