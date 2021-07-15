from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Prints the keys of our dataset : dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(iris.keys())

# To know the corresponding class associated with the labels : print(iris['DESCR']) or print(iris.DESCR)
# 0 - Setosa
# 1 - Versicolour
# 2 - Virginica
# print(iris.DESCR)

# The dimension of our dataset (150, 4) : 150 rows and 4 columns
# print(iris['data'].shape)

# Storing the petal width in a matrix. We will be using this to build our Logistic Regression Model
# We will be predicting if the label is 2 or not (2 corresponds to Iris-Virginica)
X = iris.data[:, 3:]

# Storing the labels of all the rows if the flower is Virginica or not
# Y = (iris.target == 2)

# Y is an array of boolean type if we declared as above. We will convert it to int using numpy because we will dea with numbers only.
Y = (iris.target == 2).astype(np.int32)

# Training a logistic regression classifier

CLF = LogisticRegression()

# Fitting the classifier according to the data
# X is the actual parameters that we want to use to describe our model (width of petal)
# Y is of boolean type. Used to predict the class of the flower (1 is it is Virginica else 0)
CLF.fit(X, Y)

# 1 is the iris is Virginica else 0
# 1.6 is the petal width
prediction = CLF.predict([[1.6]])

# Getting the corresponding class of the data (1 if it is Virginica else 0)
# This is because it is a binary classification.
# print(prediction)

# Visualising our Logistic Regression Model using matplotlib

# A one dimensional array containing equally spaced points between 0 and 3 included
z = np.linspace(0, 3, 1000)

# Since the the input to CLF.predict_proba() is a 2 dimensional array.
# We are reshaping the array. It works the same if we write z.reshape(10, 1) that is 10 x 1 is the dimension of our matrix
x = z.reshape(-1, 1)

# So far we used CLF.predict() to predict the class. But Logistic Regressions classifies the data based on probabilities.
# So here we will find out the actual value of the probability of our data to be predicted and will also try to plot this.

y_probability = CLF.predict_proba(x)

# Corresponding probabilities of the input
# print(y_probability)

plt.plot(x, y_probability[:,1], "g-", label = "Iris-Virginica")

# The plot is very similar to the graph of the sigmoid function. A typical S shaped graph.
# Also as we increased our input dataset from n = 10 to n = 1000, a significant change occurs in the graph in which the graph changes more towards the graph of sigmoid function.
plt.show()
