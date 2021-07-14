from sklearn import datasets
# The name of any classifier starts with a capital letter.
from sklearn.neighbors import KNeighborsClassifier

# Loading the dataset
iris = datasets.load_iris()

# To know the corresponding class associated with the labels
# 0 - Setosa
# 1 - Versicolour
# 2 - Virginica
# print(iris.DESCR)

# x1, x2, x3, ... xn (all the input for our model)
features = iris.data

# y (corresponding class to which it belongs)
labels = iris.target

# Training the classifier

# Creating the classifier
CLF = KNeighborsClassifier()

# Every classifier has a fir and predict function.
# Fit is used so that the classifier fits itself according to the data and predict is to predict the class of a new data

# Fitting the classifier
CLF.fit(features, labels)

# Predicting the class or label
predict = CLF.predict([[1, 1, 1, 1]]);

print(predict)

