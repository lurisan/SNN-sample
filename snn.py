from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
# extracting first 100 samples pertaining #to iris setosa and verginica
X = iris.data[:100, :4]
# actual output
Y = iris.target[:100]

X_norm = X.reshape(100, 4)
X_data = X_norm.T
Y_data = Y.reshape(1, 100)
print(X_data.shape)
print(Y_data.shape)


def initialiseNetwork(num_features):
    W = np.zeros((num_features, 1))
    b = 0
    parameters = {"W": W, "b": b}
    return parameters


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forwardPropagation(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return A


def cost(A, Y, num_samples):
    return -1 / num_samples * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))


def backPropagration(X, Y, A, num_samples):
    dZ = A - Y
    dW = (np.dot(X, dZ.T)) / num_samples
    db = np.sum(dZ) / num_samples
    return dW, db


def updateParameters(parameters, dW, db, learning_rate):
    W = parameters["W"] - (learning_rate * dW)
    b = parameters["b"] - (learning_rate * db)
    return {"W": W, "b": b}


def model(X, Y, num_iter, learning_rate):
    num_features = X.shape[0]
    num_samples = float(X.shape[1])
    parameters = initialiseNetwork(num_features)
    for i in range(num_iter):
        A = forwardPropagation(X, Y, parameters)
        if i % 100 == 0:
            print("cost after {} iteration: {}".format(i, cost(A, Y, num_samples)))
        dW, db = backPropagration(X, Y, A, num_samples)
        parameters = updateParameters(parameters, dW, db, learning_rate)
    return parameters


parameters = model(X_data, Y, 1000, 0.1)