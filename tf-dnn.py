from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=50000,
    n_features=8,
    n_informative=8,
    n_redundant=0,
    n_clusters_per_class=2,
    random_state=26,
)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

colors = ["black", "yellow"]
cmap = matplotlib.colors.ListedColormap(colors)
# Plot the figure
plt.figure()
plt.title("Non-linearly separable classes")
plt.scatter(X[:, 0], X[:, 1], c=y, marker="o", s=50, cmap=cmap, alpha=0.5)
plt.savefig("fig1.png", bbox_inches="tight")


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.25, random_state=25
)
print("shape of X_train:{} shape 0f Y_train:{}".format(X_train.shape, Y_train.shape))

X_train = X_train.T
Y_train = Y_train.reshape(1, len(Y_train))
X_test = X_test.T
Y_test = Y_test.reshape(1, len(Y_test))
print(
    "shape of X_train:{} shape 0f Y_train:{} after transformation".format(
        X_train.shape, Y_train.shape
    )
)


import tensorflow as tf


def placeholders(num_features):
    A_0 = tf.placeholder(dtype=tf.float64, shape=([num_features, None]))
    Y = tf.placeholder(dtype=tf.float64, shape=([1, None]))
    return A_0, Y


layer_dims = [8, 8, 8, 1]


def initialize_parameters_deep(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = tf.Variable(
            initial_value=tf.random_normal(
                [layer_dims[l], layer_dims[l - 1]], dtype=tf.float64
            )
            * 0.01
        )
        parameters["b" + str(l)] = tf.Variable(
            initial_value=tf.zeros([layer_dims[l], 1], dtype=tf.float64) * 0.01
        )
    return parameters


def linear_forward_prop(A_prev, W, b, activation):
    Z = tf.add(tf.matmul(W, A_prev), b)
    if activation == "sigmoid":
        A = Z
    elif activation == "relu":
        A = tf.nn.relu(Z)
    return A


def l_layer_forwardProp(A_0, parameters):
    A = A_0
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A = linear_forward_prop(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu"
        )
    A = linear_forward_prop(
        A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid"
    )


def deep_model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate, num_iter):
    num_features = layer_dims[0]
    A_0, Y = placeholders(num_features)
    parameters = initialize_parameters_deep(layer_dims)
    Z_final = l_layer_forwardProp(A_0, parameters)
    cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final, labels=Y)
    )
    train_net = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_iter):
            _, c = sess.run([train_net, cost], feed_dict={A_0: X_train, Y: Y_train})
            if i % 1000 == 0:
                print(c)
        correct_prediction = tf.equal(tf.round(tf.sigmoid(Z_final)), Y)
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy on testset:", accuracy.eval({A_0: X_test, Y: Y_test}))


deep_model(X_train, Y_train, X_test, Y_test, layer_dims, 0.2, 10000)
