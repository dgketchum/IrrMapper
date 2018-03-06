import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from sklearn.metrics import confusion_matrix, accuracy_score


def softmax_regression(data, binary_true=None):



    N = len(classes)
    d, d_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                            random_state=42)
    d = normalize(d)
    d_test = normalize(d_test)

    X = np.column_stack((np.ones_like(y), d))
    m = X.shape[0]
    n = X.shape[1]

    T = np.zeros((m, N))
    for t, yi in zip(T, y):
        t[yi] = 1

    N_iterations = 10

    eta = 0.001
    W = 0.1 * np.random.randn(n, N)

    for i in range(N_iterations):
        print(i)
        W -= eta * _gradient(X, W, T, m, N)

    y_pred = np.argmax(_softmax(X, W, N), axis=1)
    print('Objective Function Value: ', j_function(X, W, T),
          'Total misclassified: ', sum(y != y_pred))
    print(confusion_matrix(y_pred, y))
    print(accuracy_score(y, y_pred))

    X = np.column_stack((np.ones_like(y_test), d_test))

    y_test_pred = np.argmax(_softmax(X, W, N), axis=1)

    print(confusion_matrix(y_test_pred, y_test))
    print(accuracy_score(y_test, y_test_pred))


def j_function(X, W, T):
    return -np.sum(np.sum(T * np.log(_softmax(X, W, T)), axis=1), axis=0)


def _gradient(X, W, T, m, N):
    return -np.column_stack(
        [np.sum([(T - _softmax(X, W, N))[i, k] * X[i] for i in range(m)], axis=0) for k in range(N)])


def _softmax(X, W):
    a = np.dot(X, W)
    return np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)
