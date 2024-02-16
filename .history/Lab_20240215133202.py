import numpy as np
import matplotlib.pyplot as plt

def plot_data_internal(X, y):
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

def Grad_ascent (Max_iterations, Eta):
    B = np.zeros(len(y_train))
    print(X_train[:, 0])
    print(np.shape(B), np.shape(y_train), np.shape(X_train[:, 0]))
    Counter = 0
    while Counter != Max_iterations:
        DL = 0
        for i in range (len(y_train)):
            DL += X_train[i] @ (y[i] - 1/(1-np.exp(B.T@X_train[:,i])))
            B += Eta * DL
            Counter += 1
    return B

X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
permutation = np.random.permutation(X.shape[ 0 ])
X = X[ permutation, : ]
y = y[ permutation ]

n_train = 800
X_train = X[ 0 : n_train, : ]
X_test = X[ n_train :, : ]
y_train = y[ 0 : n_train ]
y_test = y[ n_train : ]

Eta = 0.01
Max_it = 1000

print(Grad_ascent(Max_it, Eta))