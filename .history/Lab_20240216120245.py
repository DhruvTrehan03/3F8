import numpy as np
import matplotlib.pyplot as plt


def logistic(x): 
    return 1.0 / (1.0 + np.exp(-x))

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

def log_likelihood (y, X, B):
    L= np.prod(np.power(sigmoid(X@B),(y)) * np.power(sigmoid(X@B),(1-y)))
    # for i in range(len(X[:,0])):
    #     L *= (sigmoid(B.T@X[i,:].T)**y[i])*(sigmoid(-B.T@X[i,:].T)**(1-y[i]))
    LL = np.log(L)
    return LL

def predict(X_tilde, w): 
    return logistic(np.dot(X_tilde, w))

def get_x_tilde(X): 
    return np.concatenate((np.ones((X.shape[ 0 ], 1 )), X), 1)

def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()

def compute_average_ll(X_tilde, y, w):
    output_prob = predict(X_tilde, w)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))

def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    w = np.random.randn(X_tilde_train.shape[ 1 ])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)
        w += alpha * X_tilde_train.T@(y_train-sigmoid_value)     # Gradient-based update rule for w. To be completed by the student
        ll_train[ i ] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[ i ] = compute_average_ll(X_tilde_test, y_test, w)
        print(ll_train[ i ], ll_test[ i ])

    return w, ll_train, ll_test

def Grad_ascent (Max_iterations, Eta):
    B = np.ones(len(X_train[0,:]))
    Counter = 0
    ll=[]
    while Counter != Max_iterations:
        for i in range (len(X_train[:,1])):
            B += Eta * grad_log_likelihood(y_train, X_train, B)
            ll.append(log_likelihood(y_train, X_train, B))
        Counter += 1
    plot_ll(ll)
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

Eta = 0.1
Max_it = 100

X_tilde_train = get_x_tilde(X_train)
X_tilde_test = get_x_tilde(X_test)
w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, Max_it, Eta)
plot_ll(ll_train)
plot_ll(ll_test)
