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

def plot_predictive_distribution(X, y, w, map_inputs = lambda x : x):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict(X_tilde, w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
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

def evaluate_basis_functions(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)

def conf_mat(y_test, X_tilde_test, w):
    z = predict(X_tilde_test, w)>0.5
    y_test = y_test>0.5
    P_0_0 = 0
    P_1_0 = 0
    P_0_1 = 0
    P_1_1 = 0
    for i in range(len(y_test)):
        if z[i] == 0 and y_test[i] == 0:
                print(z[i], y_test[i])
                P_0_0 += 1
        elif z[i] == 1 and y_test[i] == 0:
                print(z[i], y_test[i])
                P_1_0 += 1
        elif z[i] == 0 and y_test[i] == 1:
                print(z[i], y_test[i])
                P_0_1 += 1
        elif z[i] == 1 and y_test[i] == 1:
                print(z[i], y_test[i])
                P_1_1 += 1
    print(np.array([[P_0_0, P_1_0],[P_0_1, P_1_1]]))
    print(np.array([[P_0_0/(P_0_0+P_1_0), P_1_0/(P_0_0+P_1_0)],[P_0_1/(P_0_1+P_1_1), P_1_1/(P_0_1+P_1_1)]]))

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

Eta = 0.001
Max_it = 100

X_tilde_train = get_x_tilde(X_train)
X_tilde_test = get_x_tilde(X_test)
w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, Max_it, Eta)
plot_ll(ll_train)
plot_ll(ll_test)
plot_predictive_distribution(X, y, w)
conf_mat(y_test, X_tilde_train, w)

# l = 1 # Width of the Gaussian basis funcction. To be completed by the student

# X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
# X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

# # We train the new classifier on the feature expanded inputs

# alpha = 0.0001 # Learning rate for gradient-based optimisation with basis functions. To be completed by the student
# n_steps = 1000 # Number of steps of gradient-based optimisation with basis functions. To be completed by the student

# w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

# # We plot the training and test log likelihoods

# plot_ll(ll_train)
# plot_ll(ll_test)

# # We plot the predictive distribution

# plot_predictive_distribution(X, y, w, lambda x : evaluate_basis_functions(l, x, X_train))
# conf_mat(y_test, X_tilde_train, w)