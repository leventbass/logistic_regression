import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def GradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations,1))

    for i in range(iterations):
        theta = theta - (alpha/m) * (X.T @ (sigmoid(X @ theta) - y)) 
        J_history[i] = ComputeCost(X, y, theta)

    return (J_history, theta)


def ComputeCost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def predict(X, theta):
    return np.round(sigmoid(X @ theta))

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)


sns.scatterplot(X[:,0],X[:,1],hue=y);


y = y[:,np.newaxis]

m = len(y)

mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X-mu) / sigma

X = np.hstack((np.ones((m,1)),X))
n = np.size(X,1)
theta = np.zeros((n,1))

iterations = 1500
alpha = 0.03

InitialCost = ComputeCost(X, y, theta)

print("Initial Cost is: {}".format(InitialCost))

(J_history, theta_optimal) = GradientDescent(X, y, theta, alpha, iterations)

print("Optimal Theta is: ", theta_optimal)

plt.figure()
plt.plot(range(len(J_history)), J_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.legend(("alpha: 0.01", "alpha: 0.001", "alpha: 0.1"))
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

y_pred = predict(X, theta_optimal)
score = accuracy_score(y, y_pred)
print(score)

