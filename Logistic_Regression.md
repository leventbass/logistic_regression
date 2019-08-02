
# Logistic Regression from Scratch with NumPy
---

Welcome to another post of machine learning algorithm implementation! Today, the algorithm that will be implementing from scratch is **Logistic Regression**. After its sibling algorithm linear regression, this one is higly used in machine learning as well, due to its simplicity and robustness. Even though its called logistic *regression*, it's actually a classification algorithm that is used to classify input data into its classes (labels).
    
This powerful machine learning model can be used to answer some questions such as;
* Whether an e-mail is spam or not
* If the customer will churn
* Whether a tumor is benign or malignant

All of the questions above are simply yes-no questions, therefore they can be used to classify input data into two classes. Hence, the term binary classification is used when the data can be categorized into two distinct classes. Obviously, multi-class classification deals with data that has more than two labels (classes). After grasping the ins and outs of logistic regression to make binary classification, transition to a multi-class classification is pretty straight-forward, as a consequence, here, we will deal with data that has two classes only.
    
Remember, that in linear regression we predict numerical values based on the input values and parameters of the model.Here, in logistic regression we can also approach the model as we are trying to predict numbers but this time these numbers correspond to the probability values of input data belonging to a particular class. 

The term logistic in logistic regression is used because this time we are applying another function to the weighted sum of input data and paratemeters of the model and this function is  called logit (sigmoid) function. 

Sigmoid function always outputs values between 0 and 1, and thus can be used to calculate probabilities of input data belonging to a certain class:

$ sigmoid(x) =  \frac{\mathrm{1} }{\mathrm{1 + e^{-x}}}  $

<img src="img/sigmoid.png" width="400"/>
    
Without further ado, let's start writing the code for this implementation. I will explain the code as I go, whenever deemed necessary. 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
```

We start off by importing necessary libraries. As always, **NumPy** is the only package that we will use in order to implement the logistic regression algorithm. All the others will only help us with minor issues such as visualizing the data at hand or creating a dataset. Hence, we won't be using already implemented package solutions for logistic regression. 


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

Here, we write the code for the aforementioned sigmoid (logit) function. It is important to note that this function can be applied to all of the elements of a `numpy` array individually, simply because we make use of the exponential function from the **NumPy** package.


```python
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost
```

Next, we write the cost function for logistic regression. Note that the cost function used in logistic regression is different than the one used in linear regression. 

Remember, in linear regression we calculated the weighted sum of input data and parameters and fed that sum to the cost function to calculate the cost. When we plotted the cost function it was seen to be convex, hence a local minimum was also the global minimum.

However, in logistic regression, we apply sigmoid function to the weighted sum which makes the resulting outcome non-linear. If we feed that non-linear outcome to the cost function, what we get would be a non-convex function and we wouldn't be assured to find only one local minimum that is also the global minimum. 

As a result, we use another cost function to calculate the cost which is guaranteed to give one local minimum during the optimization.


```python
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)
```

Gradient descent implementation here is not so different than the one we used in linear regression. Only difference to be noted is the sigmoid function that is applied to the weighted sum.


```python
def predict(X, params):
    return np.round(sigmoid(X @ params))
```

While writing out the prediction function, let's not forget that we are dealing with probabilities here. Hence, if the resulting value is above 0.50, we round it up to 1, meaning the data point belongs to the class 1. Consequently, if the probability of a data point belonging to the class 1 is below 0.50, it simply means that it is part of the other class (class 0). Remember that this is binary classification, so we have only two classes (class 1 and class 0).


```python
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

y = y[:,np.newaxis]

sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=y.reshape(-1));


```


![png](img/output_11_0.png)


After coding out the necessary functions, let's create our own dataset with `make_classification` function from `sklearn.datasets`. We create 500 sample points with two classes and plot the dataset with the help of `seaborn` library.


```python
m = len(y)

X = np.hstack((np.ones((m,1)),X))
n = np.size(X,1)
params = np.zeros((n,1))

iterations = 1500
learning_rate = 0.03

initial_cost = compute_cost(X, y, params)

print("Initial Cost is: {} \n".format(initial_cost))

(cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)

print("Optimal Parameters are: \n", params_optimal, "\n")

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()
```

    Initial Cost is: [[0.69312718]] 
    
    Optimal Parameters are: 
     [[-0.45293068]
     [ 3.26552327]
     [ 0.03334871]] 
    



![png](img/output_13_1.png)


Now, let's run our algorithm and calculate the parameters of our model. Seeing plot, we can now be sure that we have implemented the logistic regression algorithm without a fault, since it decreases with every iteration until the decrease is so minimal that the cost converges to a minimum which is what we want indeed.


```python
y_pred = predict(X, params_optimal)
score = float(sum(y_pred == y))/ float(len(y))

print(score)
```

    0.966


After running the algorithm and getting the optimal parameters, we want to know how successful our model is at predicting the classes of our data. For this reason, we use `accuracy_score` function from `sklearn.metrics` to calculate the accuracy.


```python
slope = -(params_optimal[1] / params_optimal[2])
intercept = -(params_optimal[0] / params_optimal[2])

sns.set_style('white')
sns.scatterplot(X[:,1],X[:,2],hue=y.reshape(-1));

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");



```


![png](img/output_17_0.png)


Now, for the sake of visualization lets plot our dataset along with the decision boundary of our model. We simply calculate the intercept and slope values using the optimal parameters and plot the boundary that classifies the data into two classes. We can see from the plot that the classification is not 100% correct since the separation of classes is not linear naturally. Nevertheless, misclassified points (false positives and false negatives) are really few, so we have done a pretty good job with this implementation of logistic regression.
    
Feel free to go ahead and code all of this once again.
    
But this time, do it on your own.
