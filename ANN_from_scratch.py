"""
Created on Sun Sep 22 20:22:28 2018

@author: yusuf
"""
import numpy as np
np.random.seed(1)

'''
    This is an implementation of  two-layer neural network.
================= LINEAR->RELU->LINEAR->SIGMOID ===================

Initially I have declared all the required function:
    1. sigmoid: To calculate sigmoid of any numerical value.
    2. relu: To apply relu function on any number.
    3. sigmoid_backward: sigmoid function for the back propogation.
    4. relu_backward: relu function for the back propogation.
    5. initialize_parameters: It initialises the weights and biases for input, output and hidden layers.
    6. forward_propagation: it implements forward propogation of a neural network. It has been implemented 
       using for loops instead of vector multiplication. Alternative way has also been given but commented.
       LINEAR -> RELU -> LINEAR -> SIGMOID. 
    7. calculate_loss: It computes the cross-entropy cost.
    8. backward_propagation: Gradient Descent has been implemented in this function.
    9. update_parameters: This function update parameters after every iteration.
    10.build_model: This function will train our neural network.
    11.predict: This will predict the class of test values.
'''
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)    
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0        
    return dZ

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2}
    
    return parameters

def forward_propagation(A_prev, W, b, activation):
    
    Z = np.dot(W, A_prev) + b  
'''
    # Used for loop instead of vectorisation method.
    Z = np.zeros((W.shape[0], A_prev.shape[1]))
    for i in range(W.shape[0]):
        for j in range(A_prev.shape[1]):
            for k in range(A_prev.shape[0]):
                Z[i,j] += W[i,k]* A_prev[k, j]

    # Adding bias "b"            
    for i in range(W.shape[0]):
        for j in range(A_prev.shape[1]):
            Z[i,j] += b[i][0]
'''        
        
    linear_cache = (A_prev, W, b)
    
    if activation == "sigmoid":        
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":        
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache

def calculate_loss(AL, Y):

    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    cost = np.squeeze(cost)
    
    return cost

def backward_propagation(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":       
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, linear_cache[0].T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(linear_cache[1].T, dZ)
    
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
        
    return parameters

def build_model(X, Y, layer_dimention, learning_rate=0.005, num_iterations=2500, print_flag=False):
    
    grads = {}
    costs = []                              
    (n_x, n_h, n_y) = layer_dimention
    
    # Initializing parameters.
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):
        
        # Forward propagation: 
        A1, cache1 = forward_propagation(X, W1, b1, 'relu')
        A2, cache2 = forward_propagation(A1, W2, b2, 'sigmoid')
        
        # Calculate cost
        cost = calculate_loss(A2, Y)
                
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. 
        dA1, dW2, db2 = backward_propagation(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = backward_propagation(dA1, cache1, 'relu')
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update the parameters.        
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_flag and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_flag and i % 100 == 0:
            costs.append(cost)

    return parameters

def predict(X, y, parameters):
  
    m = X.shape[1]
    p = np.zeros((1,m))

    A_prev = X 
    A, cache = forward_propagation(A_prev, parameters['W1'], parameters['b1'], activation = "relu")   
    AL, cache = forward_propagation(A, parameters['W2'], parameters['b2'], activation = "sigmoid")
    
    # convert probabilities to 0 or 1 
    for i in range(0, AL.shape[1]):
        if AL[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == y)/m)))
    
    return AL

# Load very popular dataset of catvsnon-cat for the testing purpose.
from load_dataset import load_data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288    # length of input features # num_px * num_px * 3
n_h = 7        # Length of hidden layer
n_y = 1        # Length of output layer 
layer_dimention = (n_x, n_h, n_y)

#Build the model
parameters = build_model(train_x, train_y, layer_dimention = (n_x, n_h, n_y), num_iterations = 2000, print_flag=True)
#Prediction on the traininig data set
predictions_train = predict(train_x, train_y, parameters)
#Prediction on the test data set
pred_test = predict(test_x, test_y, parameters)
