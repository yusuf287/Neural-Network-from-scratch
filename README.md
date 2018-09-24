    This is an implementation of  two-layer neural network.
================= LINEAR->RELU->LINEAR->SIGMOID ===================

Following are the required functions I have used to implement ANN:

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
