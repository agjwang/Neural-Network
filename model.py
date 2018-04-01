import numpy as np
from activation import sigmoid, relu, tanh

class NeuralNet:
    def __init__(self, layer_dimensions, epsilon = 0.01, activation_function_name = 'relu', iterations = 10000,
                 print_cost = False):
        self.layer_dimensions = layer_dimensions
        self.epsilon = epsilon
        self.activation_function_name = activation_function_name
        self.iterations = iterations
        self.print_cost = print_cost
        self.model = None

    def build_model(self, X, Y):
        training_set_size = len(X)

        parameters = self.initialize_params()
        print(parameters)

        for i in range(self.iterations):
            x = 0
            # forward propagation
            # back propagation
            # update
            # print_cost

        return parameters

    def initialize_params(self):
        np.random.seed(0)
        parameters = {}
        layer_dims = self.layer_dimensions

        for i in range(1, len(layer_dims)):
            parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 1
            parameters['W' + str(i)] = np.zeros((layer_dims[i], 1))

        return parameters

    def update_params(self, parameters, gradients):

        for i in range(1, len(self.layer_dimensions)):
            parameters['W' + str(i)] -= self.epsilon * gradients['dW' + str(i)]
            parameters['b' + str(i)] -= self.epsilon * gradients['db' + str(i)]

        return parameters

    def forward_propagation(self, X, parameters):

    def back_propagation(self, parameters, probs):

    def calculate_cost(self):

    def predict(self, x):

    def activation_function(self, Z):
        if self.activation_function_name == 'relu':
            relu(Z)
        elif self.activation_function_name == 'sigmoid':
            sigmoid(Z)
        else:
            tanh(Z)
