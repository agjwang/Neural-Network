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
        self.parameters = None

    def build_model(self, X, Y):
        assert(len(X) == len(Y))

        training_set_size = len(X)

        parameters = self.initialize_params()

        for i in range(self.iterations):
            # forward propagation
            A = self.forward_propagation(X, parameters)
            # back propagation
            gradients = self.back_propagation(A, parameters)
            # update
            parameters = self.update_params(parameters, gradients)
            # print cost
            if i % 500 == 0:
                print("Loss after iteration %i: %f" %(i, self.calculate_cost(Y, training_set_size, parameters)))

        self.parameters = parameters

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
        return 0

    def back_propagation(self, A, parameters):
        return 0

    # Calculate cost using cross entropy loss
    def calculate_cost(self, X, Y, parameters):
        training_set_size = len(X)
        total_cost = 0

        for i in range(training_set_size):
            a = X[i]

            for l in range(l, len(self.layer_dimensions)):
                a = self.activation_function(np.dot(parameters['W' + str(l)], a) + parameters['b' + str(l)])

            # Create vector where y[Y[i]]] is 1 and all other values are 0
            y = np.zeros_like(a)
            y[Y[i]] = 1

            log_loss = -np.sum(y.dot(a) + ((y * -1) + 1).dot(a))
            total_cost += np.sum(log_loss)

        return total_cost / training_set_size

    def predict(self, x):
        a = x
        for i in range(1, len(self.layer_dimensions)):
            a = self.activation_function(np.dot(self.parameters['W' + str(i)], a) + self.parameters['b' + str(i)])
        return np.argmax(a)

    def activation_function(self, Z):
        if self.activation_function_name == 'relu':
            return relu(Z)
        elif self.activation_function_name == 'sigmoid':
            return sigmoid(Z)
        else:
            return tanh(Z)
