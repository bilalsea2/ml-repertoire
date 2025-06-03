# Initializing layers using class

import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense: # initializing layers using class
    def __init__(self, n_inputs, n_neurons):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons) #we don't have to transpose
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)