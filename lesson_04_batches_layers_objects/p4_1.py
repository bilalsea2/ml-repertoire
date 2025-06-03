# Batches
import numpy as np

# added batches - set of inputs, helps to calculate faster. each inner array represents one batch

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

# np.array(weights).T - transposing (swapping rows and columns) weight matrix so its shape
# matches the "inputs" matrix when multiplying matrices
output = np.dot(inputs, np.array(weights).T) + bias 

# layer 2
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

bias2 = [-1, 2, -0.5]

output2 = np.dot(output, np.array(weights2).T) + bias2

print(output, '\n', output2)