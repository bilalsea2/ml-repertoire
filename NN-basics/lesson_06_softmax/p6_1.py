# Softmax
import numpy as np

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_output)

norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True) #since there are multiple batches, we can't have sum of all batches, numpy is dimension-sensitive

print(norm_values)