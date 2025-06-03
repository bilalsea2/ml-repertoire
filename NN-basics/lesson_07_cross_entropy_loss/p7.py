# Categorical cross-entropy / Softmax Loss
# L=-sum(y*log(yhat)) 
# where y is one-hot vector (model predicts one label and ideally the result for label is 1.0) 
# and yhat is probability distribution calculated by our model
# since we are using one-hot vector, L=-log(yhat)

import math 

softmax_output = [0.7, 0.1, 0.2]
target = [1, 0, 0]

# pretty simple implementation, just for introduction of general softmax loss
loss = -target[0]*math.log(softmax_output[0]) - target[1]*math.log(softmax_output[1]) - target[2]*math.log(softmax_output[2])

print(loss)