import numpy as np
from scipy.special import softmax
inputs = [[0.023591,0.668172,0.317972,0.260201],[0.332133,0.199377,0.512162,0.839625],[0.532121,0.513474,0.642476,0.787744],[0.000732,0.178381,0.405774,0.311350],]

weights = [[0.725516,0.908170,0.131382,0.508957,0.651814],[0.644276,0.161168,0.355541,0.353099,0.670797],[0.129795,0.986572,0.804437,0.418592,0.176916],[0.186712,0.540727,0.115268,0.621998,0.490280]]

outputs = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0], [0,0,0,1,0]]
outputs_flattened = [0, 1, 2, 3]
learning_rate = 0.001
for i in range(10):
    print(f"Epoch {i+1}")
    result = np.dot(inputs, weights)
    print(result)
    result = softmax(result, axis=1)
    print(result)
    predicted = np.argmax(result, axis=1)
    print(predicted)
    loss = np.array(outputs) - result
    print(loss)
    cross_entropy = np.sum(np.log(np.abs(loss)))
    print(cross_entropy)
    update = np.dot(np.transpose(inputs), loss)
    update *= learning_rate
    weights -= update
    print("Updated weights")
    print(weights)