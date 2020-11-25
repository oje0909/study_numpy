import numpy as np

timesteps = 100
input_features = 64
output_features = 128

inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

succ_output = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    succ_output.append(output_t)
    state_t = output_t      # state_t 가 다시 다음 for loop 에서 적용됨

output_t


