import numpy as np

timesteps = 100     # 입력 시퀀스의 time step 의 수 / -> unfold 된 수?
input_features = 64     # 입력 특성의 차원 (ex. 단어 1개당 64차원으로 input)
output_features = 128   # 출력 특성의 차원

inputs = np.random.random((timesteps, input_features))      # 입력 난수 생성 (100*64차원)
state_t = np.zeros((output_features, ))     # t 시점의 상태  : 초기는 0벡터로 세팅
print('state_t : ', state_t)        # ndarray[0, 0, .... 128 개]

W = np.random.random((output_features, input_features))     # T 시점 새로운 input 과 내적할 상수 (128 * 64)
U = np.random.random((output_features, output_features))    # T 시점 t_1 state 와 내적할 상수 (128 * 128)
b = np.random.random((output_features, ))       # T 시점에 더해줄 상수 (128,)
print('b : ', b)

##################### 여기까지 데이터 준비 및 상수 준비 + 차원수 세팅

succ_output = []
for input_t in inputs:      #inputs(1000x64), 2D시에 행단위로 for문 돔, input_t는 크기가(64,)인 벡터
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)      # W와 input_t 의 내적값 / new 입력과 이전출력 상태(t_1 state)로 현재출력 얻음
    succ_output.append(output_t)        # time step 별 저장
    state_t = output_t      # state_t 가 다시 다음 for 에서 사용됨

output_t
print(output_t)
print(type(output_t))

"""
what is time step in RNN?
    In a recurrent neural network you have multiple repetitions of the same cell.
    A time step is a single occurrence of the cell -e.g. on the first time step you produce 
    output1, h0, on the second time step you produce output2 and so on.
time step (또는 sequence size)는 RNN모델을 unfold할때 몇 단계로 할 것인지를 의미한다.  
각 weight는 input/output size에 따라서 size가 설정되어야 한다.     
"""

