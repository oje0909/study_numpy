# https://wikidocs.net/22886
# hidden layer 를 갖고 있으면 다 RNN 구조 인건가?

# 1. 데이터 준비

# 2. 네트워크 생성

# 3. 학습
# (backward) 역전파 하는 과정을 학습이라고 부르는건가?

# 4. 손실을 도식화하여 보여줌


import numpy as np

# 시점, 입력 차원, 은닉 상태의 크기, 초기 은닉 상태 0 벡터 초기화
timesteps = 10  # 시점의 수, NLP 에서는 보통 문장의 길이
input_dim = 4   # 입력의 차원, NLP에서는 보통 단어 벡터의 차원 음절?
hidden_size = 8     # 은닉 상태의 크기. 메모리 셀의 용량

inputs = np.random.random((timesteps, input_dim))   # 2DTensor
hidden_state_t = np.zeros((hidden_size, ))  # 초기 은닉 상태는 0으로 초기화, 은닉 상태의 크기 hidden_size로 은닉 상태를 만든다.
print(hidden_state_t)       # [0. 0. 0. 0. 0. 0. 0. 0.]

# 가중치와 편향 정의 - 처음에는 랜덤 2DTensor 생성
Wx = np.random.random((hidden_size, input_dim))     # (8, 4)크기의 2D Tensor. 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size))   # (8, 8)크기의 2D Tensor. 은닉 상태에 대한 가중치
b = np.random.random((hidden_size, ))           # (8, ) 크기의 1D Tensor. 편향(bias) -> 이거 왜 있는거임?
print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))

# 이제 모든 시점의 은닉 상태를 출력한다고 가정하고 RNN층 동작시키기
total_hidden_states = []

# 메모리 셀 동작 (hidden layer 동작)
for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)    # Wx * Xt + Wh * Ht-1 + b(bias)
    total_hidden_states.append(output_t)        # 각 시점의 은닉 상태의 값을 계속해서 축적...왜? 그냥 보려고?
    print(np.shape(total_hidden_states))        # 각 시점 t별 메모리 셀(hidden layer)의 출력의 크기는 (timestep. output_dim)
    hidden_state_t = output_t       # 가중치 축적

# 출력 시, 값을 깔끔하게 해준다... 그래서 뭐하는 함수인데 np.stack
total_hidden_states = np.stack(total_hidden_states, axis=0)
    
print(total_hidden_states)      # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.



