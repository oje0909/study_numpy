# RNN은 시퀀스(sequence) 모델이다.
# 입력과 출력을 시퀀스 단위로 처리하는 모델이다.
# 의사 코드(pseudocode)

hidden_state_t = 0      # 초기 은닉 상태를 0(벡터)로 초기화
for input_t in input_length:    # 각 시점마다 입력을 받는다.
    output_t = tanh(input_t, hidden_state_t)    # 각 시점에 대해서 입력과 은닉 상태를 가지고 연산
    hidden_state_t = output_t       # 계산 결과는 현재 시점의 은닉 상태가 된다.

# 우선 t 시점의 은닉 상태를 hidden_state_t라는 변수로 선언하였고, 입력 데이터의 길이를 input_length 로 선언
# 이 경우, 입력 데이터의 길이는 곧 총 시점의 수(timesteps)가 됩니다. 그리고 t 시점

