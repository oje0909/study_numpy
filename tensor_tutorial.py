"""
what is difference between torch and pytorch?
Here a short comparison on pytorch and torch.

Torch:
A Tensor library like Numpy, unlike Numpy it has strong GPU support.
Lua is a wrapper for Torch (Yes! you need to have a good understanding of Lua),
and for that you will need LuaRocks package manager.

PyTorch:
No need of the LuaRocks package manager, no need to write code in Lua.
And because we are using Python, we can develop Deep Learning models with utmost flexibility.
We can also exploit major Python packages likes scipy, numpy, matplotlib and Cython with pytorch's own autograd.

There is a detailed discussion on this on pytorch forum.
Adding to that both PyTorch and Torch use THNN.
Torch provides lua wrappers to the THNN library while Pytorch provides Python wrappers for the same.
PyTorch's recurrent nets, weight sharing and memory usage with the flexibility of interfacing with C, and the current speed of Torch.
"""


# Tensor

import torch

a = torch.FloatTensor(5, 7)     # 초기화 x
a = torch.randn(5, 7)       # 평균 0, 분산 1의 정규분포를 따르는 초기화된 무작위 숫자 tensor
print(a)
print(a.size())

# in-place / out-of-place
# 첫 번째 차이점은 tensor의 모든 in-place 연산은 _ 접미사를 갖는다는 것이다.
# 예를 들어, add는 연산 결과를 돌려주는 out-of-place 연산을 하고 add_ 는 in-place 연산을 합니다.

a.fill_(3.5)
# a가 3.5 라는 값으로 채워짐
b = a.add(4.0)
# a는 여전히 3.5이고, 3.5 + 4.0 = 7.5 의 값이 반환되어 새로운 tensor b가 됩니다.
print(a, b)

# 그러나, narrow 와 같은 일부 연산들은 In-place 형태를 갖지 않기 때문에 .narrow_ 는 존재하지 않습니다.
# 또한, fill_ 은 Out-of-place 형태를 갖지 않기 떄문에 역시 .fill 도 존재하지 않습니다.

# 0-인덱스
# 또 다른 차이점은 Tensor 의 인덷스는 0부터 시작(0-인덱스)의 점입니다.
#Lua 에서 tensor는 1-인덱스를 갖습니다.

# 카멜표기법 없음

# 예제로 배우는 PyTorch
# https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/beginner/pytorch_with_examples.html

# PyTorch 주요 특징 2가지
# NumPy와 유사하지만, GPU 상에서 실행 가능한 N차원 Tensor
# 신경망을 구성하고 학습하는 과정에서의 자동미분 (Autograd)

# 이 예제에서 완전히 연결된 ReLU 신경망을 예제로 사용할 것입니다.
# 이 신경망은 하나의 은닉 계층(Hidden Layer)을 갖고 있으며,
# 신경망의 출력과 정답 사이의 유클리드 거리 (Euclidean Distance)를 최소화하는 식으로 경사하강법(Gradient Descent)을 사용하여
# 무작위의 데이터를 맞추도록 학습할 것입니다.

# 1) NumPy
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위 입력 출력 데이터 생성
x = np.random.randn(N, D_in)        # 왜? 인자 2개?
y = np.random.randn(N, D_out)

# 무작위로 가중치를 초기화
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
# for t in range(500):

# Data Loading and Processing Tutorial
# Deep Learning for NLP with Pytorch
# 일단 뛰어넘고 감.

