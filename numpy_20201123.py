from __future__ import print_function       # 파이썬 3에서 쓰던 문법을 파이썬2에서 쓸 수 있게 해주는 문법
import torch

x = torch.Tensor(5, 3)
print(x)        # 초기화되지 않은 5*3 행렬

x = torch.rand(5, 3)
print(x)        # 무작위로 초기화된 행렬 생성

print(x.size())         # torch.Size는 튜플과 같으며, 모든 튜플 연산에 사용할 수 있다.

y = torch.rand(5, 3)
print(x + y)        # or torch.add(x, y) or y.add_(x)
# in_place로, tensor 의 값을 변경하는 연산은 _를 접미사로 갖는다.
# shape란?
# Numpy의 인덱싱 표기 방법을 사용할 수 있다.
print(x[:, 1])

# tensor의 크기나 모양을 변경하고 싶을 때, torch.view 를 사용

x = torch.randn(4, 4)       # 초기화된 무작위 행렬
y = x.view(16)
z = x.view(-1, 8)           # -1인 경우 다른 차원들을 사용하여 유추한다. -?
print(x.size(), y.size(), z.size())

# 전치, 인덱싱, 슬라이싱, 수학 계산, 선형 대수, 난수 ...

a = torch.ones(5)       # 텐서
print(a)
print(type(a))         # torch.FloatTensor

b = a.numpy()
print(b)
print(type(b))          # <class 'numpy.ndarray'>

import numpy as np

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a, 1, out=a)         # out=a ?
print(a)
print(b)

c = torch.randn(5, 3)
d = torch.randn(5, 3)


# .cuda 메소드를 사용해서 Tensor 를 GPU 상으로 옮길 수 있다.
if torch.cuda.is_available():
    c = c.cuda()
    d = d.cuda()
    # print(x + y)


"""
Tensor 는 Numpy 의 ndarrays 와 유사한 것으로 계산 속도를 빠르게 하기 위해 GPU 에서 사용할
수 있는 것이라고 보면 된다.
"""


import torch

print(torch.zeros(5, 3))


