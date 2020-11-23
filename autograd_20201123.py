# Autograd : 자동 미분
# pytorch 의 모든 신경망의 중심에는 autograd 패키지가 있다.
# autograd 패키지는 Tensor 의 모든 연산에 대해 자동 미분을 제공한다.
# 실행 - 기반 - 정의 프레임워크로, (define-by_run)
# 코드를 어떻게 작성하여 실행하느냐에 따라 역전파가 정의된다는 의미
# 역전파는 학습 과정의 매 단계마다 달라진다.

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)      # torch.ones() -> 1로만 이루어진 텐서 생성
# requires_grad 속성을 True로 설정하면, 그 tensor에서 이뤄진 모든 연산들을 추적(track) 하기 시작한다.
# 계산이 완료된 후, .backward() 를 호출하여 모든 변화도(gradient)를 자동으로 계산할 수 있다.
# Tensor 가 기록을 추적하는 것을 중단하려면, .detach()를 호출하여, 연산 기록으로부터 분리(detach)하여 이후 연산들이 추적되는 것을
# 방지할 수 있다.
# 기록을 추적하는 것과 메모리를 사용하는 것을 방지하기 위해, 코드 블럭을 with torch.mo_grad(): 감쌀 수 있다.
# 특히 변화도는 필요 없지만, requires_grad=True 가 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용

print(x)

y = x + 2
print(y)
# y는 연산의 결과로 생성된 것이므로, grad_fn을 갖는다.
print(y.grad_fn)        # True 라서 추적가능
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
# <AddBackward0 object at 0x7f18a49e17c0>
# 도함수 : 함수 ƒ(x)를 미분하여 얻은 함수. y＇, ƒ＇(x), dy/dx 따위로 나타낸다.

# requires_grad = True 로 설정하면 연산을 기록할 수 있다.
z = y * y * 3
out = z.mean()

print(z, out)

# 변화도 : Gradient
# out.backward() 는 out.backward(torch.Tensor([1, 0])를 하는 것과 똑같다.

out.backward()
print(out)
print(x.grad)   # 변화도 출력 (d(out)/dx)
# 4.5로 이루어진 행렬

x = torch.randn(3)      # tensor([ 0.1255, -0.3454,  1.8875])
print(x)
x = Variable(x, requires_grad=True)     # 추척 가능

y = x * 2
while y.data.norm() < 1000:         # Norm은 벡터의 길이 혹은 크기를 측정하는 방법(함수)...?
    y = y * 2

print(y)
