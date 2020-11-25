import torch
from torch.autograd import Variable
from torch.autograd import Function

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10       # 순서대로 배치 크기, 입력의 차원, 은닉 계층의 차원, 출력 차원

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor 생성,
# 역전파 중에 이 Variable 들에 대한 변화도를 계산할 필요가 없음을 False 로 나타낸다.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)     # 64, 1000
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)    # 64, 10

# 가중치(w1->순전파용?, w2->역전파용?)를 저장하기 위해 무작위 값을 갖는 Tensor 생성(초기화됨)하고,
# requires_grad=True 로 설정하여 역전파 중에 이 Variable들에 대한 변화도를 계산할 필요가 있음을 나타냄.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)     # 1000으로 받고, 100의 은닉층으로 빠짐
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)    # 10으로 받고, 100의 은닉층으로 빠짐 -> 역전파할 거라서 반대로 가나?
# 자료형의 타입을 지정해주면서 GPU를 사용하도록..?

learning_rate = 1e-6        # 1e-6 is equal to 0.000001 in decimal
"""
The learning rate hyperparameter controls the rate or speed at which the model learns. 
Specifically, it controls the amount of apportioned error that the weights of the model are updated
with each time they are updated, such as at the end of each batch of training examples.
"""
for t in range(500):
    # 순전파 단계 Variable 연산을 사용하여 y값을 예측합니다.
    # 이는 Tensor를 사용한 순전파 단계와 완전히 동일하지만,
    # 역전파 단계를 별도로 구현하기 않기 위해 중간 값들(Intermediate Value)에 대한 참조(Reference)를 갖고 있을 필요가 없습니다.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Variable 연산을 사용하여 손실을 계산하고 출력합니다.
    # loss는 (1,) 모양을 갖는 Variable 이며, loss.data는 (1, ) 모양의 Tensor 입니다. -> 자료형 출력해보기
    # loss.data[0]은 손실(loss)의 스칼라 값입니다.
    loss = (y_pred - y).pow(2).sum()
    # print(t, loss.data[0])
    print(t, loss.data)
    print(type(loss))   # 근데 Variable 도 Tensor 안에 감싸져 있는거라고 했었던 것 같음
    print(type(loss.data))

    # autograd를 사용하여 역전파 단계를 계산합니다.
    # 이는 requires_grad=True를 갖는 모든 Variable에 대한 손실의 변화도를 계산합니다.
    # 이후 w1.grad 와 w2.grad는 w1과 w2 각각에 대한 손실의 변화도를 갖는 Variable 이 됩니다.
    loss.backward()

    # 경사하강법 (Gradient Descent)
    # 경사하강법을 사용하여 가중치를 갱신합니다.
    # w1.data와 w2.data는 Tensor이며, w1.grad와 w2.grad는 Variable이고, w1.grad.data와 w2.grad.data는 Tensor입니다.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
    w1.grad.data.zero_()
    w2.grad.data.zero_()



