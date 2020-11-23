"""
신경망(Neural Networks)
torch.nn 패캐지를 사용하여 생성할 수 있다.
nn은 모델을 정의하고 미분하는데 autograd 를 사용한다.
nn.Module은 계층(layer)와 output 을 반환하는 forward(input) 메서드를 포함하고 있다.

- 신경망의 전형적인 학습 과정
1) 학습 가능한 매개변수 또는 가중치(weight)를 갖는 신경망을 정의
2) 데이터셋(dataset) 입력을 반복합니다.
3) 입력을 신경망에서 처리한다.
4) 손실(loss) 출력이 정답으로부터 얼마나 떨어져있는지 계산한다
5) 변화도(gradient) 를 신경망의 매개변수들에 역으로 전파한다. (역전파)
6) 신경망의 가중치를 갱신한다.
가중치 = 가중치 - 학습율 * 변화도
weight = weight = learning rate * gradient

"""

# 신경망 정의하기
# 입력을 처리하고 backward 를 호출하는 것
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)     # 입력 채널, 출력 채널, 커널크기 5*5
        self.conv2 = nn.Conv2d(6, 16, 5)    # 입력 채널, 출력 채널, 커널 크기
        # an affine operation: u = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):        # 피드포워드신경망?        # 이거 정의하면, 따로 backward 함수는 정의할 필요 없다.
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
# forward 함수만 정의하게 되면, 변화도를 계산하는 backward 함수는 autograd 를 사용하여 자동으로 정의된다.
# 모델의 학습 가능한 매개변수들은 net.parameters() 에 의해 반환된다.
params = list(net.parameters())
print(len(params))      # 10
print(params[0].size())     # torch.Size([6, 1, 5, 5])

"""
MNIST: 손글씨 숫자 이미지들을 모아놓은 데이터
Batch size : sample 데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다. batch 는 mini batch 라고도 불린다.
epoch 와는 다른 개념
예를 들어, 1000개의 데이터를 batch_size = 10개로 넘겨준다고 하자.
그러면 총 10개씩 batch로서 그룹을 이뤄서 들어가게 되며, 총 100개의 step을 통해 1epoch 를 도는 것이다.
즉, 1 epoch(학습 1번) = 10(batch_size) * 100(step or iteration)

torch.Tensor - 다차원 배열.
autograd.Variable - Tensor를 감싸고 모든 연산을 기록 합니다. Tensor 와 동일한 API를 갖고 있으며, backward() 와 같이 추가된 것들도 있습니다.
또한, tensor에 대한 변화도를 갖고 있습니다.
nn.Module - 신경망 모듈. 매개변수를 캡슐화(Encapsulation)하는 간편한 방법으로,
GPU로 이동, 내보내기(exporting), 불러오기(loading) 등의 작업을 위한 헬퍼(helper)를 제공합니다.
nn.Parameter - 변수의 한 종류로, Module 에 속성으로 할당될 때 자동으로 매개변수로 등록 됩니다.
autograd.Function - autograd 연산의 전방향과 역방향 정의 를 구현합니다. 모든 Variable 연산은 하나 이상의 Function 노드를 생성하며,
각 노드는 Variable 을 생성하고 이력(History)을 부호화 하는 함수들과 연결하고 있습니다.

Variable
Function
"""

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

########################################################################
# 모든 매개변수의 변화도 버퍼(gradient buffer)를 0으로 설정하고, 무작위 값으로 역전파를 합니다:
net.zero_grad()
out.backward(torch.randn(1, 10))

# 손실 함수 (Loss Function) : (output, target)을 한 쌍(pair)의 입력으로 받아, 출력(output)이 정답(target)으로부터 얼마나 떨어져 있는지를 추정하는 값을 계산한다.
# nn.MSEloss

output = net(input)
target = Variable(torch.arange(1, 11))      # a dummy target, for example   # tensor 형태로 변경
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# .grad_fn 속성을 사용하여 loss 를 역방향에서 따라가다보면, 이러한 모습의 연산 그래프를 볼 수 있다.
"""
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
"""

# 우리가 loss.backward() 를 실행할 때, 전체 그래프는 손실에 대해 미분되면,
# 그래프 내의 모든 변수는 변화도가 누적된 .grad 함수를 갖게 됩니다.
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# 역전파 (Backprop)
# 오차(error) 를 역전파하기 위해 할 일은 loss.backward() 가 전부입니다.
# 기존 변화도를 지우는 작업이 필요한데, 이를 하지 않으면 변화도가 기존의 것에 누적되기 때문입니다.

net.zero_grad()     # zeros the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 가중치 갱신
# 실제로 많이 사용되는 가장 단순한 갱신 규칙은 확률적 경사하강법(SGD : stochastic gradient descent)
# 가중치(wiehgt) = 가중치(weight) - 학습율(learning rate) * 변화도(gradient)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 그러나, 신경망을 구성할 때, SGD, Nesterov_SGD, RMSProp 등과 같은 다양한 갱신 규칙을 사용하고 싶을 수 있습니다.
# 이를 위해서 torch.optim 라는 작은 패키지에 이러한 방법들을 모두 구현해두었습니다.

import torch.optim as optim

# Optimizer 생성
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop) 에서는 다음과 같습니다.
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)     # input(정답)을 넣어서 output 데이터를 만들고,
loss = criterion(output, target)    # 둘 사이의 손실된 값을 구해서,
loss.backward()     # 이 값으로 역전파하여
optimizer.step()        # Does the update   누적시킨다.

# optimizer.zero_grad() 를 사용하여 수동으로 변화도 버퍼를 0으로 설정하는 것에 유의하세요.
# 이는 역전파(Backprop) 섹션에서 설명한 것처럼 변화도가 누적되기 때문입니다.





