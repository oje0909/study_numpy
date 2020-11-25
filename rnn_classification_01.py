# 문자 단위 RNN으로 이름 분류하기
"""
단어를 분류하기 위해 기본적인 문자 단위 RNN을 만들고 훈련할 것입니다.
문자 단위 RNN은 문자의 연속을 읽어 들여서 각 단계의 예측과 "hidden state" 를 추력하고
다음 단계에 이전 hidden state 를 전달합니다,
즉, 단어가 속한 클래스 같은 출력을 최종 예측으로 선택합니다.

추천 자료:

최소한 Pytorch를 설치했고, Python을 알고, Tensor를 이해한다고 가정합니다.:

http://pytorch.org/ 설치 안내를 위한 자료
PyTorch로 딥러닝하기: 60분만에 끝장내기 일반적인 PyTorch 시작을 위한 자료
예제로 배우는 PyTorch 넓고 깊은 통찰을 위한 자료
Torch 사용자를 위한 PyTorch 이전 Lua Torch 사용자를 위한 자료
RNN과 그 작동 방식을 아는 것 또한 유용합니다.:

The Unreasonable Effectiveness of Recurrent Neural Networks 실생활 예들을 보여 줍니다
Understanding LSTM Networks 특히 LSTM에 관한 것이지만 일반적인 RNN에 대한 정보입니다.

"""
# 파이토치를 사용하면, 자동미분을 통해 (autograd), backward() 가 자동 실행/추적하여
# 변화도를 누적/변경할 수 있다. 변화도 == 가중치?
# 편향? 바이어스는 언제 쓰는 거?

from __future__ import unicode_literals, print_function, division       # 파이썬2일 경우, 3과의 호환?을 위해
from io import open     # 파일 입출력
import glob     # 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 바꿀 수 있음.


# 1. 데이터셋 준비
def findFiles(path): return glob.glob(path)
print(findFiles('data/names/*.txt'))

import unicodedata      # 모든 유니코드 문자에 대한 문자 속성을 정의하는 유니코드 문자 데이터베이스에 대한 엑세스를 제공합니다
import string

all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)        # 55


# 유니 코드 문자열을 일반 ASCII로 변환하십시오.
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)      # .normalize() 유니코드 문자열 unistr에 대한 정규화 형식 form을 반환, form의 유효한 값을 <NFC>, <NFKC>, <NFD> 및 <NFKD>
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
print(unicodeToAscii('Ślusàrski'))      # 테스트

# 언어별 이름 목록인 category_lines 사전을 만드십시오.
category_lines = {}
all_categories = []


# 파일을 읽고 라인으로 분리하십시오.
def readlines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readlines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# 이제 각 카데로기(언어)를 줄(이름) 목록에 매칭하는 사전인 category_lines가 있다.
# 또한 나중에 참조 가능한 all_categories(언어 목록)와 n_categories를 추적

print(category_lines['Italian'][:5])
print('category_lines : ', category_lines)
print(all_letters)      # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;
print(n_categories)     # 18  언어 카테고리 수

# 2. 이름을 Tensor 로 변경
# 하나의 문자를 표현하기 위해, 크기가 <1 * n_letters> 인 one-hot vector 사용
# one-hot 벡터는 현재 문자의 주소에만 1을 값을 가ㅣ고 그 외 나머지는 0으로 채워진다.
# 단어를 만들기 위해 그 묶음을 2차원 행렬 <line_length * 1 * n_letters>에 결합시킨다.
# 1차원을 추가하는 것은 PyTorch 가 모든 것이 batch에 있다고 가정하기 때문에 발생합니다.
# 여기서는 배치 크기 1을 사용하고 있습니다.
import torch


# all_letters 로 문자의 주소 찾기, 예시 'a' = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# 검증을 위해서 한 문자를 one-hot vector 로 변환하기
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)      # n_letters = len(all_letters) 55
    print('tensor :', tensor)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 왜 텐서가 이렇게 크게 나오는지 잘 모르겠어,,, 머리가 안 돌아가,,

# 한 줄(이름)을 <line_length * 1 * n_letters>,
# 또는 문자 벡터의 어레이로 변경하기
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToIndex('b'))
print(letterToIndex('ㅎ'))       # -1 존재하지 않음
print(letterToTensor('J'))
print(lineToTensor('Jieun').size())
print(lineToTensor('Jones').size())

# 3. 네트워크 생성
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)     # 283, 128
        self.i2o = nn.Linear(input_size + hidden_size, output_size)     # 283, 18
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)          # torch.cat(tensor, dim=0), https://pytorch.org/docs/stable/generated/torch.cat.html 참고바람
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
    """
    0을 값으로 갖는 텐서(1, 은닉층수)를 생성하고,
    Variable로 감쌉니다. 이를 통해 변화도를 자동미분할 수 있습니다.
    Variable : 이전까지 우리는 신경망의 순전파 단계와 역전파 단계를 수동으로 구현하였습니다.
    작은 2-계층 신경망에서 역전파 단계를 직접 구현하는 것은 큰 일이 아니지만, 대규모의 복잡한 신경망에서는 매우 아슬아슬한 일일 것입니다.
    다행히도 자동 미분(Autograd) 를 사용하여 신경망에서 역전파 단계의 연산을 자동화할 수 있습니다. -> PyTorch의 autogard 
    Autograd 를 사용할 때, 신경망의 순전파 단계는 연산 그래프를 정의합니다.   -> 연산 그래프?
    그래프의 노드(Node, 연결망의 교점?, A node is a point, especially in the form of lump or swelling, where one thing joins another.) 
    는 Tensor 이며, 엣지(Edge) 는 입력 Tensor 로부터 출력 Tensor를 만들어내는 함수이다.
    이 그래프를 통해 역전파를 하게 되면 변화도를 쉽게 계산할 수 있습니다.
    이는 복잡해보이지만 실제로 사용하는 것은 간단합니다.
    PyTorch Tensor를 Variable 객체로 감싸게 되면, 이 Variable 이 연산 그래프에서 노드로 표현됩니다.
    x 가 Variable 일 때, x.data 는 그 값을 갖는 Tensor 이며,
    x.grad 는 어떤 스칼라 값에 대해 x에 대한 변화도를 갖는 또 다른 Variable 입니다.
    
    PyTorch Variable은 PyTorch Tensor와 동일한 API를 제공합니다.
    Tensor에서 할 수 있는 (거의) 모든 연산은 Variable에서도 할 수 있습니다.
    차이점은 연산 그래프를 정의할 때 Variable을 사용하면, 자동으로 변화도를 계산할 수 있다는 것입니다.
    """

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)        # 55, 128, 18
"""
이 네트워크의 한 단계를 실행하려면 입력(현재 문자의 Tensor) 과 이전의 hidden stae(처음에는 0으로 초기화)를 전달해야 합니다.
출력(각 언어의 확률) 과 다음 hidden state(다음 단계를 위한 유지)를 돌려 받습니다.
PyTorch 모듈은 Tensors 에서 바로 작동하는 대신에 Variables 에서 작동한다는 것을 기억하십시오.
"""

input = Variable(letterToTensor('A'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input, hidden)
# 효율성을 위해서 매 단계마다 새로운 Tensor를 만들고 싶지 않기 때문에 letterToTensor 대신 lineToTensor 를 잘라서 사용할 것입니다.
# 이것은 Tensor의 사전 연산(pre-computing) 배치에 의해 더욱 최적화 될 수 있습니다.
input = Variable(lineToTensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input[0], hidden)
print(output)
# 보다시피 출력은 <1 * n_categories> Tensor 이고, 모든 항목은 해당 카테고리의 우도????????(likelihood)입니다. / 공산/가능성
# 더 높은 것이 더 가능성 높음


# 4. 학습
# 4_1. 학습 준비
# 학습에 들어가기 전에 몇몇 도움되는 함수를 만들어야 합니다.
# 첫째는 네트워크의 알고 있는 각 카테고리의 우도로 출력을 해석하기 입니다.
# 가장 큰 값의 주소를 알기 위해서 Tensor.topk를 쓸 수 있습니다.

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)      # Tensor out of Variable with .data
    category_i = top_i[0][0]        # 아니 이게 왜 [0][0] 인지도 모르겠어,, -> 그냥 테스트용인가?
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

# +) 학습 예시(이름과 언어)를 얻는 빠른 방법을 원할 것입니다.
import random


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category = ', category, '/ line = ', line)

# 4_2 네트워크 학습
# 이제 이 네트워크를 훈련하는데 필요한 모든 예시를 보여주고, 추측을 하고, 만일 잘못되었다면 말해줍니다.
# RNN의 마지막 레이어가 LogSoftmax 니까 손실함수로 NLLLoss가 적합
# 학습의 각 루프는 다음을 실행합니다.
# 입력과 목표 tensor 생성
# 0으로 초기화된 hidden state 생성
# 각 문자를 읽기

criterion = nn.NLLLoss()
learning_rate = 0.005       # 너무 높게 설정하면 폭발할 수 있고(? 너무 낮으면 학습이 되지 않을 수 있습니다.
learning_rate = 0.005       # 너무 높게 설정하면 폭발할 수 있고(? 너무 낮으면 학습이 되지 않을 수 있습니다.


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # learning rate를 곱한 파라미터의 경사도를 파라미터 값에 더합니다.
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data
    """
    That's because in PyTorch>=0.5, the index of 0-dim tensor is invalid.
    The master branch is designed for PyTorch 0.4.1, loss_val.data[0] works well.
    https://github.com/NVIDIA/flownet2-pytorch/issues/113
    """

# 5. 예저를 사용하여 실행
# train 함수가 출력과 손실을 반환하기 때문에 추측을 출력하고 도식화를 위한 손실을 추적할 수 있습니다.
# 1000개의 예제가 있기 때문에 모든 print_every 예제만 출력하고 손실의 평균을 얻습니다.

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# 도식화를 위한 소실 추적
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)      # 버림
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # iter 숫자, 손실, 이름, 추측 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        
    # 현재 평균 손실을 손실 리스트에 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# 6. 결과 도식화 : all_losses 를 이용한 역사적인 손실 도식화는 네트워크의 학습을 보여준다.

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()

# 7. 결과 평가: 네트워크가 다른 카테고리에서 얼마나 잘 작동하는지 보려면 네트워크에서 추측한 언어(행)과 실제 언어(행)를 나타내는
# 혼란 행렬(confusion matrix) 를 만든다.
# 혼란 행렬을 계산하기 위해 evaluate() 로 많은 수의 샘플을 네트워크에 실행합니다.
# evaluate() 는 train() 과 역전파를 빼면 동일합니다.

# 혼란 행랼에서 정확한 추측을 추적
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# 올바르게 추측된 예시와 기록을 살펴보십시오.
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 모든 행을 합계로 나눔으로써 정규화하십시오.
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()
    
# 도식 설정
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 모든 tick에서 강제로 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

# +) 사용자 입력으로 실행
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(Variable(lineToTensor(input_line)))

    # 최고 N 카테고리 얻기
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
predict('Oh')

"""
연습
“line -> category” 의 다른 데이터 집합으로 시도해보십시오, 예를 들어:
단어 -> 언어
이름 -> 성별
캐릭터 이름 -> 작가
페이지 제목 -> 블로그 또는 서브레딧
더 크고 더 나은 모양의 네트워크로 더 나은 결과를 얻으십시오.
더많은 선형 layer 추가해 보십시오
nn.LSTM 과 nn.GRU layer 추가해 보십시오
여러 개의 이런 RNN을 상위 수준 네트워크로 결합해 보십시오.
"""

