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
from __future__ import unicode_literals, print_function, division       # 파이썬2일 경우, 3과의 호환?을 위해
from io import open     # 파일 입출력
import glob     # 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 바꿀 수 있음.

# 1. 데이터셋 준비
def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata      # 모든 유니코드 문자에 대한 문자 속성을 정의하는 유니코드 문자 데이터베이스에 대한 엑세스를 제공합니다
import string

all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)


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
    tensor = torch.zeros(1, n_letters)
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

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)          # torch.cat(tensor, dim=0), https://pytorch.org/docs/stable/generated/torch.cat.html 참고바람
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
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
criterion = nn.NLLLoss()

# 학습의 각 루프는 다음을 실행합니다.
# 입력과 목표 tensor 생성
# 0으로 초기화된 hidden state 생성
# 각 문자를 읽기

learning_rate = 0.005

