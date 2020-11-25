# 문자 단위 RNN으로 이름 생성하기

# 1. for datasets
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string

# make network
import torch
import torch.nn as nn
from torch.autograd import Variable

# train
import random

# 학습에 걸리는 시간 측정
import time
import math

# 손실 도식화
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# 1. 데이터 준비
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1    # Plus EOS marker


def findFiles(path): return glob.glob(path)


# 유니 코드 문자열을 일반 ASCII로 변환하십시오. http://stackoverflow.com/a/518232/2809427 에 감사드립니다.
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 파일을 읽고 라인으로 분리하십시오
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 카테고리별 줄 목록인 category_lines 사전을 만드십시오.
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]        # ex. English
    all_categories.append(category)
    lines = readLines(filename)      # ["1 번째 줄입니다.", "2 번째 줄입니다.", ..., "10 번째 줄입니다."]
    category_lines[category] = lines        # English : [Albert, Ben, Jean...]

n_categories = len(all_categories)      # 55

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

# 2. 네트워크 생성
# 이 네트워크는 지난 튜토리어르이 RNN이 다른 것들과 연결되는 category tensor 를 추가 인자로 가지게 확장합니다.
# category tensor는 문자 입력과 마찬가지로 one-hot 벡터입니다.

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)      
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)        # 아니 인자 왜 이렇게 넣냐고
        self.dropout = nn.Dropout(0.1)      # 내가 알기로 dropout은 신경망 몇 개의 추론 or 학습을 랜덤으로 막는 것
        self.softmax = nn.LogSoftmax(dim=1)     # dim 은 또 뭔데
        # what is difference between nn.Softmax and nn.LogSoftmax?

    def forward(self, category, input, hidden):     # 어차피 역전파할거니까 output 은 필요없는건가?
        input_combined = torch.cat((category, input, hidden), 1)          # 왜 input_combined 이라고 불러?
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))       # 이게 뭐였더라 인자 1은 뭐야?

#################################################################################### - 여기까지 네트워크 생성

# +) 무작위 아이템 및 임의의 이름 얻기
# 리스트에서 무작위 아이템 반환
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# 임의의 category 및 그 category 에서 임의의 줄(이름) 얻기
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line
"""
각 스텝마다 (즉, 학습 단어의 각 문자마다) 네트워크의 입력은 (category, 현재 문자, hidden state) 이 되고,
출력은 (다음 문자, 다음 hidden state) 가 된다.
따라서 각 학습 세트마다 category, 입력 문자의 세트, 출력/목표 문자의 세트가 필요하다.
매 타임 스텝마다 현재 문자에서 다음 문자를 예측하기 때문에, 문자 쌍은 줄에서 연속된 문자 그룹입니다.
예를 들어 "ABCD<EOS>" 는 (“A”, “B”), (“B”, “C”), (“C”, “D”), (“D”, “EOS”) 로 생성합니다. -> 먼 소리야
# EOS 토큰 : 문자가 끝났음을 알 수 있는 " .,;'-"

category tensor 는 <1 * n_categories> 크기의 one-hot tensor 입니다.
학습 시에 모든 타임 스텝에서 네트워크에 그것을 전달합니다.
이것은 설계 시, 선택사항으로 초기 hidden state 부분 또는 또다른 전략이 포함될 수 있습니다.
"""

# category를 위한 One-hot 벡터
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# 입력을 위한 처음부터 마지막 문자(EOS 제외)까지의 One-hot 행렬
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# 목표를 위한 두번째 문자부터 마지막(EOS) 까지의 LongTensor
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]     # ...뭐라는거지
    letter_indexes.append(n_letters - 1)    # EOS   ...?
    return torch.LongTensor(letter_indexes)

# 학습 동안 편의를 위해 임의의 (category, line) 을 가져오고 그것을 필요한 형태(category, input, target)
# tensor로 바꾸는 randomTrainingExample 함수를 만들 예정


# 임의의 category에서 category, input, target Tensor 를 만듭니다.
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return category_tensor, input_line_tensor, target_line_tensor


# hidden layer 가 있는게 RNN 구조
# 네트워크 학습
# 앞서서 해보았던 분류의 경우에는 마지막 출력만 사용하지만, 생성에서는 모든 단계에서 예측을 수행하므로 모든 단계에서 손실을 계산해야 합니다.
criterion = nn.NLLLoss()        #  MSE를 쓸지, Crossentropy 를 쓸지 log loss 를 쓸지 어케 정해?
learning_rate = 0.0005
def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()       # 밑에 적고나면 사라지나봄 어케 그게 가능한지는 모르겠지만
    rnn.zero_grad()

    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()     # 역전파 왜 하는거죠? training 시키려고?

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.data[0] / input_line_tensor.size()[0]
    # return output, loss.data[0] / input_line_tensor.size()[0]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0      # 매 plot_every 마다 초기화

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# 손실 도식화
plt.figure()
plt.plot(all_losses)

max_length = 20


# 카테고리와 시작 문자에서 샘플 선택
def sample(category, start_letter='A'):
    category_tensor = Variable(categoryTensor(category))
    input = Variable(inputTensor(start_letter))
    hidden = rnn.initHidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(category_tensor, input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
        input = Variable(inputTensor(letter))

    return output_name


# 하나의 카테고리와 여러 시작 문자들에서 여러 샘플 얻기
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('Russian', 'RUS')
samples('German', 'GER')
samples('Spanish', 'SPA')
samples('Chinese', 'CHI')
