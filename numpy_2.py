import numpy as np

# 3. 자료형
# Numpy 배열은 동일한 자료형을 가지는 값들이 격자판 형태로 있는 것이다.
# Numpy 에선 배열을 구성하는데 사용할 수 있는 다양한 숫자 자료형을 제공합니다.
# Numpy 는 배열이 생성될 때 자료형을 스스로 추측합니다.
# 그러나 배열을 생성할 때 명시적으로 특정 자료형을 지정할 수도 있습니다.

x = np.array([1, 2])    # numpy 가 자료형을 추측해서 선택
print(x.dtype)      # int64 

x = np.array([1.0, 2.0])
print(x.dtype)      # float64

x = np.array([1, 2], dtype=np.int64)    # 특정 자료형을 명시적으로 지정
print(x.dtype)

# 배열 연산
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# 요소 별 합 : 둘 다, 다음의 배열을 만듭니다.
print(x + y)
print(np.add(x, y))
# [[ 6.  8.]
#  [10. 12.]]

# 요소 별 차 : 둘 다 다음의 배열을 만든다.
print(x - y)
print(np.subtract(x, y))

# 요소 별 곱 : 둘 다 다음의 배열을 만든다.
print(x * y)
print(np.multiply(x, y))

# 요소 별 나눗셈 : 둘 다 다음의 배열을 만든다.
print(x/y)
print(np.divide(x, y))

# 요소 별 제곱근 : 다음의 배열을 만든다.
# 제곱근 : 제곱해서 해당 요소가 되는 것
print(np.sqrt(x))

# MATLAB 과 달리 ''은 행렬 곱이 아니라 요소별 곱입니다.
# Numpy에선 벡터의 내적, 벡터와 행렬의 곱, 행렬곱을 위해 ' 대신 'dot' 함수를 사용합니다.
# 'dot'은 Numpy 모듈 함수로서도 배열 객체의 인스턴스 메소드로서도 이용 가능한 함수입니다.

# 벡터의 내적(inner/dot product) :
# product 는 곱셈이라는 뜻이 있는데 뜻을 대입해보면 내적은 내부 곱셈이라는 뜻이 됩니다.
# 벡터의 곱셈에는 총 3가지가 있다.
# 스칼라곱, 내적, 외적
# 내적은 벡터의 곱셈 중 하나인데 연산 결과가 두 벡터와 같은 공간에 있기 때문에, 내적이라고 부른다.
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# 벡터 공간 = 내적 공간 : 벡터의 내적 : 둘 다 결과는 219
print(v.dot(w))
print(np.dot(v, w))

# 행렬과 벡터의 곱 : 둘 다 결과는 rank 1인 배열 [29 67]
print(x.dot(v))
print(np.dot(x, v))

# 행렬곱 : 둘 다 결과는 rank 2인 배열
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

# numpy 는 배열 연산에 유용하게 쓰이는 많은 함수를 제공합니다. 가장 유용한 건 'sum' 입니다.
x = np.array([[1, 2], [3, 4]])

print(np.sum(x))    # 모든 요소를 합한 값을 연산. 출력 "10"
print(np.sum(x, axis=0))    # 각 열에 대한 합을 연산. 출력 "[4 6]"
print(np.sum(x, axis=1))    # 각 행에 대한 합을 연산. 출력 "[3, 7]"
# Numpy 가 제공하는 모든 수학함수의 목록은 다른 문서를 참조바랍니다.

# 4. 브로드캐스팅 (나중에,,)


# RNN






