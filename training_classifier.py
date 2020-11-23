
"""

분류기 학습하기
지금까지 어떻게 신경망을 정의하고, 손실을 계산하며 또 가중치를 갱신하는지 배웠습니다.
그렇다면 데이터는 어떻게 하나요?
일반적으로 이미지나 텍스트, 오디오나 비디오 데이터를 다룰 것인데, 이러한 데이터는 표준 Python 파키지를 사용하여 불러온 후,
Numpy 배열로 변환하면 됩니다.
그리고 그 배열을 torch.*Tensor 로 변환하면 됨.
데이터를 불러와서 이를 numpy 배열로 변환하고 이를 텐서로 변환하는거 해보고
이를 RNN 으로 돌리기
모두의 말뭉치

+) Optional: Data Parallelism

: 일단 뛰어넘음
"""
