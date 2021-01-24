import numpy as np
 
# 가중치와 바이어스
w11 = np.array([-2,-2])
w12 = np.array([2,2])
w2 = np.array([1,1])

# 편향(bias)의 값 할당
b1 = 3
b2 = -1
b3 = -1

# 그냥 시그모이드
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

print(np.e)
print(sigmoid(3))
print(sigmoid(1))

# 다층 퍼셉트론 함수 정의
def MLP(x, w, b):
    # 레이어 결과값을 numpy의 sum함수를 통하여 계산
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1


# 은닉층 연산에 대한 함수

# NAND 게이트 
def NAND(x1, x2):
    # 첫번째 노드의 결과값을 반환
    return MLP(np.array([x1, x2]), w11, b1)

# OR 게이트
def OR(x1, x2):
    # 두번째 노드의 결과값을 반환
    return MLP(np.array([x1, x2]), w12,b2)

# AND 게이트
def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2,b3)

# XOR 게이트
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


if __name__ == '__main__':
    for x in [(0,0), (1,0), (0,1), (1,1)]:
        y = XOR(x[0], x[1])
        print("입력 값: " + str(x) + "출력 값" + str(y))
