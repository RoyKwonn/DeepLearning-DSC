# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt # matplotlib의 pyplot 기
import numpy as np # 숫자 배열 가능


def draw_linear_graph(m = 1, n = 1):
    # 인자를 입력하지 않는 경우, default 값으로 m=1, n=1 설정
    x = np.arange(0,10,2) # x값 : 0부터 10까지 2간격의 수
    y = [(m*num + n) for num in x] # y = mx + n이라는 일차 방정식 그래프
    plt.plot(x,y) # x, y 값을 담은 그래프 object 만들기
    title = '[실습] 선형회기 - 최소 제곱법 : y = ' +  str(m) + 'x + ' + str(n)
    plt.title(title)

#공부시간 X와 성적 Y의 리스트를 만듭니다.
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:", mx)
print("y의 평균값:", my)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])

# 기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)

# 기울기와 y 절편 구하기
a = dividend / divisor
b = my - (mx*a)


# 출력으로 확인
print("기울기 a =", a)
print("y 절편 b =", b)

# 그래프로 나타내 봅니다.
plt.figure(figsize=(8,5))
plt.grid() # 좌표에 grid 표시
plt.xlabel('공부 시간') # x축 이름
plt.ylabel('시험 점수') # y축 이름
plt.scatter(x, y)
draw_linear_graph(a, b)
plt.show()
