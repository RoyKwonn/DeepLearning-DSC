from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

import os

#오류 발생을 위해 경로를 설정
scriptpath_noname = os.path.realpath( os.path.dirname(__file__) )

# seed 값 생성
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터 로드
dataset = numpy.loadtxt(scriptpath_noname+'/'+'pima-indians-diabetes.csv', delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))


