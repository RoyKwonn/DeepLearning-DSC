import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#데이터 셋 설정
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

print(X)
print(y)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

batch_size = 1
epochs = 1000

model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

# 7. 모델 테스트하기
predict1 = model.predict(np.array([[0,0],]))
predict2 = model.predict(np.array([[0,1],]))
predict3 = model.predict(np.array([[1,0],]))
predict4 = model.predict(np.array([[1,1],]))

print(predict1)
print(predict2)
print(predict3)
print(predict4)