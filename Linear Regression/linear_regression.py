import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")
_X = df['year'].values.tolist()
_y = df['필지'].values.tolist()


X = np.array(_X).reshape(-1, 1)  # 독립 변수
y = np.array(_y)  # 종속 변수


# 선형 회귀 모델 생성
model = LinearRegression()

# 데이터 학습
model.fit(X, y)

# 회귀 계수 및 절편 출력
print('회귀 계수 (기울기):', model.coef_)
print('절편:', model.intercept_)

# 학습된 모델로 예측
y_pred = model.predict(X)

# 결과 시각화
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()