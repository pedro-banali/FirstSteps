import numpy as np
import random
import matplotlib.pyplot as plt

def gradientDescent(x, theta, b, learningRate, arrayRow):
    for i in range(0, arrayRow):
        y = x[i][1]
        guess = theta * x[i][0] + b
        error = y - guess
        theta += (error * x[i][0]) * learningRate
        b += error * learningRate
    return theta, b


arrayRow = 100
b = 25
x = np.ones(shape=(arrayRow, 2))
y = np.zeros(shape=arrayRow)

for i in range(0, arrayRow):
    x[i][0] = random.uniform(0, 100)
    x[i][1] = i
    y[i] = i + b + random.uniform(0, 1) * 10

# plt.plot(x, 'ro')
# plt.axis([0, 100, 0, 100])
# plt.show()
result0, result1 = gradientDescent(x, 0, 0, 0.05, arrayRow)
plt.plot(x[:, 0], x[:, 1], 'ro')
plt.plot([0, result0])
plt.axis([0, 110, 0, 110])
plt.show()






