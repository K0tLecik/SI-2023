import numpy as np
import matplotlib.pyplot as plt

x = np.array([2000, 2002, 2005, 2007, 2010])
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

def gradient_descent(x, y, learning_rate, iterations):
    m = 0
    b = 0
    while iterations > 0:
        y_predicted = m * x + b
        dm = (-2 / len(x)) * sum(x * (y - y_predicted))
        db = (-2 / len(x)) * sum(y - y_predicted)
        m -= learning_rate * dm
        b -= learning_rate * db
        iterations -= 1
    return m, b

learning_rate = 0.01
iterations = 1000
x_norm = (x - x.mean()) / x.std()
m, b = gradient_descent(x_norm, y, learning_rate, iterations)
m = round(m, 3)
b = round(b, 3)

year = (12 - b) / m
year = year * x.std() + x.mean()
year = int(round(year))

print("Model regresji liniowej: y = ", m, "* x + ", b)
print("Procent bezrobotnych przekroczy 12% w roku: ", year)

