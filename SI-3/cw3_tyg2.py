import numpy as np


class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x


# funkcja AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])

p1 = Perceptron(input_size=2)
p1.fit(X, d)
print("Wagi dla funkcji AND:")
print(p1.W)

# funkcja NOT
X = np.array([[0], [1]])
d = np.array([1, 0])

p2 = Perceptron(input_size=1)
p2.fit(X, d)
print("Wagi dla funkcji NOT:")
print(p2.W)

# funkcja XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 1, 1, 0])

# warstwa ukryta
p1_hidden = np.array([p1.predict(x) for x in X]).T
p2_hidden = np.array([p2.predict(x[1]) for x in X]).T
X_hidden = np.concatenate((p1_hidden, p2_hidden), axis=1)

# warstwa wyjściowa
p3 = Perceptron(input_size=2)
p3.fit(X_hidden, d)

print("Wagi dla funkcji XOR:")
print("Wagi dla warstwy ukrytej:", p1_hidden, p2_hidden)
print("Wagi dla warstwy wyjściowej:", p3.W)