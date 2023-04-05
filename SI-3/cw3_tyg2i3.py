import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
        self.activation_fn = np.vectorize(self.unit_step_fn)

    def unit_step_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a.all()

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

    # Definicja funkcji aktywacji sigmoidalnej
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Definicja pochodnej funkcji aktywacji sigmoidalnej
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backpropagation_XOR(self, X, y, num_iterations):
        # Inicjalizacja wag sieci
        np.random.seed(1)
        weights_0 = 2 * np.random.random((2,4)) - 1
        weights_1 = 2 * np.random.random((4,1)) - 1

        # Uczenie sieci za pomocą propagacji wstecznej
        for i in range(num_iterations):
            # Propagacja wprzód
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, weights_0))
            layer_2 = self.sigmoid(np.dot(layer_1, weights_1))

            # Obliczenie błędu
            layer_2_error = y - layer_2

            # Propagacja wsteczna
            layer_2_delta = layer_2_error * self.sigmoid_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(weights_1.T)
            layer_1_delta = layer_1_error * self.sigmoid_derivative(layer_1)

            # Aktualizacja wag
            weights_1 += layer_1.T.dot(layer_2_delta)
            weights_0 += layer_0.T.dot(layer_1_delta)

        return weights_0, weights_1

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d_and = np.array([0, 0, 0, 1])
d_not = np.array([1, 0, 1, 0])

and_perceptron = Perceptron(input_size=2)
and_perceptron.fit(X, d_and)

not_perceptron = Perceptron(input_size=1)
not_perceptron.fit(X[:,0], d_not)

print("AND Perceptron weights:", and_perceptron.W)
print("NOT Perceptron weights:", not_perceptron.W)

# Testowanie perceptronów
test_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
print("AND perceptron predictions:")
for i in range(test_inputs.shape[0]):
    print(test_inputs[i], and_perceptron.predict(test_inputs[i]))

print("NOT perceptron predictions:")
for i in range(test_inputs.shape[0]):
    print(test_inputs[i, 0], not_perceptron.predict(test_inputs[i, 0]))

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

perceptron = Perceptron(input_size=2)
weights_0, weights_1 = perceptron.backpropagation_XOR(X, y, 60000)

print("Wagi warstwy ukrytej:")
print(weights_0)
print("Wagi warstwy wyjściowej:")
print(weights_1)
