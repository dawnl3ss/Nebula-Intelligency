import numpy as np

class NeuronalSystem(object):

    def __init__(self):
        self.neurone_input = 2
        self.neurone_hidden = 3
        self.neurone_output = 1
        self.W1 = np.random.randn(self.neurone_input, self.neurone_hidden) # Matrice de 2x3
        self.W2 = np.random.randn(self.neurone_hidden, self.neurone_output) # Matrice de 3x1

    def ia_forward(self, x):
        return self.sigmoid(np.dot(self.sigmoid(np.dot(x, self.W1)), self.W2))

    def sigmoid(self, number):
        return 1 / (1 + np.exp(-number))

    def inversed_sigmoid(self, number):
        return number * (1 - number)

    def ia_backward(self, x, y, o):
        self.W1 += x.T.dot(((y - o) * self.inversed_sigmoid(o).dot(self.W2.T)) * self.inversed_sigmoid(o))
        self.W2 += x.T.dot((y - o) * self.inversed_sigmoid(o))

    def handle(self, x, y):
        self.ia_backward(x, y, self.ia_forward(x))