# import numpy
import numpy as np
import random
import copy


def sigmoid(mat):

    return 1 / (1 + np.exp(-mat))


class NeuralNetwork:
    def __init__(self, numI, numH, numO):
        self.inputNodes = numI
        self.hiddenNodes = numH
        self.outputNodes = numO
        self.weights_ih = self.matrix(self.hiddenNodes, self.inputNodes)
        self.weights_ho = self.matrix(self.outputNodes, self.hiddenNodes)
        self.bias_h = self.matrix(self.hiddenNodes, 0)
        self.bias_o = self.matrix(self.outputNodes, 0)
        self.learning_rate = 0.1

    def predict(self, inputs):
        inputs = np.array(inputs)
        sh_inputs = inputs.shape
        inputs = inputs.reshape(sh_inputs[0], 1)

        hidden = self.weights_ih.dot(inputs)
        # hidden = inputs.dot(self.weights_ih)
        # print("hidd", hidden.shape)
        hidden = np.add(hidden, self.bias_h)
        hidden = sigmoid(hidden)
        # print(hidden.shape)

        outputs = self.weights_ho.dot(hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = sigmoid(outputs)

        return outputs

    def train(self, inputs, targets):

        inputs = np.array(inputs)
        sh_inputs = inputs.shape
        inputs = inputs.reshape(sh_inputs[0], 1)

        targets = np.array(targets)
        sh_targets = targets.shape
        targets = targets.reshape(sh_targets[0], 1)

        # guess outputs
        hidden = self.weights_ih.dot(inputs)
        # hidden = inputs.dot(self.weights_ih)
        # print("hidd", hidden.shape)
        hidden = np.add(hidden, self.bias_h)
        hidden = sigmoid(hidden)
        # print(hidden.shape)

        outputs = self.weights_ho.dot(hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = sigmoid(outputs)

        # output errors
        output_error = np.subtract(targets, outputs)
        # return error
        # gradient
        gradients = outputs * (1 - outputs)
        # print("hello",targets.shape, outputs.shape)
        gradients = gradients * output_error
        gradients = gradients * self.learning_rate

        hidden_T = np.transpose(hidden)
        weight_ho_deltas = gradients.dot(hidden_T)

        # Adjest the weights and biases
        self.weights_ho = np.add(self.weights_ho, weight_ho_deltas)
        self.bias_o = np.add(self.bias_o, gradients)

        # hidden layer errors
        who_t = np.transpose(self.weights_ho)
        hidden_error = who_t.dot(output_error)
        # print("hidden", hidden.shape)
        # hidden gradient
        hidden_gradients = hidden * (1 - hidden)
        hidden_gradients = hidden_gradients * hidden_error
        hidden_gradients = hidden_gradients * self.learning_rate

        # input to hidden deltas
        inputs_T = np.transpose(inputs)
        weight_ih_deltas = hidden_gradients.dot(inputs_T)

        self.weights_ih = np.add(self.weights_ih, weight_ih_deltas)
        self.bias_h = np.add(self.bias_h, hidden_gradients)

        # print("output: ", outputs)
        # print("targets: ", targets)
        # print("error: ", outpit_error)

    def matrix(self, rows, cols):
        if cols == 0:
            mat = np.random.random(rows)
            sh = mat.shape
            for i in range(sh[0]):
                mat[i] = random.uniform(-1, 1)
            mat = mat.reshape(rows, 1)
            return mat

        mat = np.random.random((rows, cols))
        sh = mat.shape
        for i in range(sh[0]):
            for j in range(sh[1]):
                mat[i, j] = random.uniform(-1, 1)

        return mat

    def copy(self):
        child = NeuralNetwork(self.inputNodes, self.hiddenNodes, self.outputNodes)

        child.inputNodes = copy.deepcopy(self.inputNodes)
        child.hiddenNodes = copy.deepcopy(self.hiddenNodes)
        child.outputNodes = copy.deepcopy(self.outputNodes)
        child.weights_ih = copy.deepcopy(self.weights_ih)
        child.weights_ho = copy.deepcopy(self.weights_ho)
        child.bias_h = copy.deepcopy(self.bias_h)
        child.bias_o = copy.deepcopy(self.bias_o)
        child.learning_rate = copy.deepcopy(self.learning_rate)

        return child

    def mutate(self, mutation_rate):
        self.weights_ih = mutate(self.weights_ih, mutation_rate)
        self.weights_ho = mutate(self.weights_ho, mutation_rate)
        self.bias_h = mutate(self.bias_h, mutation_rate)
        self.bias_o = mutate(self.bias_o, mutation_rate)


def mutate(mat, mutation_rate):
    sh = mat.shape
    for i in range(sh[0]):
        for j in range(sh[1]):
            rand = random.randint(1, 100)
            if rand <= mutation_rate:
                # Gaussian Random number generator
                mat[i, j] = mat[i, j] + np.random.normal(0, 0.1)

    return mat
