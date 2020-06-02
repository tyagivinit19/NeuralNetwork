import numpy as np
import random
import random

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

    def sigmoid(self, mat):

        return 1 / (1 + np.exp(-mat))

    def feedforward(self, inputs):
        inputs = np.array(inputs)
        sh_inputs = inputs.shape
        inputs = inputs.reshape(sh_inputs[0], 1)

        hidden = self.weights_ih.dot(inputs)
        # hidden = inputs.dot(self.weights_ih)
        # print("hidd", hidden.shape)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)
        # print(hidden.shape)

        outputs = self.weights_ho.dot(hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = self.sigmoid(outputs)


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
        hidden = self.sigmoid(hidden)
        # print(hidden.shape)

        outputs = self.weights_ho.dot(hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = self.sigmoid(outputs)

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
