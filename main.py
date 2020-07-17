from nn import *

training_data = [
    {
        "inputs": [0, 1],
        "targets": [1]
    },

    {
        "inputs": [1, 0],
        "targets": [1]
    },

    {
        "inputs": [0, 0],
        "targets": [0]
    },

    {
        "inputs": [1, 1],
        "targets": [0]
    },

]

# NeuralNetwork(inputNodes, hiddenNodes, outputNodes)
nn = NeuralNetwork(2, 4, 1)

# # inputs = np.array([[1, 0]])
# inputs = np.array([1, 0])
# sh_inputs = inputs.shape
# inputs = inputs.reshape(sh_inputs[0], 1)
# # targets = np.array([[1, 0]])
# targets = np.array([1, 0])
# sh_targets = targets.shape
# targets = targets.reshape(sh_targets[0], 1)
# # output = nn.feedforward(inputs)
# nn.train(inputs, targets)
# # print(output)

for i in range(500000):
    ran = random.randint(0, len(training_data)-1)
    data = training_data[ran]
    nn.train(data["inputs"], data["targets"])

print(nn.predict([0, 0]))
print(nn.predict([0, 1]))
print(nn.predict([1, 0]))
print(nn.predict([1, 1]))
