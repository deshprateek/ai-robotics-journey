import random  # used to generate random numbers for initializing weights and biases
from value import Value, draw_dot  # import our custom Value class that supports backpropagation


class Neuron:
    # a single neuron that takes n_inputs values and produces one output
    def __init__(self, n_inputs):
        # create one random weight for each input — weights control how much each input matters
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        # one bias value — shifts the output independently of the inputs
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # multiply each input by its corresponding weight, then add them all up starting from bias
        # e.g. w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # squash the result through tanh so the output is always between -1 and 1
        return act.tanh() if hasattr(act, 'tanh') else act

    def parameters(self):
        # return all learnable values in this neuron: all weights + the bias
        return self.w + [self.b]


class Layer:
    # a layer is just a group of neurons all receiving the same inputs
    def __init__(self, n_inputs, n_outputs):
        # create n_outputs neurons, each one independently processes the same n_inputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        # pass the input x through every neuron and collect their outputs as a list
        return [n(x) for n in self.neurons]

    def parameters(self):
        # gather parameters from every neuron in this layer into one flat list
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    # MLP = Multi-Layer Perceptron — a stack of layers forming a full neural network
    def __init__(self, n_inputs, layer_sizes):
        # build a list of sizes like [n_inputs, size1, size2, ...] to define each layer's shape
        sizes = [n_inputs] + layer_sizes
        # create one Layer for each consecutive pair of sizes, e.g. (3,4), (4,4), (4,1)
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        # pass the input through each layer one by one — output of one layer becomes input of the next
        for layer in self.layers:
            x = layer(x)
        # if the final layer has only one neuron, return the value directly instead of a list
        return x[0] if len(x) == 1 else x

    def parameters(self):
        # gather all parameters from every layer into one flat list — used during training to update weights
        return [p for layer in self.layers for p in layer.parameters()]


model = MLP(3, [4, 4, 1])  # create a neural network with 3 inputs, two hidden layers of 4 neurons each, and 1 output

# --- training data ---
# 4 examples, each with 3 inputs and a target output (what we want the network to predict)
xs = [
    [2.0,  3.0, -1.0],
    [3.0, -1.0,  0.5],
    [0.5,  1.0,  1.0],
    [1.0,  1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs for each example (the ground truth)

draw_dot(model(xs[0])).render('graph', view=True)  # visualize the computation graph for the first example

learning_rate = 0.01  # how big a step to take when updating weights — too big and it overshoots, too small and it's slow

# --- training loop ---
for step in range(100):  # repeat the process 100 times, each repetition is called an epoch/step

    # 1) FORWARD PASS — run each input through the network to get predictions
    predictions = [model(x) for x in xs]

    # 2) COMPUTE LOSS — measure how wrong the predictions are
    # we square the difference so it's always positive and penalises big errors more
    # e.g. if prediction=0.9 and target=1.0, loss contribution = (0.9 - 1.0)² = 0.01
    loss = sum((pred - y) ** 2 for pred, y in zip(predictions, ys))

    # 3) ZERO GRADIENTS — reset all gradients to 0 before backward pass
    # if we don't do this, gradients from the previous step accumulate and corrupt the update
    for p in model.parameters():
        p.grad = 0.0

    # 4) BACKWARD PASS — compute gradients of loss with respect to every weight and bias
    loss.backward()

    # 5) UPDATE WEIGHTS — nudge every parameter in the opposite direction of its gradient
    for p in model.parameters():
        p.data -= learning_rate * p.grad  # opposite direction = subtract gradient

    print(f"step {step:3d} | loss {loss.data:.6f}")  # print progress so we can see loss going down