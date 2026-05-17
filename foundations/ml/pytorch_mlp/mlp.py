import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()  # a list to hold the layers of the network
        sizes = layer_sizes
        for i in range(len(layer_sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)  # last layer without activation
    

# model = MLP([3, 4, 4, 1])
# xs = [
#     [2.0,  3.0, -1.0],
#     [3.0, -1.0,  0.5],
#     [0.5,  1.0,  1.0],
#     [1.0,  1.0, -1.0],
# ]
# ys = [1.0, -1.0, -1.0, 1.0]
# xenc = torch.tensor(xs)
# yenc = torch.tensor(ys).unsqueeze(1)

# for step in range(200):
#     #forward pass
#      # make ys a column vector
#     pred = model(xenc)
#     loss = F.mse_loss(pred, yenc)  # mean squared error loss

#     for param in model.parameters():
#         param.grad = None

#     #backward pass
#     loss.backward()  # compute gradients


#     for param in model.parameters():
#         param.data -= 0.01 * param.grad  # update parameters with gradient descent

#     if step % 20 == 0:                                                                                                              
#       print(f"step {step} | loss {loss.item():.6f}")   