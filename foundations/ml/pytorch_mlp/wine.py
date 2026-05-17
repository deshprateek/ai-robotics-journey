from sklearn.datasets import load_wine
import torch

from mlp import MLP

def load_data():
    data = load_wine()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)  # target labels as integers for classification
    return X, y

# print(load_data()[1].shape)  # should print (178, 13) — 178 examples, each with 13 features


model = MLP([13, 16, 16, 3])  # input size 13 (features), two hidden layers of 16 neurons each, output size 3 (classes)
xs, ys = load_data()
idx = torch.randperm(len(xs))                                                                                                                               
xs, ys = xs[idx], ys[idx] 
x_train = xs[:150]  # first 150 examples for training
y_train = ys[:150]  # corresponding labels for training
x_test = xs[150:]  # remaining examples for testing
y_test = ys[150:]  # corresponding labels for testing
mean = x_train.mean(dim=0) # compute mean and std on training data only, then use those to normalize both train and test data                                                                                                                                  
std = x_train.std(dim=0) # this prevents information from the test set leaking into the training process, which could lead to overfitting and unrealistic performance estimates                                                                                                                               
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std  # use train mean/std, not test    

for step in range(300):
    pred = model(x_train)
    loss = torch.nn.functional.cross_entropy(pred, y_train)  # cross-entropy loss for classification

    for param in model.parameters():
        param.grad = None

    loss.backward()  # compute gradients

    for param in model.parameters():
        param.data -= 0.01 * param.grad  # update parameters with gradient descent

    if step % 20 == 0:                                                                                                              
      print(f"step {step} | loss {loss.item():.6f}")


with torch.no_grad():                                     
      acc = (model(x_test).argmax(dim=1) == y_test).float().mean()                                                                                            
      print(f"test accuracy: {acc:.2%}")     