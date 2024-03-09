import torch

from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import FeedForward

training_data = datasets.MNIST(
    root="mnist_data", 
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.MNIST(
    root="mnist_data", 
    train=False,
    download=True,
    transform=ToTensor())


# print(test_data[3817][0])
# print(test_data[1387][1])
# print(len(test_data))
# print(len(training_data))

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
for X, y in test_dataloader:
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    break

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

print(f"Using {device} device")

    
model = FeedForward().to(device)
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X) 
        loss = loss_fn(pred, y)

        loss.backward() #computes weight adjustments that are necessary
        optimizer.step() # updates weight in model
        optimizer.zero_grad() #resets optimizer for next iteration

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum(). item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Errror: \n Accuracy{(100*correct):0.1f}%, Avg loss:{test_loss:>8f}\n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    train(train_dataloader, model, loss_function, optimizer)
    test(test_dataloader, model, loss_function)

torch.save(model.state_dict(), "model.pth")