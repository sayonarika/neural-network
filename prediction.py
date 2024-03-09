from model import FeedForward
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from PIL import Image, ImageEnhance

test_data = datasets.MNIST(
    root="mnist_data", 
    train=False,
    download=True,
    transform=ToTensor())

device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ) 

model = FeedForward().to(device)
model.load_state_dict(torch.load("model.pth"))

total = 0

# for i in range(100):
#     pred = model(test_data[i][0].to(device))
#     print("Prediction:", pred.argmax(1).item())
#     print("Actual answer:", test_data[i][1])
#     if pred.argmax(1).item() == test_data[i][1]:
#         total += 1

# print(str(total)+"%")


with Image.open("badpictureof3.png") as im:
    # im = im.resize((28,28), Image.ANTIALIAS)
    im = im.convert('L')
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(9)
    data = list(im_output.getdata())

data = [round(n/255 * -1 + 1, 4) for n in data]

shoyuramen = torch.FloatTensor(data)
shoyuramen = shoyuramen.unsqueeze(0)
print(shoyuramen.shape)
shoyuramen = shoyuramen.to(device)
pred = model(shoyuramen).argmax(1)
print("The prediction", pred)

    

# print(test_data[0][0])


# test_data[0][1]
# test_data[0][1]

