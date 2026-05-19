import torch

# Let's test the FNN
import deepxde as dde

net = dde.nn.FNN(
    layer_sizes=[5, 128, 128, 128, 128, 2],
    activation="tanh",
    kernel_initializer="Glorot normal"
)

device = torch.device("cuda")
net.to(device)
net.to("cpu")

try:
    x = torch.randn(10, 5, device="cpu")
    y = net(x)
    print("FNN to('cpu') works.")
except Exception as e:
    print(f"Error: {e}")
