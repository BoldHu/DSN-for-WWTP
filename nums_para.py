# calculat the numbers of parameters
import torch
from thop import profile

mynet = torch.load('models/DSN1_model_X_kla480.mat_X_mu0.7.mat.pth')
mynet = mynet.to('cuda')

input = torch.randn(32,15)
input = input.to('cuda')
flops, params = profile(mynet, inputs=(input, 'source', 'all', 0))
print('flops:', flops)
print('params:', params)