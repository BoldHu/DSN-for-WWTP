import torch
from model_compat import DSN

# Create an instance of the DSN class
model = DSN()

# Create some input data
input_data = torch.randn(32, 15)  # Assuming batch size of 32 and input size of 15

# Test the forward method with mode='source' and rec_scheme='share'
result = model.forward(input_data, mode='source', rec_scheme='share')
print(result)  # Print the result

# Test the forward method with mode='target' and rec_scheme='all'
result = model.forward(input_data, mode='target', rec_scheme='all')
print(result)  # Print the result

# Test the forward method with mode='target' and rec_scheme='private'
result = model.forward(input_data, mode='target', rec_scheme='private')
print(result)  # Print the result