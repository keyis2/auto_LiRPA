"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
import pickle

## Step 1: Define computational graph by implementing forward()
class UnitModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UnitModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the dimensions for the unit model
input_dim = 4  # Input: current state (x, y) and previous state (x', y')
hidden_dim = 8  # Hidden layer dimension (you can adjust this as needed)
output_dim = 2  # Output: predicted next state (x'', y'')

# Create an instance of the unit model
model = UnitModel(input_dim, hidden_dim, output_dim)

# Optionally, load the pretrained weights.
checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), 'pretrained/unit_model.pth'),
    map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
# model.eval()

## Step 2: Prepare dataset as usual
# Load the dataset from the saved file
with open(os.path.join(os.path.dirname(__file__), 'data/dataset.pkl'), 'rb') as file:
    dataset = pickle.load(file)

dataset = [(torch.tensor(data[0]), torch.tensor(data[1])) for data in dataset]

# For illustration we only use N time step from dataset
N=2
input_data = torch.cat([data[0] for data in dataset[:N]], dim=0).view(N,4)
next_state = torch.cat([data[1] for data in dataset[:N]], dim=0).view(N,2)

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph,
# and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(input_data), device=input_data.device)
print('Running on', input_data.device)

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.1
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
input_data = BoundedTensor(input_data, ptb)
# Get model prediction as usual
pred = lirpa_model(input_data)


## Step 5: Compute bounds for final output
for method in [
        'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)',
        'CROWN-Optimized (alpha-CROWN)']:
    print('Bounding method:', method)
    if method != 'IBP':
        continue
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    lb, ub = lirpa_model.compute_bounds(x=(input_data,), method=method.split()[0])
    for i in range(N):
        print(f'time step {i} input state {input_data[i].tolist()} predicted next state {pred[i].tolist()}')
        for j in range(output_dim):
            print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} (GT): {gt}'.format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), gt = next_state[i][j].item()))
    print()
