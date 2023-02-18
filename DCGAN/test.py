import torch
from torch.autograd.variable import Variable

bs = 5

real_label = 1
fake_label = 0

real_label = torch.full((bs,), real_label)
print(real_label)
print(real_label.shape)

real_label.fill_(fake_label)
print(real_label)
print(real_label.shape)

# helper function to create ones array
def create_ones(n_samples):
    return Variable(torch.ones(n_samples))

# helper function to create zeros array
def create_zeros(n_samples):
    return Variable(torch.zeros(n_samples))


new_label = create_ones(5)
print(new_label)
print(new_label.shape)