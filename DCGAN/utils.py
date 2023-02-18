import numpy as np

from sklearn.metrics import accuracy_score

import torch
from torch.autograd.variable import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt

# setting the device to "mps" instead of default "cpu"
device = torch.device("mps" if torch.backends.mps.is_available else "cpu")


# helper function to create noise
def create_noise(n_samples, noise_size=128):
    return Variable(torch.randn(n_samples, noise_size, 1, 1)).to(device)

# helper function to create ones array
def create_ones(n_samples):
    return Variable(torch.ones(n_samples)).to(device)

# helper function to create zeros array
def create_zeros(n_samples):
    return Variable(torch.zeros(n_samples)).to(device)

# helper function to evaluate the generator
def generate_images(generator, save_path):
    # create noise_vec
    noise_vec = create_noise(64, generator.nz)

    # generate images from noise
    with torch.no_grad():
        generated_images = generator(noise_vec)
        generated_images = generated_images.cpu()

    # save the images as a grid of size (n_rows, n_samples//n_rows)
    save_image(generated_images, save_path, normalize=True, nrow=8)