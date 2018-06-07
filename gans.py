import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import argparse

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def parseArguments():
    parser = argparse.ArgumentParser(description = "CS231N")
    parser.add_argument('--ls', action = "store_true", help = "Include if using least square loss")
    parser.add_argument('--cycle', action = "store_true", help = "Include if using cycle loss rather than color gradient loss")
    parser.add_argument('--b', dest = "batch_size", type = int, nargs = '?', default = -1, help = "Batch size")
    parser.add_argument('--r', dest = "reg", type = float, nargs = '?', default = 1.0, help = "Regularization")
    parser.add_argument('--e', dest = "epochs", type = int, nargs = '?', default = 100, help = "Number of epochs")
    parser.add_argument('--lr', dest = "lr", type = float, nargs = '?', default = 1e-3, help = "learning rate")

    args = parser.parse_args()  
    return args.ls, args.cycle, args.batch_size, args.reg, args.epochs, args.lr

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    return 2 * (torch.rand(batch_size, dim) - 0.5)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=3, H=28, W=32):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    return optimizer

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    N, _ = logits_real.size()
    M, _ = logits_fake.size()
    real_loss = bce_loss(logits_real, torch.ones(N).type(dtype))
    fake_loss = bce_loss(logits_fake, torch.zeros(M).type(dtype))
    loss = real_loss + fake_loss
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    M, _ = logits_fake.size()
    loss = bce_loss(logits_fake, torch.ones(M).type(dtype))
    return loss

def cycle_loss(grad, fake_images, rep_images, reg):
    loss = reg * torch.mean(torch.abs(fake_images - rep_images))
    return loss

def grad_loss(grad, fake_images, rep_images, reg):
    print(reg)
    loss = reg * torch.mean(torch.abs(grad * fake_images - grad * rep_images))
    return loss

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = 0.5 * (((scores_real - 1) ** 2).mean() + (scores_fake ** 2).mean())
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        Unflatten(-1, 3, 32, 32),
        nn.Conv2d(3, 32, 5),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1600, 1600),
        nn.LeakyReLU(),
        nn.Linear(1600, 1)
    )

def build_dc_generator():
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        Flatten(),
        nn.Linear(3 * 32 * 32, 1600),
        nn.ReLU(),
        nn.BatchNorm1d(1600),
        nn.Linear(1600, 8 * 8 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(8 * 8* 128),
        Unflatten(-1,128,8,8),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 3, 4 , stride=2, padding=1),
        nn.Tanh(),
        Flatten()
    )

def run_a_gan(X_a_train, X_h_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss, regularization, X_h_train_grad, reg, show_every=10, 
              batch_size=128, noise_size=96, num_epochs=100):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    length = len(X_a_train)
    num_batches = int(length / batch_size + 1) if length % batch_size != 0 else int(length / batch_size)
    for epoch in range(num_epochs):
        for i in range(num_batches):
            start = i * batch_size
            end = (i+1) * batch_size if (i+1)*batch_size < length else length
            X_a_train_b = X_a_train[start:end] 
            D_solver.zero_grad()
            real_data = X_a_train_b.type(dtype)
            real_data = 2 * (real_data - 0.5)
            logits_real = D(real_data).type(dtype)

            X_h_train_b = X_h_train[start:end] 
            fake_data = X_h_train_b.type(dtype)
            fake_data = 2* (fake_data - 0.5)
            fake_images = G(fake_data).type(dtype).detach()
            fake_images = fake_images.view(end-start, 3, 32, 32)
            logits_fake = D(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            fake_images = G(fake_data).type(dtype)
            fake_images = fake_images.view(end-start, 3, 32, 32)
            gen_logits_fake = D(fake_images)
            g_error = generator_loss(gen_logits_fake)
            
            X_h_train_grad_b = X_h_train_grad[:,start:end,:,:,:] 
            g_error += regularization(X_h_train_grad_b, fake_data, fake_images, reg=reg)
            g_error.backward()
            G_solver.step()



        if (iter_count % show_every == 0):
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
            print()
        iter_count += 1
    return G



def main():
        global dtype, learning_rate
        dtype = torch.FloatTensor

        ls, cycle, batch_size, reg,epochs, learning_rate = parseArguments()
        d_loss = ls_discriminator_loss if ls else discriminator_loss
        g_loss = ls_generator_loss if ls else generator_loss
        regularization = cycle_loss if cycle else grad_loss

        X_a_train = torch.from_numpy(np.load('X_a_train.npy'))
        X_a_val = torch.from_numpy(np.load('X_a_val.npy'))
        y_a_train = torch.from_numpy(np.load('y_a_train.npy'))
        y_a_val = torch.from_numpy(np.load('y_a_val.npy'))
        X_h_train = np.load('X_h_train.npy')
        X_h_train_grad = torch.from_numpy(np.array(np.gradient(X_h_train)))
        X_h_train = torch.from_numpy(X_h_train)
        X_h_val = torch.from_numpy(np.load('X_h_val.npy'))
        y_h_train = torch.from_numpy(np.load('y_h_train.npy'))
        y_h_val = torch.from_numpy(np.load('y_H_val.npy'))

        if batch_size == -1:
            batch_size = len(X_a_train)

        X_a_train = X_a_train.type('torch.FloatTensor')
        X_a_val = X_a_val.type('torch.FloatTensor')
        y_a_train = y_a_train.type('torch.LongTensor')
        y_a_val = y_a_val.type('torch.LongTensor')
        X_h_train = X_h_train.type('torch.FloatTensor')
        X_h_train_grad = X_h_train_grad.type('torch.FloatTensor')
        X_h_val = X_h_val.type('torch.FloatTensor')
        y_h_train = y_h_train.type('torch.LongTensor')
        y_h_val = y_h_val.type('torch.LongTensor')

        D_DC = build_dc_classifier().type(dtype) 
        D_DC.apply(initialize_weights)
        G_DC = build_dc_generator().type(dtype)
        G_DC.apply(initialize_weights)
        G2_DC = build_dc_generator().type(dtype)
        #G2_DC.apply(initialize_weights)

        D_DC_solver = get_optimizer(D_DC)
        G_DC_solver = get_optimizer(G_DC)
        #G2_DC_solver = get_optimizer(G2_DC)


        G = run_a_gan(X_a_train, X_h_train, D_DC, G_DC, D_DC_solver, G_DC_solver,
                d_loss, g_loss, regularization, X_h_train_grad, reg, batch_size=batch_size, num_epochs=epochs)

        with torch.no_grad():
                fake_data = X_h_train.type(dtype)
                fake_data = 2* (fake_data - 0.5)
                X_h_train = G(fake_data).type(dtype).view(len(X_h_train), 3, 32, 32)
                fake_data = X_h_val.type(dtype)
                fake_data = 2* (fake_data - 0.5)
                X_h_val = G(fake_data).type(dtype).view(len(X_h_val), 3, 32, 32)

        X_train = np.concatenate((X_a_train, X_h_train), axis=0)
        y_train = np.concatenate((y_a_train, y_h_train), axis=0)
        X_val = np.concatenate((X_a_val, X_h_val), axis=0)
        y_val = np.concatenate((y_a_val, y_h_val), axis=0)

        np.save('X_train_gan.npy', X_train)
        np.save('y_train_gan.npy', y_train)
        np.save('X_val_gan.npy', X_val)
        np.save('y_val_gan.npy', y_val)




if __name__ == "__main__":
        main()