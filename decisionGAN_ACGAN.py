# ACGAN code base from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py
# modified for my purposes to train a decisionGAN

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from base import simpleDataloader

# decisionGAN training
# python decisionGAN_ACGAN.py --uniform_loss uniform_Boundary --save_dir ./decisionGAN_2_3_5 --n_epochs 100

parser = argparse.ArgumentParser()
parser.add_argument("--uniform_loss", type=str, default='uniform_Boundary', help="loss function to use for uniform labels [none, uniform_MSE, uniform_CE, uniform_KL, uniform_Boundary]")
parser.add_argument("--uniform_beta", type=int, default=0, help="multiplication factor for uniform label loss to push generation to boundary")
parser.add_argument("--cuda", type=int, default=0, help="cuda device number to use")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpus to use")
parser.add_argument("--dataframe_GANtrain", type=str, default='/home/jjeong35/simpleCode/dataframes/coloredMNIST/GANtrain.csv', help="dataset to use MNIST: mnist, SVHN: svhn")
parser.add_argument("--save_dir", type=str, default='./decisionGAN_tests/', help="directory to save results")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--workers", type=int, default=4, help="number of workers")
parser.add_argument("--weighted_sampling", type=bool, default=False, help="use weighted sampling?")
parser.add_argument("--sample_interval", type=int, default=4000, help="interval between image sampling")
parser.add_argument("--cls_select", type=str, default='2,3,5', help='selecting specific classes to use only: e.g. 0  0,3,7, 0,8')
parser.add_argument('--img_shape', type=tuple, default=(3, 64, 64), help='image shape: (n_channels, height, width)')
opt = parser.parse_args()
print(opt)

opt.img_size = opt.img_shape[-1]
opt.channels = opt.img_shape[0]
opt.n_classes = len(opt.cls_select.split(','))

# set save directory
os.makedirs(opt.save_dir, exist_ok=True)
os.makedirs("{}/images/".format(opt.save_dir), exist_ok=True)

# set cuda device
cuda = torch.device(opt.cuda if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

# weight initialization function
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# define generator class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # label embedding
        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# define discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())  # predicting real/fake
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())  # predicting label

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
print('Initialize loss functions')
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
uniform_loss = torch.nn.MSELoss()
kldivergence_loss = torch.nn.KLDivLoss()

# Initialize generator and discriminator
print('Initialize models')
generator = Generator()
discriminator = Discriminator()

# put model in GPU
print('Setting to CUDA')
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
print('Initialize weights')
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure dataloader
print('Configure dataloader')
transforms_train = transforms.Compose([
    transforms.Resize((opt.img_shape[-1], opt.img_shape[-1])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_GANtrain, transforms_train)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
# put the beta into cuda
beta = torch.tensor(opt.uniform_beta).cuda().type(torch.cuda.FloatTensor)

def sample_image(n_row, epochs_done, save_dir):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "{}/images/e{}.png".format(save_dir, epochs_done), nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

# set initial values for losses
best_loss = 999
for epoch in range(opt.n_epochs):
    for i, (imgs, labels, img_id) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        # generate a uniform tensor for boundary generation
        BoundaryTensor = torch.tensor(np.ones([batch_size, opt.n_classes])*(1/opt.n_classes)).cuda().type(torch.cuda.FloatTensor)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        
        # use specific losses
        if opt.uniform_loss == 'uniform_MSE':  # uniform label loss
            g_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels) + (beta * uniform_loss(pred_label, BoundaryTensor))) / 3  
        elif opt.uniform_loss == 'uniform_KL':
            g_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels) + (beta * kldivergence_loss(pred_label, BoundaryTensor))) / 3
        elif opt.uniform_loss == 'uniform_Boundary':
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, BoundaryTensor))
        else:
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) # original g_loss
        
        #g_loss = g_loss.type(torch.cuda.FloatTensor)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2 

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        if opt.uniform_loss == 'uniform_MSE':  # uniform label loss
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels) + (beta * uniform_loss(fake_aux, BoundaryTensor))) / 3
        elif opt.uniform_loss == 'uniform_KL':
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels) + (beta * kldivergence_loss(fake_aux, BoundaryTensor))) / 3
        elif opt.uniform_loss == 'uniform_Boundary':
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, BoundaryTensor)) / 2
        else:
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2  # original d_fake_loss

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch+1, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
        # save model if total loss improves
        if d_loss.item() + g_loss.item() < best_loss:
            torch.save(generator, os.path.join(opt.save_dir, 'generator_best.pth'))
            torch.save(discriminator, os.path.join(opt.save_dir, 'discriminator_best.pth'))

    # save every epoch
    sample_image(n_row=opt.n_classes, epochs_done=(epoch+1), save_dir=opt.save_dir)
            
    # save model every epoch
    torch.save(generator, os.path.join(opt.save_dir, 'generator_e{}.pth'.format(epoch+1)))
    torch.save(discriminator, os.path.join(opt.save_dir, 'discriminator_e{}.pth'.format(epoch+1)))