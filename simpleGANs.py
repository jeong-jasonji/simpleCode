import torch
import torch.nn as nn
from torchvision import transforms

from base import simpleDataloader
from options.experimentOptions import GANTrainOptions
from base.simpleGAN_models import simpleDCGANGenerator, simpleDCGANDiscriminator
from base.simpleGAN_train import train_GAN

# code for simple linear GAN with or without condition
# python simpleGANs.py
opt = GANTrainOptions().parse()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Initialize generator and discriminator
print('Initialize generator and discriminator')
gen = simpleDCGANGenerator(opt.img_shape, condition=opt.conditioned, selu=opt.selu, nl=opt.num_classes)
gen.apply(weights_init)
dis = simpleDCGANDiscriminator(opt.img_shape, gen.n_layers, condition=gen.condition, nl=gen.nl, selu=gen.selu, acgan=opt.ACGAN)
dis.apply(weights_init)

print(gen)
print(dis)

exit()

models = {'Generator': gen.cuda(), 'Discriminator': dis.cuda()}
# GAN Loss functions
print('Initialize GAN loss functions')
validity_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
criterions = {'validity': validity_loss.cuda(), 'auxiliary': auxiliary_loss.cuda()}

# Configure dataloader
print('Configure dataloader')
transforms_train = transforms.Compose([
    transforms.Resize((opt.img_shape[-1], opt.img_shape[-1])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_GANtrain, transforms_train)
print('training size: {}'.format(len(dataloader)), file=opt.log)

# Optimizers
print('Configure Optimizers')
optimizer_G = torch.optim.Adam(models['Generator'].parameters(), lr=opt.g_lr, betas=(opt.beta_1, opt.beta_2))
optimizer_D = torch.optim.Adam(models['Discriminator'].parameters(), lr=opt.d_lr, betas=(opt.beta_1, opt.beta_2))
optimizers = {'Generator': optimizer_G, 'Discriminator': optimizer_D}

train_GAN(opt, dataloader, models, criterions, optimizers)
