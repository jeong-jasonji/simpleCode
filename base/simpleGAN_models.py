import torch
import torch.nn as nn
import numpy as np

class simpleLinearGenerator(nn.Module):
    def __init__(self, opt):
        super(simpleLinearGenerator, self).__init__()

        self.img_shape = opt.img_shape
        self.conditioned = opt.conditioned
        self.label_emb = nn.Embedding(opt.num_classes, opt.num_classes)

        self.n_layers = self.calculate_linear_layers(np.prod(self.img_shape))

        # make blocks for the generator
        start_dim = opt.latent_dim + opt.num_classes if self.conditioned else opt.latent_dim
        self.model = nn.Sequential()
        self.model.add_module('Flatten', self.add_linear_block(start_dim, 128, normalize=False))
        last_layer = 128
        for layer in range(self.n_layers):
            self.model.add_module('UPblock_{}'.format(layer + 1), self.add_linear_block(last_layer, last_layer * 2))
            last_layer = last_layer * 2
        self.model.add_module('LinearShapeMatch', nn.Linear(last_layer, int(np.prod(self.img_shape))))
        self.model.add_module('Tanh', nn.Tanh())

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1) if self.conditioned else noise
        img = self.model(gen_input)
        img = img.view(img.size(0), * self.img_shape)
        return img

    def calculate_linear_layers(self, end_dim):
        """
        calculate the number of linear layers to make (doubling every time)
            start_dim = always 128 (based off of eriklindernoren)
            end_dim = img_h * img_w * img_channels
        """
        n_layers = -1
        last_dim = 128
        not_close = True
        while not_close:
            last_dim = last_dim * 2
            n_layers += 1
            if end_dim - last_dim < 0:
                not_close = False
        return n_layers

    def add_linear_block(self, in_feat, out_feat, normalize=True):
        """
        linear upsampling block from:
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
        """
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

class simpleLinearDiscriminator(nn.Module):
    def __init__(self, opt, n_layers):
        super(simpleLinearDiscriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.num_classes, opt.num_classes)
        self.conditioned = opt.conditioned
        self.n_layers = n_layers
        self.img_shape = opt.img_shape

        self.model = nn.Sequential()
        self.in_dim = opt.num_classes + int(np.prod(self.img_shape)) if self.conditioned else int(np.prod(self.img_shape))
        self.model.add_module('Flatten', nn.Sequential(nn.Linear(self.in_dim, 512),
                                                       nn.LeakyReLU(0.2, inplace=True)))
        for i in range(self.n_layers):
            self.model.add_module('LinearDropout_{}'.format(i+1), self.add_linear_block())
        self.model.add_module('FinalBinary', nn.Linear(512, 1))

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1) if self.conditioned else img.view(img.size(0), -1)
        validity = self.model(d_in)
        return validity

    def add_linear_block(self):
        """
        linear classification block from:
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
        """
        layers = [nn.Linear(512, 512),
                  nn.Dropout(0.4),
                  nn.LeakyReLU(0.2, inplace=True)]

        return nn.Sequential(*layers)