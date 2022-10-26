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

class simpleDCGANGenerator(nn.Module):
    def __init__(self, final_img_size, condition=False, nz=100, nl=10, selu=True):
        super(simpleDCGANGenerator, self).__init__()
        """
        simple DCGAN with conditional elements based on ACGAN and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            default latent vector (nz) = 100
            default labels (nl) = 10
            default final image size (final_img_size) = [3, 64, 64]
                default channels (nc) = img_size[0]
                default generator feature map (ngf) = img_size[-1]
            default condition (condition) = bool
                conditioned generation or not
            default SELU (selu) = True
                to use SELU instead of batch norm and relu
        """
        self.img_size = final_img_size
        self.nz = nz
        self.nl = nl
        self.selu = selu
        self.condition = condition
        # calculate the number of layers needed for the output image
        self.n_layers, feat_mult = self.calc_layers(self.img_size[-1])

        # make the latent vector input branch
        self.input_branch_4x4 = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.img_size[-1] * feat_mult, 4, 1, 0, bias=False),
        )
        current_size = 4
        # make the label branch
        self.label_branch = nn.Sequential(
            # input is L (nl), going into a convolution
            nn.ConvTranspose2d(1, self.nl, 4, 1, 0, bias=False),
        )
        # make the generator model
        self.main = nn.Sequential()
        start_feat = self.img_size[-1] * feat_mult + self.nl if self.condition else self.img_size[-1] * feat_mult
        if self.selu:
            self.main.add_module('SELU', nn.SELU(inplace=True))
        else:
            self.main.add_module('BatchNorm+ReLU',
                nn.Sequential(
                    nn.BatchNorm2d(start_feat),
                    nn.ReLU(True)
                )
            )
        for i in range(self.n_layers - 1):
            # each layer doubles the image size
            current_size *= 2
            if i == 0:
                self.main.add_module('ConvTranspose_{}x{}'.format(current_size, current_size),
                                     self.add_convTranspose_block(start_feat, self.img_size[-1] * feat_mult // 2, 4,
                                                                  2, 1)
                                     )
            else:
                self.main.add_module('ConvTranspose_{}x{}'.format(current_size, current_size),
                                     self.add_convTranspose_block(self.img_size[-1] * feat_mult, self.img_size[-1] * feat_mult // 2, 4, 2, 1)
                                     )
            feat_mult = feat_mult // 2
        current_size *= 2
        self.main.add_module('ConvTranspose+Tanh_{}x{}'.format(current_size, current_size),
                             nn.Sequential(
                                 nn.ConvTranspose2d(self.img_size[-1], self.img_size[0], 4, 2, 1, bias=False),
                                 nn.Tanh()
                             )
                             )

    def forward(self, input, label=None):
        inputs = self.input_branch_4x4(input)
        if self.condition:
            labels = self.label_branch(label)
            embed = torch.cat([inputs, labels], 1)
            return self.main(embed)
        else:
            return self.main(inputs)

    def add_convTranspose_block(self, in_ch, out_ch, kernel, stride, padding):
        """
        convolutional transpose layer modeled after pytorch DCGAN tutorial
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        """
        if self.selu:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                      nn.SELU(inplace=True)]
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(True)]

        return nn.Sequential(*layers)

    def calc_layers(self, img_size, current_size=4):
        n_layers = 0
        while current_size < img_size:
            current_size *= 2
            n_layers += 1

        return n_layers, 2**(n_layers-1)


class simpleDCGANDiscriminator(nn.Module):
    def __init__(self, input_img_size, n_layers, condition=False, nl=10, selu=True, acgan=False):
        super(simpleDCGANDiscriminator, self).__init__()
        self.img_size = input_img_size
        self.condition = condition
        self.nl = nl
        self.selu = selu
        self.n_layers = n_layers
        self.acgan = acgan

        self.label_branch = nn.Sequential(
            # input is 1 x 1 x 1
            nn.ConvTranspose2d(1, nl, kernel_size=self.img_size[-1], stride=1, padding=0, bias=False),
        )

        # make the discriminator model
        self.main = nn.Sequential()
        current_size = self.img_size[-1]
        for i in range(self.n_layers):
            if i == 0:
                start_size = self.img_size[0] + self.nl if self.condition and not self.acgan else self.img_size[0]
                # each layer doubles the image size
                self.main.add_module('Conv2D_{}x{}'.format(start_size, current_size),
                                     self.add_conv_block(i, start_size, current_size, 4, 2, 1)
                                     )
            else:
                self.main.add_module('Conv2D_{}x{}'.format(current_size, int(current_size * 2)),
                                     self.add_conv_block(i, current_size, int(current_size * 2), 4, 2, 1)
                                     )
                current_size = int(current_size * 2)
        if self.acgan:
            # variation of acgan where teh original ACGAN uses linear classifier vs this one uses the final 2D conv layer
            self.adv_layer = nn.Sequential(
                nn.Conv2d(current_size, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
            self.aux_layer = nn.Sequential(
                nn.Conv2d(current_size, self.nl, 4, 1, 0, bias=False),
                nn.Softmax()
            )
        else:
            self.main.add_module('Conv2D+Sigmoid_{}x{}'.format(current_size, current_size),
                                 nn.Sequential(
                                     nn.Conv2d(current_size, 1, 4, 1, 0, bias=False),
                                     nn.Sigmoid()
                                 )
                                 )

    def forward(self, input, in_label=None):
        if self.condition:
            if self.acgan:
                out = self.main(input)
                validity = self.adv_layer(out)
                label = self.aux_layer(out)
                return validity, label
            else:
                labels = self.label_branch(in_label)
                embed = torch.cat([input, labels], 1)
                return self.main(embed), None
        else:
            return self.main(input), None

    def add_conv_block(self, i, in_ch, out_ch, kernel, stride, padding):
        """
        convolutional layer modeled after pytorch DCGAN tutorial
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        """
        if self.selu:
            layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                      nn.SELU(inplace=True)]
        else:
            if i == 0:
                layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                          nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                          nn.BatchNorm2d(out_ch),
                          nn.LeakyReLU(0.2, inplace=True)]

        return nn.Sequential(*layers)