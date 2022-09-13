import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from base import simpleGAN_models, simpleDataloader
from options.experimentOptions import GANTrainOptions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# code for simple linear GAN with or without condition
# python simpleGANs.py

def plot_history(generator_loss, discriminator_loss, save_dir):
    plt.xlabel("Training Epochs")
    plt.title("G/D Loss over Epochs")
    plt.ylabel('Loss')
    plt.plot(range(len(generator_loss)), generator_loss)
    plt.plot(range(len(discriminator_loss)), discriminator_loss)
    plt.legend(['Generator', 'Discriminator'])
    #plt.xticks(np.arange(1, epoch + 2, 1.0))
    plt.savefig('{}/{}.png'.format(save_dir, 'loss_history'))
    # clear figure to generate new one for AUC
    plt.clf()

# python simpleGANs.py
opt = GANTrainOptions().parse()

# Initialize generator and discriminator
generator = simpleGAN_models.simpleLinearGenerator(opt)
discriminator = simpleGAN_models.simpleLinearDiscriminator(opt, generator.n_layers - 1)
# GAN Loss function
adversarial_loss = torch.nn.MSELoss()

# move things to cuda
generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()

# Configure dataloader
transforms_train = transforms.Compose([
    transforms.Resize((opt.img_shape[-1], opt.img_shape[-1])),
    transforms.ToTensor()
    ])

dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_GANtrain, transforms_train)
print('training size: {}'.format(len(dataloader)), file=opt.log)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.beta_1, opt.beta_2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.beta_1, opt.beta_2))

# set tensors
FloatTensor = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else torch.LongTensor

# ----------
#  Training
# ----------
interval = np.max([opt.sample_interval, int(len(dataloader)/2)])
best_loss = 999.0
# setting up histories to plot
gen_hist = []
dis_hist = []
for epoch in tqdm(range(opt.max_epochs), desc='Epochs'):
    for i, (imgs, labels, img_id) in tqdm(enumerate(dataloader), desc='Batches', leave=False):

        batch_size = imgs.shape[0]

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
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.num_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # save histories
        gen_hist.append(g_loss.item())
        dis_hist.append(d_loss.item())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.max_epochs, i, len(dataloader), d_loss.item(), d_loss.item()),
            file=opt.log
        )

        batches_done = epoch * len(dataloader) + i
        # sample and save images every half epoch or sample interval, whatever is larger
        if batches_done % interval == 0:
            # Sample noise - get a n by n grid where n is the number of classes
            sample_noise = Variable(FloatTensor(np.random.normal(0, 1, (opt.num_classes ** 2, opt.latent_dim))))
            # Get labels ranging from 0 to n_classes for n rows
            sample_labels = np.array([num for _ in range(opt.num_classes) for num in range(opt.num_classes)])
            sample_labels = Variable(LongTensor(sample_labels))
            gen_imgs = generator(sample_noise, sample_labels)
            save_image(gen_imgs.data, "{}/images/{}.png".format(opt.save_dir, batches_done), nrow=opt.num_classes, normalize=True)

        # save model if total loss improves
        if d_loss.item() + g_loss.item() < best_loss:
            torch.save(generator.state_dict(), "{}/models/generator_best.pth".format(opt.save_dir))
            torch.save(discriminator.state_dict(), "{}/models/discriminator_best.pth".format(opt.save_dir))

    # plot history
    plot_history(gen_hist, dis_hist, opt.save_dir)

    # save models every epoch
    if opt.save_every_epoch:
        torch.save(generator.state_dict(), "{}/models/generator_e{}.pth".format(opt.save_dir, epoch+1))
        torch.save(discriminator.state_dict(), "{}/models/discriminator_e{}.pth".format(opt.save_dir, epoch+1))

