import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.utils import save_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(generator_loss, discriminator_loss, save_dir):
    # make it so that it averages per over more ticks so it's smoother
    plt.xlabel("Training Epochs")
    plt.title("G/D Loss over Epochs")
    plt.ylabel('Loss')
    plt.plot(range(len(generator_loss)), generator_loss)
    plt.plot(range(len(discriminator_loss)), discriminator_loss)
    plt.legend(['Generator', 'Discriminator'])
    plt.savefig('{}/{}.png'.format(save_dir, 'loss_history'))
    # clear figure to generate new one for AUC
    plt.clf()


def train_GAN(opt, dataloader, models, criterions, optimizers):
    # ----------
    # GAN Training
    # ----------
    # setting up histories to plot
    G_losses = []
    D_losses = []
    prev_D_x = 999.0
    prev_D_G_z = 999.0
    prev_D_G_diff = 999.0
    prev_total_loss = 999.0

    # set up learning rate scheduler
    lambda_lr = lambda step: 0.90 ** step  # lambda function to reduce learning rate by 10%
    schedulers = {
        'Generator': lr_scheduler.LambdaLR(optimizers['Generator'], lr_lambda=lambda_lr),
        'Discriminator': lr_scheduler.LambdaLR(optimizers['Discriminator'], lr_lambda=lambda_lr)
    }

    real_label = 1
    fake_label = 0

    print('Starting GAN training...', file=opt.log)
    for epoch in tqdm(range(opt.max_epochs), desc='Epochs'):
        for i, (imgs, labels, img_id) in tqdm(enumerate(dataloader), desc='Batches', leave=False):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            ## Train Discriminator with all-real batch
            models['Discriminator'].zero_grad()
            # Format batch
            real_imgs = imgs.cuda()  # real images
            batch_size = real_imgs.shape[0]
            real_img_label = labels.cuda()  # image class label
            real_img_label = real_img_label.reshape(batch_size, 1, 1, 1).type(torch.float32)
            rf_label = torch.full((batch_size,), real_label, dtype=torch.float32).cuda()  # real or fake label
            # Forward pass real batch through D conditioned on type of GAN (only DCGAN doesn't gets labels)
            validity, pred_label = models['Discriminator'](real_imgs) if opt.model_name == 'DCGAN' else models['Discriminator'](real_imgs, real_img_label)
            print(validity)
            print(rf_label)
            print(pred_label)
            print(real_img_label)

            ## Calculate the D(real) losses
            errD_real_val = criterions['validity'](validity.squeeze(), rf_label)  # validity loss (seeing if its real or not)
            errD_real_aux = criterions['auxiliary'](pred_label.squeeze(), real_img_label) if pred_label is not None else None  # auxiliary loss (is the class correct or not - only DCGANs will have a None value)
            # Calculate gradients for D in backward pass
            errD_real = errD_real_val if errD_real_aux is None else 0.5 * (errD_real_val + errD_real_aux)
            errD_real.backward()
            D_x = validity.mean().item()  # D(real) loss

            ## Train Discriminator with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, opt.latent_dim, 1, 1).cuda()
            rand_class = torch.randint(opt.num_classes, (batch_size, 1, 1, 1), dtype=torch.float32).cuda()
            # Generate fake image batch with G
            fake = models['Generator'](noise, rand_class)
            rf_label.fill_(fake_label)
            # Classify all fake batch with D
            validity, pred_label = models['Discriminator'](fake.detach()) if opt.model_name == 'DCGAN' else models['Discriminator'](fake.detach(), rand_class)
            ## Calculate the D(fake) losses
            errD_fake_val = criterions['validity'](validity.squeeze(), rf_label)
            errD_fake_aux = criterions['auxiliary'](pred_label.squeeze(), rand_class) if pred_label is not None else None
            # Calculate gradients for D in backward pass
            errD_fake = errD_fake_val if errD_fake_aux is None else 0.5 * (errD_fake_val + errD_fake_aux)
            errD_fake.backward()
            D_G_z1 = validity.mean().item()  # D(fake) loss 1

            # Add the gradients from the all-real and all-fake batches and average
            errD = errD_real + errD_fake
            # Update D
            optimizers['Discriminator'].step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            models['Generator'].zero_grad()
            rf_label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            validity, pred_label = models['Discriminator'](fake.detach()) if opt.model_name == 'DCGAN' else models['Discriminator'](fake.detach(), rand_class)
            # Calculate G's loss based on this output - only do validity loss for generator
            errG_val = criterions['validity'](validity.squeeze(), rf_label)
            errG_aux = criterions['auxiliary'](pred_label.squeeze(), rand_class) if pred_label is not None else None
            errG = 0.5 * (errG_val + errG_aux) if pred_label is not None else errG_val
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = validity.mean().item()
            # Update G
            optimizers['Generator'].step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, opt.max_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # sample and save images every half epoch or sample interval, whatever is larger
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_noise = torch.randn(batch_size, opt.latent_dim, 1, 1).cuda()
                sample_class = torch.randint(opt.num_classes, (batch_size, 1, 1, 1), dtype=torch.float32).cuda()
                gen_imgs = models['Generator'](sample_noise, sample_class)
                save_image(gen_imgs.data, "{}/images/{}.png".format(opt.save_dir, batches_done), nrow=opt.num_classes, normalize=True)

        D_x = np.around(D_x, decimals=4)
        D_G_z = np.average([D_G_z1, D_G_z2])
        D_G_z = np.around(D_G_z, decimals=4)
        D_G_diff = abs(D_x - D_G_z)
        total_loss = sum((G_losses[-1], D_losses[-1]))

        ## check all losses and improvements and iteratively adjust them
        if epoch > opt.max_epochs * 0.1:
            if D_x < prev_D_x:  # if discriminator loss improves:
                prev_D_x = D_x  # update min loss and save state_dicts
                torch.save(models['Discriminator'].state_dict(), "{}/models/discriminator_best.pth".format(opt.save_dir))
            else:
                # load the previous_best state dicts
                models['Discriminator'].load_state_dict(torch.load("{}/models/discriminator_best.pth".format(opt.save_dir)))
                # and lower the learning rate based on the learning rate schedulers
                schedulers['Generator'].step()
            if D_G_z < prev_D_G_z and epoch > opt.max_epochs * 0.1:  # if generator loss improves:
                prev_D_G_z = D_G_z  # update min loss and save state_dicts
                torch.save(models['Generator'].state_dict(), "{}/models/generator_best.pth".format(opt.save_dir))
            else:  # reduce the discriminator learning rate
                # load the previous_best state dicts
                models['Generator'].load_state_dict(torch.load("{}/models/generator_best.pth".format(opt.save_dir)))
                # and lower the discriminator learning rate based on the learning rate schedulers
                schedulers['Discriminator'].step()

        # plot history
        plot_history(G_losses, D_losses, opt.save_dir)

        # save models every epoch
        if opt.save_every_epoch:
            torch.save(models['Generator'].state_dict(), "{}/models/generator_e{}.pth".format(opt.save_dir, epoch+1))
            torch.save(models['Discriminator'].state_dict(), "{}/models/discriminator_e{}.pth".format(opt.save_dir, epoch+1))
