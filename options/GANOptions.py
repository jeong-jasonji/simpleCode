import os
import torch
import argparse
from base import utilsProcessing


class BaseGANOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        # experiment specifics
        self.parser.add_argument('--test_name', type=str, default='coloredMNIST_2_3_5_ACGAN', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/home/jjeong35/decisionGAN/', help='models are saved here')
        self.parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train for')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_classes', type=int, default=3, help='number of classes in the classification task')
        self.parser.add_argument('--dataframes_dir', type=str, default='./dataframes/coloredMNIST', help='directory where the dataframes are stored')
        self.parser.add_argument('--cls_select', type=str, default='2,3,5', help='selecting specific classes to use only: e.g. 0  0,3,7, 0,8')
        self.parser.add_argument('--save_every_epoch', type=bool, default=True, help='save the model for every epoch')

        # training hyperparameters
        self.parser.add_argument('--img_shape', type=tuple, default=(3, 64, 64), help='image shape: (n_channels, height, width)')
        self.parser.add_argument('--latent_dim', type=int, default=100, help='latent dimension for GAN to pull noise from')
        self.parser.add_argument('--g_lr', type=float, default=0.0002, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=0.00000001, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta_1', type=float, default=0.5, help='adam beta 1')
        self.parser.add_argument('--beta_2', type=float, default=0.999, help='adam beta 2')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
        self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        self.parser.add_argument('--weighted_sampling', type=bool, default=False, help='use weighted sampling')

        # model specifics
        self.parser.add_argument('--model_name', type=str, default='ACDCGAN', help='DCGAN, cDCGAN, ACDCGAN')
        self.parser.add_argument('--selu', type=bool, default=False, help='use SELU or batchNorm + ReLU')
        self.parser.add_argument('--prev_model', type=str, default=None, help='path to a previous model to load if specified')
        self.parser.add_argument('--rgb_merging', type=bool, default=False, help='merge three images to make an RGB image?')

        # for setting inputs
        self.parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
        self.parser.add_argument('--workers', type=int, default=4, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--verbose', type=bool, default=False, help='verbosity on or off')

        self.initialized = True

    def parse(self, save=True):
        # set up initialization
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        # set gpu ids and set cuda devices
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # set up params for GAN model
        self.opt.conditioned = False if self.opt.model_name == 'DCGAN' else True
        self.opt.ACGAN = True if self.opt.model_name == 'ACDCGAN' else False

        args = vars(self.opt)

        # set up dataframes - opt.dataframes_dir should only have dataframe files in csv or pkl
        self.opt.dataframe_GANtrain = None
        for k in os.listdir(self.opt.dataframes_dir):
            if 'GANtrain' in k:
                self.opt.dataframe_GANtrain = os.path.join(self.opt.dataframes_dir, k)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # set up save dir and logs
        self.opt.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.test_name + '/')
        utilsProcessing.mkdirs(self.opt.save_dir)
        # set up image and model dirs too
        utilsProcessing.mkdirs(os.path.join(self.opt.save_dir, 'images'))
        utilsProcessing.mkdirs(os.path.join(self.opt.save_dir, 'models'))
        if save and self.opt.prev_model is None:
            # save test options
            file_name_opt = os.path.join(self.opt.save_dir, 'opt.txt')
            with open(file_name_opt, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
            # make a log file
            log_file = os.path.join(self.opt.save_dir, 'log.txt')
            self.opt.log = open(log_file, 'a')
            print('------------ Log -------------\n', file=self.opt.log)

        return self.opt