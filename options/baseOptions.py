import os
import torch
import argparse
from base import utilsProcessing

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        # experiment specifics
        self.parser.add_argument('--test_name', type=str, default='simpleTest', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--max_epochs', type=int, default=5, help='maximum number of epochs to train for')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_classes', type=int, default=3, help='number of classes in the classification task')
        self.parser.add_argument('--dataframes_dir', type=str, default='./dataframes', help='directory where the dataframes are stored')

        # training hyperparameters
        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
        self.parser.add_argument('--weight_decay', type=float, default=0.3, help='weight decay for the optimizer')
        self.parser.add_argument('--optim_metric', type=str, default='avg_loss', help='validation metric to optimize (avg_loss, avg_acc, avg_f1) - more to be added')
        self.parser.add_argument('--loss_fx', type=str, default='crossEntropy', help='loss function to use (crossEntropy, sigmoid_focal, weighted_focal, )')
        self.parser.add_argument('--cls_weights', type=str, default=None, help='manual class weights e.g None, 0.01,0.09,0.9')
        self.parser.add_argument('--weighted_sampling', type=bool, default=False, help='use weighted sampling')
        self.parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
        self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

        # model specifics
        self.parser.add_argument('--model_name', type=str, default='resnext101', help='model architecture to use')
        self.parser.add_argument('--model_freeze', type=float, default=0.0, help='percentage of the model weights to freeze')
        self.parser.add_argument('--prev_model', type=str, default=None, help='path to a previous model to load if specified')
        self.parser.add_argument('--save_every_epoch', type=bool, default=True, help='save models every epoch')
        self.parser.add_argument('--is_inception', type=bool, default=False, help='is the model a variant of InceptionNet?')
        self.parser.add_argument('--use_pretrained', type=bool, default=True, help='use ImageNet pretrained weights?')
        self.parser.add_argument('--fusion_model', type=bool, default=False, help='is this a fusion model with multiple inputs - currently not available until updated')

        # preprocessing arguments
        self.parser.add_argument('--cardiac_ablation', type=str, default=None, help='quadrant to remove (None, 1-8, cross_only, cross_remove)')
        self.parser.add_argument('--cardiac_remove_bottom_third', type=bool, default=False, help='keep the bottom third or not')

        # augmentations
        self.parser.add_argument('--aug_mode', type=str, default='transforms', help='augmentations to use (transforms or GAN) - currently not available until updated')

        # for setting inputs
        self.parser.add_argument('--workers', type=int, default=4, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--verbose', type=bool, default=False, help='verbosity on or off')

        self.initialized = True

    def parse(self, save=True):
        # set up initialization
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        # set gpu ids and set cuda devices
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # set up save dir and logs
        self.opt.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.test_name + '/')
        utilsProcessing.mkdirs(self.opt.save_dir)
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

        # set up dataframes - opt.dataframes_dir should only have dataframe files in csv or pkl
        for k in os.listdir(self.opt.dataframes_dir):
            self.opt.dataframe_train = os.path.join(self.opt.dataframes_dir, k) if 'train' in k else None
            self.opt.dataframe_val = os.path.join(self.opt.dataframes_dir, k) if 'val' in k else None
            self.opt.dataframe_test = os.path.join(self.opt.dataframes_dir, k) if 'test' in k else None
            self.opt.dataframe_ext = os.path.join(self.opt.dataframes_dir, k) if 'ext' in k else None

        return self.opt