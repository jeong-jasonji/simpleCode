import os
import json
import torch
import argparse
from base import utilsProcessing

class BaseOptions():
    def __init__(self, json_filepath):
        self.initialized = False
        self.json_filepath = json_filepath

    def initialize(self):
        # load from the json file the test arguments
        self.opt = argparse.Namespace(**json.load(open(self.json_filepath, "r")))
        self.initialized = True

    def parse(self, save=True):
        # set up initialization
        if not self.initialized:
            self.initialize()
        #self.opt.isTrain = self.isTrain   # train or test

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
        
        # set up dataframes - opt.dataframes_dir should only have dataframe files in csv or pkl
        self.opt.dataframe_train = None
        self.opt.dataframe_val = None
        self.opt.dataframe_test = None
        self.opt.dataframe_ext = None
        for k in os.listdir(self.opt.dataframes_dir):
            if 'train' in k:
                self.opt.dataframe_train = os.path.join(self.opt.dataframes_dir, k)
            elif 'val' in k:
                self.opt.dataframe_val = os.path.join(self.opt.dataframes_dir, k)
            elif 'test' in k:
                self.opt.dataframe_test = os.path.join(self.opt.dataframes_dir, k)
            elif 'ext' in k:
                self.opt.dataframe_ext = os.path.join(self.opt.dataframes_dir, k)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # set up save dir and logs
        self.opt.save_dir = os.path.join(self.opt.checkpoints_dir, str(self.opt.test_name) + '/')
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

        return self.opt