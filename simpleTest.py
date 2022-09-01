# !/usr/bin/env python
# coding: utf-8

import os
import torch
import pickle
import pandas as pd
from torchvision import transforms
from options.experimentOptions import TestOptions
from base import simpleModels, simpleDataloader, simpleClassification, simpleTransforms

# python simpleTrain.py
opt = TestOptions().parse()

# initialize model
model_ft, opt.params_to_update, opt.input_size, opt.is_inception = simpleModels.initialize_model(opt)
# make data parallel if multi-gpu
if len(opt.gpu_ids) > 1:
    model_ft = torch.nn.DataParallel(model_ft, device_ids=opt.gpu_ids)
# get the previous model path and load it if it exists
model_ft = model_ft.cuda()
opt.prev_model = os.path.join(opt.save_dir, [i for i in os.listdir(opt.save_dir) if '.pth' in i][0])
model_ft = model_ft if opt.prev_model == None else torch.load(opt.prev_model)

# make preprocessing and transforms
transforms_eval = transforms.Compose([
    simpleTransforms.makeRGB(),
    transforms.Resize((opt.input_size, opt.input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# option to add the train and val predictions outputs as csvs
if opt.output_train_val:
    for test_mode in ['train', 'val']:
        dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_val if test_mode == 'val' else opt.dataframe_train, transforms_eval, shuffle=False)
        pred_dict = simpleClassification.model_predict(opt, model_ft, dataloader)
        mode_df = pd.DataFrame(pred_dict)
        mode_df.to_csv(opt.save_dir + '{}_{}.csv'.format(opt.model_name, test_mode))

for test_mode in ['test', 'ext']:
    if test_mode == 'test' and opt.dataframe_test is not None:
        dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_test, transforms_eval, shuffle=False)
    elif test_mode == 'ext' and opt.dataframe_ext is not None:
        dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_ext, transforms_eval, shuffle=False)
    else:
        print('{} does not exist'.format('test dataframe' if test_mode == 'test' else 'external dataframe'))
        break
    pred_dict = simpleClassification.model_predict(opt, model_ft, dataloader)
    mode_df = pd.DataFrame(pred_dict)
    mode_df.to_csv(opt.save_dir + '{}_{}.csv'.format(opt.model_name, test_mode))