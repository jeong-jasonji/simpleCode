# !/usr/bin/env python
# coding: utf-8

import torch
import pickle
import pandas as pd
from torchvision import transforms
from options.experimentOptions import TrainOptions
from base import simpleModels, simpleDataloader, simpleClassification, simpleTransforms

# general data paths: /home/jason/decisionGAN/data/

# python simpleTrain.py
opt = TrainOptions().parse()

# initialize model
model_ft, opt.params_to_update, opt.input_size, opt.is_inception = simpleModels.initialize_model(opt)

# make preprocessing and transforms
transforms_train = transforms.Compose([
    # add transforms from: simpleTransforms
    simpleTransforms.makeRGB(),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((opt.input_size, opt.input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transforms_val = transforms.Compose([
    simpleTransforms.makeRGB(),
    transforms.Resize((opt.input_size, opt.input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# make dataloaders
train_loader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_train, transforms_train)
print('training size: {}'.format(len(train_loader)), file=opt.log)
val_loader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_val, transforms_val, shuffle=False)
print('validation size: {}'.format(len(val_loader)), file=opt.log)
dataloaders = {'train': train_loader, 'val': val_loader}

# make data parallel if multi-gpu
if len(opt.gpu_ids) > 1:
    model_ft = torch.nn.DataParallel(model_ft, device_ids=opt.gpu_ids)

# load the model, optimizer, and loss functions
model_ft, optimizer_ft, criterion = simpleModels.load_model(opt, model_ft)

# train and evaluate
model_ft, histories = simpleClassification.train_epochs(opt, model_ft, dataloaders, criterion, optimizer_ft)

# save the final versions
torch.save(model_ft, opt.save_dir + '{}.pth'.format(opt.model_name))
pickle.dump(histories, open(opt.save_dir + "{}_history.pkl".format(opt.model_name), "wb"))

# option to add the train and val predictions outputs as csvs
if opt.output_csv:
    # default transform
    transforms_eval = transforms.Compose([
        simpleTransforms.makeRGB(),
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # load the model, optimizer, and loss functions
    model_ft, optimizer_ft, criterion = simpleModels.load_model(opt, model_ft)

    for test_mode in ['train', 'val']:
        dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_val if test_mode == 'val' else opt.dataframe_train, transforms_eval, shuffle=False)
        pred_dict = simpleClassification.model_predict(opt, model_ft, dataloader)
        mode_df = pd.DataFrame(pred_dict)
        mode_df.to_csv(opt.save_dir + '{}_{}.csv'.format(opt.model_name, test_mode))