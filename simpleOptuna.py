import torch
import pickle
import pandas as pd
from torchvision import transforms
from options.baseOptions import BaseOptions
from base import simpleModels, simpleDataloader, simpleClassification, simpleTransforms
# import optuna for hyperparameter search
import optuna
import json

# python simpleOptuna.py

# 1. Define an objective function to be maximized.
def objective(trial):
    # load and modify the test options
    base_json = json.load(open('base_options.json', "r"))
    # 2. Suggest values of the hyperparameters using a trial object.
    base_json['test_name'] = trial.suggest_int("test_name", low=0, high=100, step=1)
    base_json['dataframes_dir'] = './dataframes/heartCalcification/'
    base_json['model_name'] = trial.suggest_categorical('model_name', ['resnet50', 'resnet101', 'resnet152', 'resnext101', 'se-resnext101', 'densenet121'])
    base_json['learning_rate'] = trial.suggest_float("learning_rate", low=0.0000002, high=0.00002, step=0.0000002)
    base_json['weight_decay'] = trial.suggest_float("weight_decay", low=0.00, high=0.5, step=0.05)
    base_json['batch_size'] = trial.suggest_int("batch_size", low=16, high=64, step=8)
    base_json['img_size'] = trial.suggest_int("img_size", low=256, high=512, step=256)
    base_json['checkpoints_dir'] = '/media/Datacenter_storage/ChestXray_cardiology/JasonData/tests/Optuna_finetune/'
    # save the test json file
    json.dump(base_json, open('test.json', 'w'))
    # initialize options for tests
    opt = BaseOptions(json_filepath='test.json').parse()
    # specifically for heart calcification:
    opt.df_labels = {'file': 'img_path', 'Calcium_Group_four': 'img_label', 'AccessionNumber': 'img_id'}

    # initialize model
    model_ft, opt.params_to_update, opt.input_size, opt.is_inception = simpleModels.initialize_model(opt)

    # make preprocessing and transforms
    transforms_train = transforms.Compose([
        # add transforms from: simpleTransforms
        simpleTransforms.makeRGB(),
        # simpleTransforms.SegmentCardiacEcho(simpleCrop=(0.1, 0.05)),
        # transforms.Pad(50),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((opt.input_size, opt.input_size)),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # simpleTransforms.AddGaussianNoise(mean=0., std=0.1)
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
    model_ft, histories, best_states = simpleClassification.train_epochs(opt, model_ft, dataloaders, criterion, optimizer_ft)

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
            dataloader = simpleDataloader.simpleDataLoader(opt,
                                                           opt.dataframe_val if test_mode == 'val' else opt.dataframe_train,
                                                           transforms_eval, shuffle=False)
            pred_dict = simpleClassification.model_predict(opt, model_ft, dataloader)
            mode_df = pd.DataFrame(pred_dict)
            mode_df.to_csv(opt.save_dir + '{}_{}.csv'.format(opt.model_name, test_mode))

    # return a metric for optuna to optimize
    return best_states['avg_loss']

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize', storage='sqlite:///heartCalficiation_optuna.db')
study.optimize(objective, n_trials=100)

# save the best params
best_params = study.best_params
print(best_params)

pickle.dump(study, open('study.pkl', 'w'))