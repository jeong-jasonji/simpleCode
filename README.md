# simpleCode
Simplified the code that I use in a daily basis.

# base
## simpleClassification.py
Defines functions for training a model for defined epochs with early stopping criteria. Also has a function for running inference with a model.
## simpleDataloader.py
Defines functions for a custom dataloader that reads a .csv or pickled dataframe.
## simpleLosses.py
Defines functions for loss functions to be used in the model training.
## simpleModels.py
Defines functions for loading models to be used with options for using pretrained weights or not.
## simpleOptimizer.py
Defines functions for loading optimizers to be used (mostly variants of Adam for now).
## simpleTransforms.py
Defines custom transforms to be used in model training.
## utilsProcessing.py
Defines functions that are useful for processing.

# options
## baseOptions.py
Defines the basic arguments for the experiment like experiment specifics, training hyperparameters, model specifics, etc. and generates the experiment folder and logs.
## experimentOptions.py
Defines arguments specific for training or testing, specifically for output generation.

# eval
## simpleEvaluation.py
Functions for calculating metrics for evaluating the model performance e.g. boostrap confidence intervals, AUCROC curves, etc.

# dataframes
Directory where the dataframes (.csv or pickled dataframes) for training, validation, testing, and external sets are. Dataframes require the following columns: 
**img_path**: the path to the image file (.png, .jpg, and other images able to be loaded by PILLOW)
**img_id**: the identifier for the patient/exam level separation
**img_label**: integer class labels e.g. 0, 1, 2, etc.
