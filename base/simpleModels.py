"""
Defined functions for general classification model building
Essentially the functions from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet
from pretrainedmodels import se_resnext101_32x4d, inceptionresnetv2
from efficientnet_pytorch import EfficientNet

from .simpleLosses import create_loss_fx
from .simpleOptimizer import create_optimizer

def load_model(opt, model_ft):
    # input
    model_ft = model_ft.cuda()
    model_ft = model_ft if opt.prev_model == None else torch.load(opt.prev_model)

    ### NEED SEPERATE SCRIPTS FOR OPTIMIZERS AND LOSS FUNCTIONS ###

    # Observe that all parameters are being optimized
    optimizer_ft = create_optimizer(opt)

    # Setup the loss fxn
    criterion = create_loss_fx(opt)

    return model_ft, optimizer_ft, criterion

def set_parameter_requires_grad(model_ft, model_freeze):
    """
    function to set the parameters to be trained based on the model_freeze
    model_freeze (float):
        - '1.0': freeze the whole model, only extract features
        - '0.0': do not freeze the model at all, train the full model parameters
        - '0.X': freeze the first fraction (0.X) of the model parameters
    """
    params_to_update = []
    if model_freeze == 1.0:
        for param in model_ft.parameters():
            param.requires_grad = False
    else:
        params_to_update = []
        num_layers = len([name for name, param in model_ft.named_parameters()])
        frozen_layers = round(num_layers * model_freeze)
        if frozen_layers != 0 and frozen_layers % 2 != 0:  # rounds up to even number for weight and bias combinations
            frozen_layers += 1
        layer_count = 0
        for name, param in model_ft.named_parameters():
            if layer_count < frozen_layers:
                param.requires_grad = False
            else:
                params_to_update.append(param)
            layer_count += 1
    return params_to_update

def initialize_model(opt):
    """
    Requires: opt.model_name, opt.use_pretrained, opt.model_freeze, opt.num_classes

    To add another model into this function:
    1) install the model; 2) create a model_name; 3) copy the format of the other models and create the
    model_ft, set_parameter, etc. 3.5) the num_ftrs might need to load different names (i.e. model_ft.fc or
    model_ft._fc - different models will have different final layers but will generally be 'fc' for pytorch pretrained
    models; '_fc' for efficientNet and 'last_layer' for pretrainedmodels)
    4) add the final layer as just a linear layer*
    4.5)* final layer needs to be a linear layer because pytorch loss functions include softmax and sigmoid activations
    within them
    5) make sure to read the originial architecture and set the input size

    Currently the models I have added are:
    EfficientNet-B7, Resnet18, Resnet50, Resnet101, Resnet152, Alexnet, VGG11_bn, VGG19_bn, Squeezenet1.1, Densenet121,
    Densenet169, Densenet161, MobileNetV2, and Inception v3

    see link for more models and details on each: https://pytorch.org/docs/stable/torchvision/models.html
    """
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = None
    is_inception = False

    if opt.model_name == "resnext101":
        """resnext101_32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "se-resnext101":
        """ SE-ResNeXt101
        """
        model_ft = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet') if opt.use_pretrained else se_resnext101_32x4d(num_classes=1000, pretrained=None)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224
    
    elif opt.model_name == "inceptionresnetv2":
        """ Inception ResNet v2
        """
        model_ft = inceptionresnetv2(num_classes=1000, pretrained='imagenet') if opt.use_pretrained else inceptionresnetv2(num_classes=1000, pretrained=None)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 299
        is_inception = True

    elif 'efficientnet' in opt.model_name:
        """ EfficientNet-BX (model name should be: 'efficientnet-bX', where X=0-7)
        """
        model_ft = EfficientNet.from_pretrained(opt.model_name) if opt.use_pretrained else EfficientNet.from_name(opt.model_name)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif 'resnet' in opt.model_name:
        if opt.model_name == 'resnet9':
            """ Modified Resnet18 with only one block per layer so that it's 'half'
            - this is a good way to make custom models from pytorch classes
            """
            model_ft = resnet._resnet('resnet18_half', resnet.BasicBlock, [1, 1, 1, 1], False, False)
        elif opt.model_name == "resnet18":
            """ Resnet18 """
            model_ft = models.resnet18(pretrained=opt.use_pretrained)
        elif opt.model_name == "resnet50":
            """ Resnet50 """
            model_ft = models.resnet50(pretrained=opt.use_pretrained)
        elif opt.model_name == "resnet101":
            """ Resnet101 """
            model_ft = models.resnet101(pretrained=opt.use_pretrained)
        elif opt.model_name == "resnet152":
            """ Resnet152 """
            model_ft = models.resnet152(pretrained=opt.use_pretrained)
        else:
            print('{} is not a supported ResNet model'.format(opt.model_name))
            return model_ft, input_size

        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "alexnet":
        """ Alexnet """
        model_ft = models.alexnet(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif 'vgg' in opt.model_name:
        if opt.model_name == "vgg11bn":
            """ VGG11_bn """
            model_ft = models.vgg11_bn(pretrained=opt.use_pretrained)
        elif opt.model_name == "vgg19bn":
            """ VGG19_bn """
            model_ft = models.vgg19_bn(pretrained=opt.use_pretrained)
        else:
            print('{} is not a supported VGG model'.format(opt.model_name))
            return model_ft, input_size
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "squeezenet1.1":
        """ Squeezenet1.1 """
        model_ft = models.squeezenet1_1(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        model_ft.classifier[1] = nn.Conv2d(512, opt.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = opt.num_classes
        input_size = 224

    elif 'densenet' in opt.model_name:
        if opt.model_name == "densenet121":
            """ Densenet121"""
            model_ft = models.densenet121(pretrained=opt.use_pretrained)
        elif opt.model_name == "densenet161":
            """ Densenet161"""
            model_ft = models.densenet161(pretrained=opt.use_pretrained)
        elif opt.model_name == "densenet169":
            """ Densenet169"""
            model_ft = models.densenet169(pretrained=opt.use_pretrained)
        else:
            print('{} is not a supported DenseNet model'.format(opt.model_name))
            return model_ft, input_size
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)	
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "mobilenet":
        """ MobileNetV2
        """
        model_ft = models.mobilenet_v2(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 299
        is_inception = True

    else:
        print("Invalid model name")

    return model_ft, params_to_update, input_size, is_inception