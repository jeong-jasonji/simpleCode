import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

def create_loss_fx(opt):
    # be very careful about the loss functions used and the inputs and targets required for each.
    # ex: CrossEntropyLoss() requires input as [batch_size, #classes] and target as 1D numbers

    # get the manual weights if specified
    if opt.cls_weights != None:
        cls_weights = [float(i) for i in opt.cls_weights.split(',')]
        cls_weights.cuda()
    else:
        cls_weights = None

    if opt.loss_fx == 'sigmoid_focal':
        criterion = sigmoid_focal_loss
    elif opt.loss_fx == 'weighted_focal':
        criterion = WeightedFocalLoss() if cls_weights == None else nn.WeightedFocalLoss(weight=cls_weights)
    else:
        criterion = nn.CrossEntropyLoss() if cls_weights == None else nn.CrossEntropyLoss(weight=cls_weights)

    return criterion

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, weights=None, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.weights = weights
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.weights != None:
            CE_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        else:
            CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-CE_loss)
        F_loss = at*(1-pt)**self.gamma * CE_loss
        return F_loss.mean()