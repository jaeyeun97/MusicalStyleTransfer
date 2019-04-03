import torch
import torch.nn as nn


class ClassificationLoss(nn.Module): 
    def __init__(self, A_label=0.0, B_label=1.0): 
        super(ClassificationLoss, self).__init__()
        self.register_buffer('A_label', torch.tensor(A_label))
        self.register_buffer('B_label', torch.tensor(B_label))
        self.loss = nn.MSELoss()
       
    def get_target_tensor(self, prediction, label):
        if label == 0:
            target_tensor = self.A_label
        else:
            target_tensor = self.B_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
