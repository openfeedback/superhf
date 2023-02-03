import torch 
import torch.nn as nn

class PreferenceCrossEntropyLoss(nn.Module):
    def __init__(self, weight = None, reduction = 'mean'):
        super().__init__()
        self.weight = weight        # could be useful for quality feedback
        self.reduction = reduction 
        
    def forward(self, winner, loser):
        """
        winner: size (batchsize,) tensor of reward scores of winning responses
        loser: size (batchsize,) tensor of reward scores of losing responses
        """
        
        weighted_probabilities = nn.Sigmoid(winner - loser)
        if self.weight is not None:
            weighted_probabilities *= self.weight
            
        loss = torch.log(weighted_probabilities) 
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
        
class PreferenceHingeLoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction 
    
    def forward(self, winner, loser):
        """
        winner: size (batchsize,) tensor of reward scores of winning responses
        loser: size (batchsize,) tensor of reward scores of losing responses
        """
        loss = 1 - (winner - loser)
        loss[loss < 0] = 0
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
        
