"""


    Domain Adaptation for Mitotic Figure Assessment
    
    Marc Aubreville, FAU Erlangen-Nürnberg, GPL v3 license
    

    For results and further information, please read our paper:
    
    Learning New Tricks from Old Dogs - Inter-Species, Inter-Tissue Domain Adaptation for Mitotic Figure Assessment
    M. Aubreville, C. Bertram, S. Jabari, C. Marzahl, R. Klopfleisch, A. Maier
    submitted for:
    Bildverarbeitung für die Medizin, 2020


"""

from fastai import *
from fastai.vision import *


class DomainAdaptationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, output, target_class, target_domain):
        pred_class, pred_domain = output
        losses = CrossEntropyFlat()(pred_domain, target_domain).cuda()
        for p_cla, t_cla, t_dom in zip(pred_class,target_class, target_domain):
            if (t_dom==0): # only do in source domain
                losses += CrossEntropyFlat()(p_cla, t_cla).cuda() / np.sum((target_domain.cpu().numpy()==0))*20
        return losses

class DomainNonAdaptationLoss(nn.Module):
    """
       ignores domain in learning
    """
    def __init__(self):
        super().__init__()


    def forward(self, output, target_class, target_domain):
        pred_class, pred_domain = output
        losses = None
        for p_cla, t_cla, t_dom in zip(pred_class,target_class, target_domain):
            if (t_dom==0): # only do in source domain
                if (losses is None):
                    losses = CrossEntropyFlat()(p_cla, t_cla).cuda() / np.sum((target_domain.cpu().numpy()==0))*20

                else:
                    losses += CrossEntropyFlat()(p_cla, t_cla).cuda() / np.sum((target_domain.cpu().numpy()==0))*20
        return losses
