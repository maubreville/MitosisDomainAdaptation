from fastai import *
from fastai.vision import *

from fastai.vision.learner import num_features_model,create_body
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



def create_classification_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                concat_pool:bool=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    retval = []
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
        if (actn is None):
            retval += [nn.Sequential(*layers)]
            layers = [] 
        else:
            retval += [nn.Sequential(*layers[0:-1])]
            layers = [layers[-1]]
            

    return retval 

def create_domain_head(nf: int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                       concat_pool:bool=True):
    
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    retval = []
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    layers += bn_drop_lin(lin_ftrs[0], lin_ftrs[1], True, ps[0], actns[0])
    retval = [nn.Sequential(*layers[0:-1])]
    layers = [layers[-1]]
    
    layers += bn_drop_lin(2*lin_ftrs[1], lin_ftrs[2], True, ps[1], actns[1])
    retval += [nn.Sequential(*layers)]
            

    return retval    


class MitosisClassifier_DomainAdaptationInterconnected(nn.Module):
    def __init__(self):
        super().__init__()
        concat_pool = True
        self.encoder = create_body(models.resnet18, True)
        nf = num_features_model(nn.Sequential(*(self.encoder).children())) * (2 if concat_pool else 1)
        self.domain_head_1,self.domain_head_2 = create_domain_head(nf,2,lin_ftrs=[256])
        self.class_head_1, self.class_head_2 = create_classification_head(nf,2,lin_ftrs=[256])
        self.alpha = 0
    
    
    def forward(self, input):
        features1 = self.encoder(input)
        class_features = self.class_head_1(features1)
        classes = self.class_head_2(class_features)

        
        reverse_features1 = ReverseLayerF.apply(features1, self.alpha)
        domain_features_1 = self.domain_head_1(reverse_features1)
        reverse_class_head = ReverseLayerF.apply(class_features, self.alpha)
        domain_features = torch.cat([domain_features_1, reverse_class_head], 1)
        domain = self.domain_head_2(domain_features)
        
        
        return classes, domain

