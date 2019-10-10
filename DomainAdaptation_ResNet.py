"""


    Domain Adaptation for Mitotic Figure Assessment
    
    Marc Aubreville, FAU Erlangen-Nürnberg, GPL v3 license
    

    For results and further information, please read our paper:
    
    Learning New Tricks from Old Dogs - Inter-Species, Inter-Tissue Domain Adaptation for Mitotic Figure Assessment
    M. Aubreville, C. Bertram, S. Jabari, C. Marzahl, R. Klopfleisch, A. Maier
    submitted for:
    Bildverarbeitung für die Medizin, 2020


"""

import matplotlib
matplotlib.use('Agg')
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import pickle
from models import *
from losses import *
from utils import tsneplot_twolayers, evaluate_multiclasslearner
from data_loader import *

class Logger(object):
    def __init__(self,dataset_source,dataset_target):
        self.terminal = sys.stdout
        self.log = open("DomainAdaptation_%s_%s.log" % (dataset_source, dataset_target), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

listOfDatasets = ['CCMCT','CMC','MITOS2014','HUMMEN']

results=dict()
parser = argparse.ArgumentParser()
parser.add_argument('source', help=('Source dataset (%s)' % ','.join(listOfDatasets)), type=str)
parser.add_argument('target', help=('Target dataset (%s)' % ','.join(listOfDatasets)), type=str)
parser.add_argument('run')

args = parser.parse_args()

dataset_source = args.source
dataset_target = args.target


# Write stdout also to a file
sys.stdout = Logger(dataset_source, dataset_target)

run = sys.argv[3]


classes_str = ['Nonmitosis','Mitosis']
filenames = []
domains = []
classes = []
isvalid = []

# Add Source data set
f = open(Path('dataset') / dataset_source / (dataset_source+'.txt'),'r')
for line in f.readlines():
    fname, cls = line.split(' ')
    fname = 'dataset' +os.sep+ dataset_source +os.sep+  fname
    domains.append('SOURCE')
    filenames.append(fname)
    classes.append(classes_str[int(cls[0])])
    isvalid.append(np.random.random(1)<0.2)

# Complete target dataset is part of training AND validation set. Note that we are not using the validation set
# anywhere for model selection or model adaptation. In training, only images from the source domain (=0) are used 
# for the cell type classification loss.
f = open(Path('dataset') / dataset_target / (dataset_target+'.txt'),'r')
for line in f.readlines():
    fname, cls = line.split(' ')
    fname = 'dataset' +os.sep+ dataset_target +os.sep+ fname
    domains.append('TARGET')
    filenames.append(fname)
    classes.append(classes_str[int(cls[0])])
    isvalid.append(1)

    domains.append('TARGET')
    filenames.append(fname)
    classes.append(classes_str[int(cls[0])])
    isvalid.append(0)
    
# Create dataframe and shuffle it
df = pd.DataFrame(list(zip(filenames, domains, classes, isvalid)), columns=['filename', 'domain', 'cellType','is_valid'])
df = df.sample(frac=1) # shuffle

# Now find minimum class to have symmetric dataset
df3 = ((df['domain']=='TARGET')&(df['cellType']=='Mitosis'))
targetMitosis = (df3[df3].index)
df3 = ((df['domain']=='TARGET')&(df['cellType']=='Nonmitosis'))
targetNonmitosis = (df3[df3].index)
df3 = ((df['domain']=='SOURCE')&(df['cellType']=='Mitosis'))
sourceMitosis = (df3[df3].index)
df3 = ((df['domain']=='SOURCE')&(df['cellType']=='Nonmitosis'))
sourceNonmitosis = (df3[df3].index)
minLen = min([len(targetMitosis),len(sourceNonmitosis),len(targetNonmitosis),len(sourceMitosis) ])

# Limit to 1600 images (800 per class in training and validation set)
if (minLen>1600):
    minLen=1600

# Generate new dataframe with equal distribution
allIndices = targetMitosis[0:minLen].copy()
allIndices=allIndices.append(sourceMitosis[0:minLen])
allIndices=allIndices.append(targetNonmitosis[0:minLen])
allIndices=allIndices.append(sourceNonmitosis[0:minLen])
df = df.loc()[allIndices]

df = df.sample(frac=1) # shuffle
df.to_excel('DA_%s_%s_%s.xlsx' % (run, dataset_source, dataset_target))

np.random.seed(42)
np.random.random()

print('Number of mitotic figures and non-mitotic figures per domain is: ', minLen)

# Some debugging
print('Dataset is: ',df.head(10))

# Add classes

pdata = ''

celltype_labels = (
    NanLabelImageList.from_df(df, path=pdata, cols='filename')
    .split_from_df(col='is_valid')
    .label_from_df(cols='cellType')
)
domain_labels = (
    NanLabelImageList.from_df(df, path=pdata, cols='filename')
    .split_from_df(col='is_valid')
    .label_from_df(cols='domain')
)

multitask_project = {
    'celltype': {
        'label_lists': celltype_labels,
        'metric': accuracy
    },
    'domain': {
        'label_lists': domain_labels,
        'metric': accuracy
    },
}
ItemLists.label_from_mt_project = label_from_mt_project
image_lists = ImageList.from_df(df, path=pdata, cols='filename').split_from_df(col='is_valid')
mt_label_lists = image_lists.label_from_mt_project(multitask_project)

tfms = get_transforms()
data = mt_label_lists.transform(tfms, size=128).databunch(bs=48).normalize(imagenet_stats)

metrics = mt_metrics_generator(multitask_project, data.mt_lengths, data=data)
metrics += [accuracy_source]
metrics += [accuracy_target]

epochs = 30

# Model with adaptation loss
model = MitosisClassifier_DomainAdaptationInterconnected()
crit = DomainAdaptationLoss()    
learn = Learner(data, model, loss_func=crit, callback_fns=[ShowGraph], #BBMetrics, ShowGraph
                metrics=metrics
               )

# Alpha is the factor in the gradient reversal layer
learn.model.alpha = 1.0

# Initial fit with only heads active
lr = 1e-3
learn.fit(1,lr)

# Then unfreeze model completely and train for 30
learn.unfreeze()
factor=1.0
learn.fit_one_cycle(epochs, slice(lr), callbacks=[ShowGraph(learn)])

# T-SNE plot after first training
tsneplot_twolayers(learn)
plt.savefig('figures/%s_%s_%s_01_initial_training_DA_tSNE.pdf' % (dataset_source, dataset_target, run))

print('Evaluation of 1st step domain adaptation learner ..')
results['acc_da_1'] = evaluate_multiclasslearner(learn, stage='DA_1st')

# increase alpha, repeat
learn.model.alpha=10.0
learn.fit_one_cycle(epochs, slice(lr), callbacks=[ShowGraph(learn)])


# T-SNE plot after second training
tsneplot_twolayers(learn)
plt.savefig('figures/%s_%s_%s_02_2nd_training_DA_tSNE.pdf' % (dataset_source, dataset_target, run))

print('Evaluation of 2nd step domain adaptation learner ..')
results['acc_da_2'] = evaluate_multiclasslearner(learn, stage='DA_2nd')

learn.model.alpha=10.0

learn.fit_one_cycle(epochs, slice(lr), callbacks=[ShowGraph(learn)])

# T-SNE plot after third training
tsneplot_twolayers(learn)
plt.savefig('figures/%s_%s_%s_03_3rd_training_DA_tSNE.pdf' % (dataset_source, dataset_target,run))

learn.save('%s_%s_%s_DA_DomainAdaptation_3rd' % (run,dataset_source, dataset_target))

print('Evaluation of 3rd step domain adaptation learner ..')
results['acc_da_3'] = evaluate_multiclasslearner(learn, stage='DA_3rd')


# Recreating model and changing loss --> Now without domain adaptation

model = MitosisClassifier_DomainAdaptationInterconnected()
crit = DomainNonAdaptationLoss()
learn = Learner(data, model, loss_func=crit, callback_fns=[ShowGraph], #BBMetrics, ShowGraph
                metrics=metrics
               )

lr = 1e-3
learn.fit(1,lr)
learn.unfreeze()

learn.fit_one_cycle(epochs, slice(lr), callbacks=[ShowGraph(learn)])

tsneplot_twolayers(learn)
plt.savefig('figures/%s_%s_%s_04_initial_training_NoDA_tSNE.pdf' % (dataset_source, dataset_target, run))

print('Evaluation of 1st step domain adaptation learner ..')
results['acc_noda_1'] = evaluate_multiclasslearner(learn, stage='NoDA_1st')

# increase alpha, repeat
learn.model.alpha=0.0
learn.fit_one_cycle(epochs, slice(lr), callbacks=[ShowGraph(learn)])

tsneplot_twolayers(learn)
plt.savefig('figures/%s_%s_%s_05_2nd_training_NoDA_tSNE.pdf' % (dataset_source, dataset_target, run))

print('Evaluation of 2nd step domain adaptation learner ..')
results['acc_noda_2'] =  evaluate_multiclasslearner(learn, stage='NoDA_2nd')

learn.model.alpha=0.0

learn.fit_one_cycle(epochs, slice(lr), callbacks=[ShowGraph(learn)])

tsneplot_twolayers(learn)
plt.savefig('figures/%s_%s_%s_06_3rd_training_NoDA_tSNE.pdf' % (dataset_source, dataset_target,run))
learn.save('%s_%s_%s_DA_DomainNonAdaptation_3rd' % (run,dataset_source, dataset_target))

print('Evaluation of 3rd step domain non adaptation learner ..')

results['acc_noda_3']=evaluate_multiclasslearner(learn, stage='NoDA_3rd')

print('Evaluation of 3rd step domain non adaptation learner ..')

pickle.dump(results, open(f'results_{run}_{dataset_source}_{dataset_target}.p','wb'))

