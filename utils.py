"""


    Domain Adaptation for Mitotic Figure Assessment
    
    Marc Aubreville, FAU Erlangen-Nürnberg, GPL v3 license
    

    For results and further information, please read our paper:
    
    Learning New Tricks from Old Dogs - Inter-Species, Inter-Tissue Domain Adaptation for Mitotic Figure Assessment
    M. Aubreville, C. Bertram, S. Jabari, C. Marzahl, R. Klopfleisch, A. Maier
    submitted for:
    Bildverarbeitung für die Medizin, 2020


"""

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from MulticoreTSNE import MulticoreTSNE as TSNE
from fastai import *
from fastai.vision import *

def _try_legend(*kwargs):
    try:    
        plt.legend(*kwargs)
    except:
        print('Legend unable to plot.')

def tsneplot_twolayers(learn, n_batches=20):

    dl = learn.dl(DatasetType.Valid)
    w = dl.num_workers
    dl.num_workers = 0
    total_target_class = []
    total_isSource = []
    total_isTarget = []
    total_target_domain = []
    features = []
    features_class_layer2 = []
    with torch.no_grad():
        for img_batch, target_batch in tqdm(iter(dl)):
            if (len(features)>n_batches):
                continue
            features1 = learn.model.encoder(img_batch)
            features.append(features1.view(img_batch.shape[0],-1))
            features_class_layer2.append(learn.model.class_head_1(features1).view(img_batch.shape[0],-1))
            target_class, target_domain = target_batch
            isSource = target_domain==0
            isTarget = target_domain==1
            total_isSource.append(isSource.cpu())
            total_isTarget.append(isTarget.cpu())
            total_target_class.append(target_class.cpu())
            total_target_domain.append(target_domain.cpu())

    features = torch.cat(features)
    total_target_domain = torch.cat(total_target_domain)
    total_target_class = torch.cat(total_target_class)
    total_isSource = torch.cat(total_isSource)
    total_isTarget = torch.cat(total_isTarget)
    features_class_layer2 = torch.cat(features_class_layer2)

    plt.figure(figsize=(8,8))
    tsne = TSNE(n_jobs=4,verbose=1000)
    Y = tsne.fit_transform(np.array(features))
    ax = plt.subplot(2,2,1)
    print(Y.shape, total_target_domain.numpy().shape)
    scatter =ax.scatter(Y[total_target_domain.numpy()==0,0],Y[total_target_domain.numpy()==0,1], color='red', label='source', alpha=0.3)
    scatter =ax.scatter(Y[total_target_domain.numpy()==1,0],Y[total_target_domain.numpy()==1,1], color='blue', label='target', alpha=0.3)
    _try_legend()
    plt.title('Domains')
    
    ax = plt.subplot(2,2,2)
    scatter =ax.scatter(Y[total_target_class.numpy()==1,0],Y[total_target_class.numpy()==1,1], color='purple', label='nonmitosis', alpha=0.3)
    scatter =ax.scatter(Y[total_target_class.numpy()==0,0],Y[total_target_class.numpy()==0,1], color='green', label='mitosis', alpha=0.3)
    plt.title('Classes')
    _try_legend()

    tsne = TSNE(n_jobs=4,verbose=1000)
    Y = tsne.fit_transform(np.array(features_class_layer2))
    ax = plt.subplot(2,2,3)
    print(Y.shape, total_target_domain.numpy().shape)
    scatter =ax.scatter(Y[total_target_domain.numpy()==0,0],Y[total_target_domain.numpy()==0,1], color='red', label='source', alpha=0.3)
    scatter =ax.scatter(Y[total_target_domain.numpy()==1,0],Y[total_target_domain.numpy()==1,1], color='blue', label='target', alpha=0.3)
    _try_legend()
    plt.title('Domains - 2nd layer')
    
    ax = plt.subplot(2,2,4)
    scatter =ax.scatter(Y[total_target_class.numpy()==1,0],Y[total_target_class.numpy()==1,1], color='purple', label='nonmitosis', alpha=0.3)
    scatter =ax.scatter(Y[total_target_class.numpy()==0,0],Y[total_target_class.numpy()==0,1], color='green', label='mitosis', alpha=0.3)
    plt.title('Classes - 2nd layer')
    _try_legend()


import matplotlib.gridspec as gridspec

def tsneplot_4classes(learn, n_batches=20, domains=[]):
    fig = plt.figure(figsize=(15,12))
    gs1 = gridspec.GridSpec(4, 6)
    ax = fig.add_subplot(gs1[0:4,0:4])
    total_target_class = []
    total_isSource = []
    total_isTarget = []
    total_target_domain = []
    features = []
    dl = learn.dl(DatasetType.Valid)
    w = dl.num_workers
    dl.num_workers = 0
    images = []

    with torch.no_grad():
        for img_batch, target_batch in tqdm(iter(dl)):
            if (len(features)>n_batches):
                continue
            feats = learn.model.encoder(img_batch).view(img_batch.shape[0],-1)
            features.append(feats)
            target_class, target_domain = target_batch
            isSource = target_domain==0
            isTarget = target_domain==1
            total_isSource.append(isSource.cpu())
            total_isTarget.append(isTarget.cpu())
            total_target_class.append(target_class.cpu())
            total_target_domain.append(target_domain.cpu())
            images.append(img_batch.cpu())

    features = torch.cat(features,0)
    total_target_domain = torch.cat(total_target_domain)
    total_target_class = torch.cat(total_target_class)
    total_isSource = torch.cat(total_isSource)
    total_isTarget = torch.cat(total_isTarget)
    images = torch.cat(images)

    tsne = TSNE(n_jobs=4,verbose=1000)
    Y = tsne.fit_transform(np.array(features))
#    ax = fig.add_subplot(gs[0:4, 0:4])
    print(Y.shape, total_target_domain.numpy().shape)
    scatter =ax.scatter(Y[total_target_domain.numpy()==0,0],Y[total_target_domain.numpy()==0,1], color='red', label=domains[0], alpha=0.3)
    scatter =ax.scatter(Y[total_target_domain.numpy()==1,0],Y[total_target_domain.numpy()==1,1], color='blue', label=domains[1], alpha=0.3)
    scatter =ax.scatter(Y[total_target_domain.numpy()==2,0],Y[total_target_domain.numpy()==2,1], color='green', label=domains[2], alpha=0.3)
    scatter =ax.scatter(Y[total_target_domain.numpy()==3,0],Y[total_target_domain.numpy()==3,1], color='cyan', label=domains[3], alpha=0.3)
    selidx = 0
    leftX = np.min(Y[:,0])
    rightX = np.max(Y[:,0])
    centerX = 0.5*(leftX+rightX)
    
    bins = 10
    intervals_y = np.linspace(np.min(Y[:,1]),np.max(Y[:,1]),bins)
    
    colors = ['r','b','g','c']
    for idx,ypos in enumerate(intervals_y[:-1]):

        try:
            img_sel=np.where((Y[:,1]>ypos) & (Y[:,1] < intervals_y[idx+1]) & (Y[:,0]<centerX))[0]
            img_sel=np.random.choice(img_sel)
        except:
            img_sel=np.where((Y[:,1]>ypos) & (intervals_y[idx-1] < intervals_y[idx+1]) & (Y[:,0]<centerX))[0]
            img_sel=np.random.choice(img_sel)
        xsize=(intervals_y[1]-intervals_y[0])*1.8
        ysize=xsize
        xpos=leftX-2*xsize-(idx%2)*xsize
        single_img = Image(learn.data.denorm(images[img_sel]))
        plt.imshow(single_img.data.permute(1,2,0).numpy(), extent=[xpos,xpos+xsize,ypos,ypos+ysize])

        plt.plot([xpos+xsize,Y[img_sel,0]],[ypos+0.5*ysize,Y[img_sel,1]],linestyle='--',color=colors[total_target_domain.numpy()[img_sel]])

        try:
            img_sel=np.where((Y[:,1]>ypos) & (Y[:,1] < intervals_y[idx+1]) & (Y[:,0]>centerX))[0]
            img_sel=np.random.choice(img_sel)
        except:
            img_sel=np.where((Y[:,1]>intervals_y[idx-1]) & (Y[:,1] < intervals_y[idx+1]) & (Y[:,0]>centerX))[0]
            img_sel=np.random.choice(img_sel)
            
        xpos=rightX+xsize+(idx%2)*xsize
        single_img = Image(learn.data.denorm(images[img_sel]))
        plt.imshow(single_img.data.permute(1,2,0).numpy(), extent=[xpos,xpos+xsize,ypos,ypos+ysize])
        plt.plot([xpos,Y[img_sel,0]],[ypos+0.5*ysize,Y[img_sel,1]],linestyle='--',color=colors[total_target_domain.numpy()[img_sel]])
        
    currlim=plt.ylim()
    newlim = (currlim[0]*1.3, currlim[1])
    plt.ylim(newlim)
        
    plt.legend(loc='lower center')
    plt.title('Domains')

    ax2 = fig.add_subplot(gs1[1:3,4:6])
    scatter =ax2.scatter(Y[total_target_class.numpy()==1,0],Y[total_target_class.numpy()==1,1], color='purple', label='nonmitosis', alpha=0.3)
    scatter =ax2.scatter(Y[total_target_class.numpy()==0,0],Y[total_target_class.numpy()==0,1], color='green', label='mitosis', alpha=0.3)
    plt.title('Classes')
    _try_legend()
    return ax, ax2, Y, images

def evaluate_multiclasslearner(learn, stage=''):
    total_pred_class = []
    total_target_class = []
    total_isSource = []
    total_isTarget = []
    dl = learn.dl(DatasetType.Valid)
    w = dl.num_workers
    dl.num_workers = 0
    learn.model.eval()
    with torch.no_grad():
        for img_batch, target_batch in tqdm(iter(dl)):
            target_class, target_domain = target_batch
            prediction_batch = learn.model(img_batch)
            pred_class, pred_domain = prediction_batch[:2]
            pred_class = pred_class.cpu()
            pred_domain = pred_domain.cpu()
            target_class = target_class.cpu()
            target_domain = target_domain.cpu()
            isSource = target_domain==0
            isTarget = target_domain==1
            total_isSource.append(isSource)
            total_isTarget.append(isTarget)
            total_target_class.append(target_class)
            total_pred_class.append(pred_class)

    total_pred_class = torch.cat(total_pred_class)
    total_target_class = torch.cat(total_target_class)
    total_isSource = torch.cat(total_isSource)
    total_isTarget = torch.cat(total_isTarget)

    accuracy_source = accuracy(total_pred_class[total_isSource], total_target_class[total_isSource].cpu())
    try:
        accuracy_target = accuracy(total_pred_class[total_isTarget], total_target_class[total_isTarget].cpu())
    except:
        accuracy_target = 0

    print(stage,'Source: ',accuracy(total_pred_class[total_isSource], total_target_class[total_isSource].cpu()))
    print(stage,'Target: ',accuracy_target)

    y_true = torch.cat([total_target_class[total_isSource].view(-1,1)==0,total_target_class[total_isSource].view(-1,1)==1],1)
    
    # Note: Mitosis is actually class==0, so we have to invert here
    gt = total_target_class[total_isSource].numpy()
    detected = np.argmax(total_pred_class[total_isSource].numpy(),1)
    TN = np.sum((detected==1) & (gt==1))
    FN = np.sum((detected==1) & (gt==0))
    FP = np.sum((detected==0) & (gt==1))
    TP = np.sum((detected==0) & (gt==0))
    F1_source = 2 * TP / (2 * TP + FP + FN)
    prec = TP / (TP+FP)
    rec = TP / (TP+FN)
    
    print('SOURCE F1:',F1_source, 'TP:',TP,'FP:',FP,'FN:',FN, 'TN:',TN, 'F1 alt:',2*prec*rec/(prec+rec), 'Acc:',(TP+TN)/(TP+FN+FP+TN),'N=',detected.shape[0])
    print('SOURCE Prec:',prec,'Rec:',rec)
    

    y_true = torch.cat([total_target_class[total_isTarget].view(-1,1)==0,total_target_class[total_isTarget].view(-1,1)==1],1)

    
    try:
        if np.sum(total_isTarget.cpu().numpy())>0:
            gt = total_target_class[total_isTarget].numpy()
            detected = np.argmax(total_pred_class[total_isTarget].numpy(),1)

            TN = np.sum((detected==1) & (gt==1))
            FN = np.sum((detected==1) & (gt==0))
            FP = np.sum((detected==0) & (gt==1))
            TP = np.sum((detected==0) & (gt==0))
            F1_target = 2 * TP / (2 * TP + FP + FN)
            prec = TP / (TP+FP)
            rec = TP / (TP+FN)

            print('TARGET F1:',F1_target, 'TP:',TP,'FP:',FP,'FN:',FN, 'TN:',TN, 'F1 alt:',2*prec*rec/(prec+rec), 'Acc:',(TP+TN)/(TP+FN+FP+TN),'N=',detected.shape[0])
            print('TARGET Prec:',prec,'Rec:',rec)
        else:
            F1_target = 0
            
        
    except:
        F1_target = 0

    learn.model.train()
        
    return {'Acc_source': accuracy_source, 
            'Acc_target': accuracy_target, 
            'F1_source': F1_source, 
            'F1_target':F1_target}

