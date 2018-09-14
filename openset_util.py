import torch
from torch.autograd import Variable
import shutil
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal, MultivariateNormal
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, confusion_matrix
import pandas as pd
import pickle as pk
from scipy.stats import multivariate_normal

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class category(object):
    def __init__(self, mu=0, sigma=1, thr=0, n_sample=1, label=None, intrain=False, sample_idx=None):
        self.mu = mu.clone()
        self.sigma = sigma.clone()
        self.thr = thr
        self.label = label
        self.n_sample = n_sample
        self.intrain = intrain
        self.dim = 1 if np.isscalar(mu) else len(mu)
        self.sample_idx = [] if sample_idx is None else sample_idx
        self.vsg2 = (self.n_sample+3)*sigma**2
        self.fscore=0
        
    def __repr__(self):
        return 'mu:'+str(self.mu)+'\n'+'sigma:'+str(self.sigma)+'\n'+'thr:'+str(self.thr)+'\n'+  \
               'n_sample:'+str(self.n_sample)+'\n'+'label:'+str(self.label)+'\n'+'isintrain:'+str(self.intrain)+'\n'+ \
               'fscore:'+str(self.fscore)+'\n'
        
    def isnovel(self, x):       
        # likelihood = multivariate_normal.logpdf(x, self.mu, self.sigma**2)
        likelihood = gauss_logpdf(x, torch.tensor(self.mu).cuda(), torch.tensor(self.sigma).cuda())
        return likelihood.cpu().numpy() < self.thr, likelihood
    
    def update(self, dt):   
        if len(dt.shape) < 2:
            dt = dt.unsqueeze(0)
            
        self.vsg2 = self.vsg2+len(dt)*dt.var(dim=0,unbiased=False)+  \
                    len(dt)*self.n_sample*(self.mu-dt.mean(dim=0))**2/(self.n_sample+len(dt))
        self.mu = (self.n_sample*self.mu + dt.sum(dim=0))/(self.n_sample+len(dt))
        self.n_sample = self.n_sample + len(dt)
        self.sigma = (self.vsg2/(self.n_sample+3)).sqrt()
        
    def thrcompute(self, output, target):
        prob = gauss_logpdf(output, self.mu, self.sigma)
        self.thr, self.fscore = find_threshold2(prob, target==self.label)
        
        

def train(train_loader, model, criterion, optimizer,  epoch,  iter_size=10, iflabelid=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time        
        data_time.update(time.time() - end)
        inputs = data['data'].cuda()
        targets = data['lblid'].cuda() if iflabelid else data['label'].cuda()
        output = model(inputs)
        loss = criterion(output, targets)
        # print(targets)
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i+1) % iter_size == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    return losses.avg

def validate(val_loader, model, criterion, iflabelid=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pred, tgt = [], []
        for i, data in enumerate(val_loader):
            input = data['data'].cuda()
            target = data['lblid'].cuda() if iflabelid else data['label'].cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            pred.append(output.argmax(dim=1))
            tgt.append(target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        pred = torch.cat(pred).cpu().numpy()
        tgt = torch.cat(tgt).cpu().numpy()

    return top1.avg, losses.avg, confusion_matrix(tgt,pred)
        
        
def validate1(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        unknown_out = []
        unknown_target = []
        for i, data in enumerate(val_loader):
            inputs, targets = data['image'].cuda(), data['label'].cuda()
            output = model(inputs)
            loss = criterion(output, targets)  
            losses.update(loss.item(),inputs.size(0))
            acc_bt = accuracy(output, targets, topk=(1,))
            acc.update(acc_bt[0], inputs.size(0))
            
#            known = torch.tensor(np.isin (targets.cpu().numpy(),range(criterion.nclass)).astype(np.uint8))
#            acc_bt = accuracy_likelihood(output[known], targets[known], criterion.mu, criterion.sigma, topk=(1,))
#            acc.update(acc_bt[0], inputs.size(0))
            
#            unknown_out.append(output[~known])
#            unknown_target.append(targets[~known])

            # measure elapsed time
#            batch_time.update(time.time() - end)
#            end = time.time()

            if (i+1) % 10 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Known classes Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Known classes acc@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                      'Known classes sigma {sigma}'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,acc=acc, sigma=str(criterion.sigma.data)))
#        un_out = torch.cat(unknown_out).cpu().numpy()
#        un_tar = torch.cat(unknown_target).cpu().numpy()
#        unknown_cls = np.unique(un_tar)
#        unknown_loss_cls = np.array([-likelihood(un_out[un_tar==c]) for c in unknown_cls])
#        unknown_sigma = np.array([un_out[un_tar==c].std(axis=0).mean() for c in unknown_cls])
#        #unknown_loss = -sum(unknown_loss_cls)
#        print('Unkonwn loss: {}\t Unknown sigma: {}'.format(str(unknown_loss_cls), str(unknown_sigma)))
            
    return acc.avg, losses.avg, #unknown_loss_cls, unknown_sigma 

def likelihood(x, ifmean=True):
    print(x)
    mu = x.mean(axis=0)
    sigma = x.var(axis=0)
    if ifmean:
        return multivariate_normal.logpdf(x, mean=mu, cov=np.diag(sigma)).mean(axis=0)
    else :
        return multivariate_normal.logpdf(x, mean=mu, cov=np.diag(sigma))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #with torch.no_grad():
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).data.cpu().numpy()[0])
        return res
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,filename+'.best')
        
def showdata(feature,label):      
    pca=PCA(2)
    fea=pca.fit_transform(feature)
    for i in np.unique(label):
        plt.scatter(fea[label==i][:,0], fea[label==i][:,1], label=str(i))
    plt.legend()
    plt.show()    
    
def gauss_logpdf(x, mu, sigma):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    if len(sigma.shape) < 2:
        return Normal(mu,sigma.abs()).log_prob(x).sum(dim=1)
    elif len(sigma.shape) == 2:
        return MultivariateNormal(mu,scale_tril=sigma).log_prob(x).sum(dim=1)

    
def accuracy_likelihood(output, target, mu, sigma, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        prob = torch.zeros(output.shape[0],mu.shape[0]).cuda()
        
        for i in range(mu.shape[0]):
            prob[:,i] = gauss_logpdf(output, mu[i], sigma if sigma.shape[0]==1 else sigma[i])
            
        _, pred = prob.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).data.cpu().numpy()[0])
        return res
    
# maximum likelihood estimation for unknown classes
def validate2(val_loader, model, criterion):
    print('computer sigma and threshold for validation classes')
    innerp = defaultdict(dict)
    interp = defaultdict(dict)
    thr = {}
    fscore = {}
    sgm = {}
    mu = {}
    model.eval()
    with torch.no_grad():
        out, tar, idx = [], [], []
        for data in val_loader:
            inputs, target, index = data['data'].cuda(), data['label'].cuda(), data['index']
            output = model(inputs)
            out.append(output)
            tar.append(target)
            idx.append(index)
        output = torch.cat(out)
        target = torch.cat(tar)
        index = torch.cat(idx)

        cls = target.cpu().unique( sorted=True).tolist()
        for c in cls:
            print('class:',c)
            if c in criterion.cls:
                i=criterion.cls.index(c)
                sigma = criterion.sig if criterion.sig.shape[0]==1 else criterion.sig[i]
                sgm[c] = sigma.cpu().numpy()
                mu[c] = criterion.mu[i].cpu().numpy()
                inner = gauss_logpdf(output[target==c], criterion.mu[i], sigma)
                innerp[c].update( dict(zip(index[target==c], inner)))

                inter = gauss_logpdf(output[target!=c], criterion.mu[i], sigma)
                interp[c].update( dict(zip(index[target!=c], inter)))
            else:
                m = output[target==c].mean(dim=0)
                s = output[target==c].std(dim=0)
                mu[c] = m.cpu().numpy()
                sgm[c]= s.cpu().numpy()
                inner = gauss_logpdf(output[target==c], m, s).cpu().numpy()
                innerp[c].update( dict(zip(index[target==c], inner)))

                inter = gauss_logpdf(output[target!=c], m, s).cpu().numpy()
                interp[c].update( dict(zip(index[target!=c], inter)))                
                    
        for c in cls:
            thr[c], fscore[c] = find_threshold(innerp[c].values(), interp[c].values())            
                 
    return thr, fscore, sgm, mu, innerp, interp

# MAP estimation for known and unknown classes
def validate3(val_loader, model, criterion, train_count, ifMAP=False):     
    print('compute MAP estimate for all classes')    
    cat={}
    sgm = {}
    mu = {}
    
    model.eval()
    with torch.no_grad():
        out, tar, idx = [], [], []
        for data in val_loader:
            inputs, target, index = data['data'].cuda(), data['label'].cuda(), data['index']
            output = model(inputs)
            out.append(output)
            tar.append(target)
            idx.append(index)
        output = torch.cat(out)
        target = torch.cat(tar)
        index = torch.cat(idx)

        # class_size = train_set.meta.groupby('class_id').size()
        cls = target.cpu().unique( sorted=True).tolist()
        for c in cls:
            if c in criterion.cls:
                i=criterion.cls.index(c)
                sigma = criterion.sig if criterion.sig.shape[0]==1 else criterion.sig[i]
                sgm[c] = sigma.cpu().numpy()
                mu[c] = criterion.mu[i].cpu().numpy()
                cat[c] = category(mu = criterion.mu[i], sigma = sigma, n_sample = train_count[c], label = c, intrain = True )
                if ifMAP:
                    cat[c].update(output[target==c])
                cat[c].thrcompute(output,target)
            else:
                m = output[target==c].mean(dim=0)
                s = output[target==c].std(dim=0)
                mu[c] = m.cpu().numpy()
                sgm[c]= s.cpu().numpy()
                cat[c] = category(mu = m, sigma = s, n_sample = (target==c).sum().item(), label = c, intrain = False )
                cat[c].thrcompute(output,target)          
                 
    return cat, sgm, mu

def find_threshold(inner, inter ):
    precision, recall, thr = precision_recall_curve([1]*len(inner)+[0]*len(inter),list(inner)+list(inter),pos_label=1)
    fscore = 2.0 * (precision * recall) / (precision + recall)
    return thr[np.nanargmax(fscore)], np.nanmax(fscore)

def find_threshold2(prob, label ):
    precision, recall, thr = precision_recall_curve(label, prob, pos_label=1)
    fscore = 2.0 * (precision * recall) / (precision + recall)
    return thr[np.nanargmax(fscore)], np.nanmax(fscore)

def isnovel(sample, cate):
    return gauss_logpdf(sample, cate.mu, cate.sigma) < cate.threshold

def classify(input, cates, transf_sigma, transf_thr):
    novelty = [c.isnovel(input) for c in cates]  
    known = [(i,n[1]) for i, n in enumerate(novelty) if not n[0][0]]
    # novel class
    if len(known)==0:   
        return 1, category(input, transf_sigma, transf_thr, label=str(len(cates))+'_novel', intrain=False)
    else:  #known class
        idx, _ = max(known , key=lambda x: x[1])
        return 0, cates[idx]
    
def classify3(input, cates, transf_sigma, transf_thr, ifopen=True):
    novelty = {c:cates[c].isnovel(input) for c in cates}  
    known = {c:novelty[c][1] for c in novelty if not novelty[c][0]} if ifopen else {c:novelty[c][1] for c in novelty} 
    
    # novel class
    if len(known)==0:   
        return 1, category(input, transf_sigma, transf_thr, label=str(len(cates))+'_novel', intrain=False)
    else:  #known class
        c = max(known , key=lambda x: known[x])
        # if not cates[c].intrain:
        #     cates[c].update(input)
        return 0, cates[c]
        

def test(test_loader, model, cates, transf_sigma, transf_thr):
    model.eval()  
    con_mat = pd.DataFrame(0,index=['isnovel']+[c.label for c in cates], columns=[c.label for c in cates])
    with torch.no_grad():
        count=0
        for data in test_loader:
            input, target, idx = data['data'].cuda(), data['label'], data['index']
            output = model(input)
            for i, out in enumerate(output):
                if not target[i].item() in con_mat.index:
                    con_mat.loc[target[i].item(), :] = 0
                isnovel, c = classify(out, cates, transf_sigma, transf_thr)
                c.update(idx[i])
                count =count + 1
                print('%d -> %s,  conut:%d'%(target[i].item(),c.label, count))                
                if isnovel:
                    con_mat.loc[:, c.label] = 0
                    con_mat.loc['isnovel', c.label]=1
                    con_mat.loc[target[i].item(), c.label]=1
                    cates.append(c)
                else:
                    if not target[i].item() in con_mat.index:
                        con_mat.loc[target[i].item(), :] = 0
                    con_mat.loc[target[i].item(), c.label]=1+con_mat.loc[target[i].item(), c.label]
                #print(con_mat.iloc[1:,:].sum().sum())

    return con_mat

def test3(test_loader, model, cates, transf_sigma, transf_thr):
    model.eval()  
    con_mat = pd.DataFrame(0,index=['isnovel']+[c for c in cates], columns=[c for c in cates])
    with torch.no_grad():
        count=0
        for data in test_loader:
            input, target, idx = data['data'].cuda(), data['label'], data['index']
            output = model(input)
            for i, out in enumerate(output):
                if not target[i].item() in con_mat.index:
                    con_mat.loc[target[i].item(), :] = 0
                isnovel, c = classify3(out, cates, transf_sigma, transf_thr, True)
                count =count + 1
                print('%d -> %s,  conut:%d'%(target[i].item(),c.label, count))                
                if isnovel:
                    con_mat.loc[:, c.label] = 0
                    con_mat.loc['isnovel', c.label]=1
                    con_mat.loc[target[i].item(), c.label]=1
                    cates[c.label]=c
                else:
                    if not target[i].item() in con_mat.index:
                        con_mat.loc[target[i].item(), :] = 0
                    con_mat.loc[target[i].item(), c.label]=1+con_mat.loc[target[i].item(), c.label]
                #print(con_mat.iloc[1:,:].sum().sum())

    return con_mat
    
def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)    
    
def computef1(filename):
    cm = pd.read_csv(filename, index_col=0)
    if 'ohsumed' in filename:
        cm.loc['isnovel','17']=1
    else:
        cm.loc['isnovel','7']=1
    label_known = cm.columns[cm.loc['isnovel']==0]
    label_gen = cm.columns[cm.loc['isnovel']!=0]
    label_unknown = cm.index.difference(label_known).drop('isnovel')
    
    f1_known = {}
    for lb in label_known:    
        f1_known[lb] = 2*cm.loc[lb,lb]/(cm.loc[lb].sum()+cm[lb].sum())
        
    tp_known = sum([cm.loc[i,i] for i in label_known])
    fptp_known = cm[label_known].sum().sum()
    fntp_known = cm.loc[label_known].sum().sum()
    f1_known['micro'] = tp_known*2/(fptp_known+fntp_known)
    f1_known['macro'] = sum([f1_known[lb] for lb in label_known])/len(label_known)
    
    f1_unknown={}
    for lb in label_unknown:
        aslb = cm.loc[lb,label_gen].argmax()
        f1_unknown[lb] = 2*cm.loc[lb,aslb]/(cm.loc[lb].sum()+cm[aslb].sum())
    f1_unknown['asone'] = 2*cm.loc[label_unknown,label_gen].values.sum() / \
                          (cm.loc[label_unknown].values.sum()+cm.loc[cm.index!='isnovel',label_gen].values.sum())
    return f1_known, f1_unknown

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    