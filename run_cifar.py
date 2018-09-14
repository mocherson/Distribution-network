import argparse

from readdata import * 
from mymodel import *
from openset_util import *

parser = argparse.ArgumentParser(description='Training the distribution network for CIFAR10')
parser.add_argument('--seed', default=1, type=int, metavar='N',  help='the random seed')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--path', default='/home/hddraid/shared_data/CIFAR10/', type=str, help='data path')
parser.add_argument('--savepath', default='/home/hddraid/opensetmodel/CIFAR10/cifar10_map_tr7/', type=str, help='path to save models')
parser.add_argument('-d', '--dim', default=10, type=int, metavar='N', help='the dimention of latent space (default:10)')
parser.add_argument('-s', '--sigmashape', default='share', type=str, help='the form of covariance matrix')
parser.add_argument('-e','--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', type=int, default=-1, metavar='N', help='the GPU number (default auto schedule)')
parser.add_argument('--action', default='train-test', type=str, help='train or test')

args = parser.parse_args()

seed = args.seed 
batch_size=args.batch_size   
torch.manual_seed(seed)

train_label=range(7)
val_label=range(8)
test_label=range(10)
train_set = MyCIFAR10(root=args.path, use='train', label=train_label)
val_set = MyCIFAR10(root=args.path, use='validation', label=val_label)
test_set = MyCIFAR10(root=args.path, use='test', label=test_label)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
train_count = Counter(train_set.lbl)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ndim = args.dim
sform= args.sigmashape

if args.gpu>=0:
    torch.cuda.set_device(args.gpu)
model = MyVggNet16_bn_CF10(outnum=ndim).cuda()
criterion = LikelihoodLoss(cls=train_label, ndim=ndim, sigmaform = sform).cuda()
optimizer = optim.Adam([{'params' : model.parameters(), 'lr':0.001},
                        {'params' : criterion.parameters(), 'lr' : 0.001}], lr=0.001)

savenm = join(args.savepath,'model_sform%s_dim%d'%(sform, ndim))

if 'train' in args.action:
    res = []
    best = 0
    for epoch in range(args.start_epoch, args.epochs):                  
        loss_tr = train(train_loader, model, criterion, optimizer, epoch, 10)
        print(criterion.mu, criterion.sigma)
        cat, sgm, mu = validate3(val_loader, model, criterion, train_count,ifMAP=True)   
        mfs = np.array([cat[c].fscore for c in train_label]).mean()+ \
              np.array([cat[c].fscore for c in val_label if c not in train_label]).mean()
        save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'criterion': criterion.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'loss_tr': loss_tr,
                            'category': cat,
                            'fscore:': mfs
                        }, mfs>best, savenm+'.pth.tar')
        if mfs>best:
            best=mfs

        res.append((cat,loss_tr, sgm, mu))
        print("category:"+str(cat)+'\n'+'mean fscore:'+str(mfs))                   
        save_obj(res, savenm+'.pkl')

if 'test' in args.action :
    for i in range(10):
        checkpoint = torch.load(savenm+'.pth.tar.best')
        model.load_state_dict(checkpoint['state_dict'])
        criterion.load_state_dict(checkpoint['criterion'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cat = checkpoint['category']
        val_label=range(8)
        cat = {k:cat[k] for k in val_label}
        s = sum([cat[x].sigma for x in set(val_label)-set(train_label)])/len(set(val_label)-set(train_label))
        t = sum([cat[x].thr for x in set(val_label)-set(train_label)])/len(set(val_label)-set(train_label))
        conf_matrix = test3(test_loader,model, cat, transf_sigma= s, transf_thr= t)
        conf_matrix.to_csv(savenm+'_%d.csv'%(i) )
        save_obj(cat, savenm+'_%d.cat'%(i) )
