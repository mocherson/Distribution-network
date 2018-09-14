import argparse

from readdata import * 
from mymodel import *
from openset_util import *

parser = argparse.ArgumentParser(description='Training the distribution network for Ohsumed')
parser.add_argument('--seed', default=1, type=int, metavar='N',  help='the random seed')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--path', default='/home/hddraid/shared_data/ohsumed-all/', type=str, help='data path')
parser.add_argument('--wordvecfile', default='glove.6B.300d.txt', type=str, help='wordvector files')
parser.add_argument('--savepath', default='/home/hddraid/opensetmodel/ohsumed/ohsumed_map_tr13/', type=str, help='path to save models')
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


train_label=(0,3,4,5,7,9,10,11,12,13,14,15,16)
val_label=(0,3,4,5,7,9,10,11,12,13,14,15,16,17,19,20)
test_label=(0,3,4,5,7,9,10,11,12,13,14,15,16,17,19,20)
train_set = ohsudata(path=args.path, use='train', label=train_label)
val_set = ohsudata(path=args.path, use='validation', label=val_label)
test_set = ohsudata(path=args.path, use='test', label=test_label)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
train_count = Counter(train_set.lbl)

word_vector_map=loadWord2Vec(args.wordvecfile)
sub_embedding = get_sub_embedding(word_vector_map,train_set.word2id)

ndim = args.dim
sform= args.sigmashape

if args.gpu>=0:
    torch.cuda.set_device(args.gpu)
model = textcnn(outnum=ndim, embedding=sub_embedding).cuda()
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
                            'fscore': mfs
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
        val_label=(0,3,4,5,7,9,10,11,12,13,14,15,16,17)
        cat = {k:cat[k] for k in val_label}
        s = sum([cat[x].sigma for x in set(val_label)-set(train_label)])/len(set(val_label)-set(train_label))
        t = sum([cat[x].thr for x in set(val_label)-set(train_label)])/len(set(val_label)-set(train_label))
        conf_matrix = test3(test_loader,model, cat, transf_sigma= s, transf_thr= t)
        conf_matrix.to_csv(savenm+'_%d.csv'%(i) )
        save_obj(cat, savenm+'_%d.cat'%(i) )
