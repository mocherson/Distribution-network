import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms, utils, models
import numpy as np 
from openset_util import gauss_logpdf


class MyAlexNet(nn.Module):
    def __init__(self,outnum=8):
        super(MyAlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(256, 1024, 3, padding=1),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(16))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyCNN(nn.Module):
    def __init__(self,inshape=28, outnum=10):
        super(MyCNN, self).__init__()
        self.inshape = inshape
        self.features = nn.Sequential(nn.Conv2d(3, 512, 3, padding=1),
                                      nn.ReLU(inplace=True), 
                                      nn.MaxPool2d(2),
                                      nn.Conv2d(512, 1024, 3, padding=1),
                                      nn.ReLU(inplace=True), 
                                      nn.MaxPool2d(2)
                                     )
        self.classifier = nn.Sequential(nn.Linear(1024*(inshape**2//16), 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, outnum)
                                       )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024*(self.inshape**2//16))
        x = self.classifier(x)
        return x
    
class MyVggNet16_bn_MNIST(nn.Module):
    def __init__(self, outnum=10):
        super(MyVggNet16_bn_MNIST, self).__init__()
        original_model = models.vgg16_bn(pretrained=False)
        #self.features = nn.Sequential(*[original_model.features[i] for i in range(23)])
        self.features = original_model.features[:23]
        self.features.add_module('gpool',nn.MaxPool2d(7))
        self.classifier = nn.Linear(256, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,256)
        x = self.classifier(x)
        return x
    
class MyVggNet16_bn_CF10(nn.Module):
    def __init__(self, outnum=10):
        super(MyVggNet16_bn_CF10, self).__init__()
        original_model = models.vgg16_bn(pretrained=False)
        #self.features = nn.Sequential(*[original_model.features[i] for i in range(23)])
        self.features = original_model.features
        self.classifier = nn.Linear(512, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,512)
        x = self.classifier(x)
        return x
    
class MyVggNet16_bn_XRAY(nn.Module):
    def __init__(self, outnum=10):
        super(MyVggNet16_bn_XRAY, self).__init__()
        original_model = models.vgg16_bn(pretrained=False)
        #self.features = nn.Sequential(*[original_model.features[i] for i in range(23)])
        self.features = original_model.features
        self.classifier = nn.Linear(512*2*2, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,512*2*2)
        x = self.classifier(x)
        return x
    
class textcnn(nn.Module):
    def __init__(self,  outnum=50, text_length=400, embedding=(20000,300)):
        super(textcnn, self).__init__()
        if isinstance(embedding, (tuple, list)):
            self.embedding=nn.Embedding(*embedding)
        elif isinstance(embedding, np.ndarray):
            self.embedding=nn.Embedding.from_pretrained(  \
                torch.FloatTensor(embedding),freeze=False)
        else:
            raise ValueError('invalid embedding')
            
        self.features=nn.Sequential()
        self.features.add_module(str(text_length), nn.Sequential(nn.Conv1d(self.embedding.weight.shape[1], 512, 5, padding=2),
                                                                 nn.BatchNorm1d(512),
                                                                 nn.ReLU(inplace=True), 
                                                                 nn.MaxPool1d(text_length, ceil_mode=True)))
#         text_length = math.ceil(text_length/2.0)

#         self.features.add_module(str(text_length), nn.Sequential(nn.Conv1d(512, 512, 3, padding=1),
#                                                              nn.BatchNorm1d(512),
#                                                              nn.ReLU(inplace=True), 
#                                                              nn.MaxPool1d(text_length, ceil_mode=True)))

            
        self.dens=nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, outnum))
        
    def forward(self, x):
        x=self.features(self.embedding(x).transpose(1,2))
        x = x.view(-1, self.dens[0].in_features)
        x = self.dens(x)
        return x
    
class LikelihoodLoss(nn.Module):
    def __init__(self, classweight=1,cls=range(10), ndim=10, sigmaform = 'identical'):
        super(LikelihoodLoss, self).__init__()
        nclass=len(cls)
        self.cls=cls
        if np.isscalar(classweight):
            self.classweight = np.ones(nclass) * classweight
        else:
            self.classweight = classweight
        self.nclass = nclass
        self.sigmaform = sigmaform
        self.ndim = ndim
        self.mu = Parameter(torch.randn(nclass, ndim))
        if sigmaform=='identical':
            self.sigma = Parameter(torch.rand(nclass)) # assume identical matrix
            self.sig = self.sigma
        elif sigmaform=='diagnal':
            self.sigma = Parameter(torch.rand(nclass, ndim))    # assume diagonal cov matrix
            self.sig = self.sigma
        elif sigmaform=='share':
            self.sigma = Parameter(torch.rand(1))  
            self.sig = self.sigma.expand(nclass)
        elif sigmaform=='sharediag':
            self.sigma = Parameter(torch.rand(1, ndim)) 
            self.sig = self.sigma.expand(nclass, ndim)
                                   
    def forward(self, input, target):  
        
        if self.sigmaform=='share':
            self.sig = self.sigma.expand(self.nclass)
        elif self.sigmaform=='sharediag':
            self.sig = self.sigma.expand(self.nclass, self.ndim)
        else:
            self.sig = self.sigma
        loss = 0
        
        for idx, cls in enumerate(self.cls):
            if (target==cls).any():
                loss = loss-torch.tensor(self.classweight[idx]).cuda() * \
                    gauss_logpdf(input[target==cls], self.mu[idx], self.sig[idx]).mean()

            #loss = loss - torch.tensor(self.classweight[i]).cuda() *  \
            #(gauss_logpdf(input[target==i], self.mu[i], sigma[i]).sum()-
             #0.1*gauss_logpdf(input[target!=i], self.mu[i], sigma[i]).sum())
        
        #print('loss', loss)
        return loss
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    