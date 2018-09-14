from os.path import join 
import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from text_utils import *

## transform gray to RGB to fit the input shape
transform_mnist =  transforms.Compose([transforms.Lambda(lambda x: np.repeat(x[:,:,np.newaxis],3,axis=2)),
                                 transforms.ToTensor(),
                                ])

class MinstDataset(Dataset):    
    def __init__(self, path = '/home/hddraid/shared_data/MINIST/', use = 'train',label=range(10), transform=transform_mnist):
    
        if use in ['train', 'validation']:
            with open(join(path, 'train-labels-idx1-ubyte'), 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)

            with open(join(path, 'train-images-idx3-ubyte'), 'rb') as fimg:
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=0)
            tr_idx ,val_idx =next(sss.split(lbl, lbl))
            if use is 'train':
                img, lbl = img[tr_idx], lbl[tr_idx]
            else:
                img, lbl = img[val_idx], lbl[val_idx]
        elif use is 'test':
            with open(join(path, 't10k-labels-idx1-ubyte'), 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)

            with open(join(path, 't10k-images-idx3-ubyte'), 'rb') as fimg:
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
        else:
            raise ValueError("dt must be 'test', 'train' or 'validation'.")
        
        mask = np.isin(lbl, label)
        # self.label_set, self.lbl = np.unique(lbl[mask], return_inverse=True)
        self.lbl = lbl[mask].astype(np.long)
        self.img = img[mask]
        self.transform=transform
            
    def __len__(self):
        if len(self.lbl)==self.img.shape[0]:
            return len(self.lbl)
        else:
            raise ValueError( "label and image must have the same number.")
            
    def __getitem__(self,idx):
        return({'data': self.transform(self.img[idx]) if self.transform else self.img[idx], 
                'label': self.lbl[idx],
                'index': idx
               })
    

class MyCIFAR10(datasets.CIFAR10):
    def __init__(self, root='/home/hddraid/shared_data/CIFAR10/', use='train', label=range(10), transform = transforms.ToTensor()):
        super(MyCIFAR10, self).__init__(root=root, train=True if use!='test' else False, transform=transform)
        if use!='test':
            sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=0)
            tr_idx ,val_idx =next(sss.split(self.train_labels, self.train_labels))
        if use == 'train':
            data, labels = self.train_data[tr_idx], np.array(self.train_labels)[tr_idx]  
        elif use == 'test' :
            data, labels = self.test_data, np.array(self.test_labels)
        elif use == 'validation':
            data, labels = self.train_data[val_idx], np.array(self.train_labels)[val_idx]
        else:
            raise ValueError('use must be "train" or "validation" or "test" ')
        mask = np.isin(labels, label)
        # self.label_set, self.lbl = np.unique(np.array(labels)[mask], return_inverse=True)
        self.lbl = labels[mask]
        self.img = data[mask]
        
    def __len__(self):
        if len(self.lbl)==self.img.shape[0]:
            return len(self.lbl)
        else:
            raise ValueError( "label and image must have the same number.")
            
    def __getitem__(self,idx):
        return({'data': self.transform(self.img[idx]) if self.transform else self.img[idx], 
                'label': self.lbl[idx],
                'index': idx
               })

# ohsu_vocab = load_obj('/home/hddraid/shared_data/ohsumed-all/vocab_clean.pkl')
class ohsudata(Dataset):
    def __init__(self, path='/home/hddraid/shared_data/ohsumed-all/',
                 use ='all', label=range(23), clean_fn=clean_str, text_length=400):
        self.path=path
        metafile=join('single_label.csv')
        meta = pd.read_csv(metafile,index_col=0,dtype={'fname':str})   
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        tr_val, test_idx =next(sss.split(meta.loc[:,'class_id'], meta.loc[:,'class_id']))
        tr, val =next(sss.split(meta.iloc[tr_val,2], meta.iloc[tr_val,2]))     
        tr_idx, val_idx=tr_val[tr], tr_val[val]
        if use=='train':
            meta=meta.iloc[tr_idx]
        elif use=='validation':
            meta=meta.iloc[val_idx]
        elif use=='test':
            meta=meta.iloc[test_idx]
        elif use=='all':
            pass
        else:
            raise ValueError("dt must be 'test', 'train' or 'validation'.")
            
        self.meta = meta.loc[meta['class_id'].isin(label)]
        self.lbl = self.meta['class_id']
        self.unique_label, self.label_id = np.unique(self.lbl,return_inverse=True)
        self.clean_fn=clean_fn
        self.text_length=text_length
        self.word2id=load_obj('vocab_clean.pkl')      
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self,idx):
        fname = join(self.path, self.meta.iloc[idx,1],self.meta.iloc[idx,0])
        with open(fname,'r') as f:
            s = f.read()
        if self.clean_fn:
            word_list = self.clean_fn(s)
        else:
            word_list = nltk.word_tokenize(s)
         

        id_list=[self.word2id[x] for x in word_list if x in self.word2id]
        if len(id_list)<self.text_length:
            id_list.extend([self.word2id['<PAD>']]*(self.text_length-len(id_list)))
        else:
            id_list=id_list[:self.text_length]
        
        sample={'data': np.array(id_list), 'label':self.lbl.iloc[idx],
                'lblid':self.label_id[idx], 'index':self.meta.index[idx]}
        return sample
    

