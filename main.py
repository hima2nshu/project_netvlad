
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from torch.optim import Adam
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
from vggresnet import *
import h5py
import faiss
from tensorboardX import SummaryWriter
import numpy as np
from torchvision import transforms
convert_tensor = transforms.ToTensor()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import netvlad
# import vggresnet

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--batchSize', type=int, default=4, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=3, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
#parser.add_argument('--dataPath', type=str, default='/nfs/ibrahimi/data/', help='Path for centroid data.')
parser.add_argument('--dataPath', type=str, default='/home/aesicd_42/CHANDRAJIT/Anuradha/pytorch-NetVlad-master/cluster/', help='Path for centroid data.')

#parser.add_argument('--runsPath', type=str, default='/nfs/ibrahimi/runs/', help='Path to save runs to.')
parser.add_argument('--runsPath', type=str, default='/home/aesicd_42/CHANDRAJIT/Anuradha/pytorch-NetVlad-master/runs/', help='Path to save runs to.')

parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
#parser.add_argument('--cachePath', type=str, default=environ['TMPDIR'], help='Path to save cache to.')
parser.add_argument('--cachePath', type=str, default='/home/aesicd_42/CHANDRAJIT/Anuradha/pytorch-NetVlad-master/cache/', help='Path to save cache to.')

parser.add_argument('--resume', type=str, default='./vgg16_netvlad_checkpoint/', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='pittsburgh', 
        help='Dataset to use', choices=['pittsburgh'])
####################################################################
# Make change in model
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet','vggresnet'])
#####################################################################
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
#########################################################
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def save_checkpoint(state, is_best, filename='ST_checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Teacher Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

def Teacher_model():       
    encoder_dim = 512
    encoder_T = models.vgg16(pretrained=True)
    layers_T = list(encoder_T.features.children())[:-2]
    for l in layers_T[:-5]: 
        for p in l.parameters():
            p.requires_grad = False

    encoder_T = nn.Sequential(*layers_T)
    modelT = nn.Module() 
    modelT.add_module('encoder', encoder_T)
   
    net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
    modelT.add_module('pool', net_vlad)
    return modelT

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def Teacher_features(eval_set, model, epoch=0):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)
    encoder_dim = 512
    model.eval()
    with torch.no_grad():
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            del input, image_encoding, vlad_encoding
    del test_data_loader

    return dbFeat

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Teacher_output(whole_test_set):
    model = Teacher_model()    
    resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    opt.start_epoch = checkpoint['epoch']
    best_metric = checkpoint['best_score']
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
  
    Teacher_feats = Teacher_features(whole_test_set, model, epoch = 1)
    return Teacher_feats    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def Student_features(eval_set, model, epoch=0):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.train()
    encoder_dim = 512
    pool_size = encoder_dim
    if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
    dbFeat = np.empty((len(eval_set), pool_size))

    for iteration, (input, indices) in enumerate(test_data_loader, 1):
         input = input.to(device)
         image_encoding = model.encoder(input)
         vlad_encoding = model.pool(image_encoding) 

         dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
         del input, image_encoding, vlad_encoding
    del test_data_loader

    return dbFeat

def Student_model():
    encoder_dim = 512
    encoder_S = vggresnet(1000)
    layers_S = list([encoder_S.block_1, encoder_S.block_2, encoder_S.block_3, encoder_S.block_4, encoder_S.block_5])
        
    encoder_S = nn.Sequential(*layers_S)
    modelS = nn.Module() 
    modelS.add_module('encoder', encoder_S)
    
    net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
    modelS.add_module('pool', net_vlad)
    modelS = modelS.to(device)
    return modelS

def Student_output(eval_set):
    model = Student_model()
    model.train()
    Features = Student_features(eval_set, model, epoch=0)
    return Features

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def my_loss(scores, targets, T=5):
    soft_pred = softmax_op(torch.from_numpy(scores) / T)
    soft_targets = softmax_op(torch.from_numpy(targets) / T)
    loss = mseloss_fn(soft_pred, soft_targets)
    return loss
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def std_loss():
	
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
if __name__ == "__main__":
   
    opt = parser.parse_args()
    print(opt)

    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print('===> Loading dataset(s)')
    whole_train_set = dataset.get_whole_training_set()
    whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    train_set = dataset.get_training_query_set(opt.margin)
    print('====> Training query set:', len(train_set))
    whole_test_set = dataset.get_whole_val_set()
    print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   # print('====> Training query set:', len(train_set))
    #whole_test_set = dataset.get_whole_val_set()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    if opt.mode.lower() == 'train':
            model = Student_model() 
            if opt.optim.upper() == 'ADAM':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
            elif opt.optim.upper() == 'SGD':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
    else:
            raise ValueError('Unknown optimizer: ' + opt.optim)


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    
    writer = SummaryWriter(log_dir=join(opt.resume, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    opt.savePath = logdir
        
    if not opt.resume:
        makedirs(opt.savePath)

    with open(join(opt.savePath, 'flags.json'), 'w') as f:
        f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
    print('===> Saving state to:')#, logdir)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    not_improved = 0
    best_score = 0
    temp = 5
    lr = 5e-3
    
    model = Student_model()
    softmax_op = nn.Softmax(dim=1)
    mseloss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    val_acc = []
    train_acc = []
    train_loss = [0]
    for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
         scores = Student_output(whole_train_set)  
         targets = Teacher_output(whole_train_set)
         loss = my_loss(scores, targets, T = temp)
         optimizer.zero_grad()
        #  optimizer.Adam()
        #  print('***************************************************')
        #  print(loss)
         loss.backward()
        #  print('***************************************************')
        #  print(loss)
         optimizer.step()      
         print('epoch:', epoch,'; Loss:', loss)
         save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

               

    print('******************** Done *************************')
 
