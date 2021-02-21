#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 01:07:48 2020

@author: ari
"""
import torchvision.models as models
from backbones.shallow_backbone import Conv4Net, Conv4Net_RN, Flatten
from backbones.wide_resnet import wide_resnet
from torch import nn
import shutil
import torch 
import os



#class linear_layer(nn.Module):
#    def __init__(self, out_size=25088, hid_dim=1024):
#        super(linear_layer, self).__init__()
#        self.fc = nn.Linear(out_size, hid_dim)
#        self.relu = nn.ReLU()
#        self.out = nn.Linear(hid_dim, hid_dim)
#
#    def forward(self, x):
#        z = self.relu(self.fc(x))
#        return self.out(z)


class linear_layer(nn.Module):
    def __init__(self, out_size=25088, hid_dim=1600):
        super(linear_layer, self).__init__()
        self.fc = nn.Linear(out_size, hid_dim)
        self.relu = nn.ReLU()

    def forward(self, x): 
        return self.relu(self.fc(x))



def clear_temp(data_path):
    if os.path.isdir(data_path+'temp'):
        for _, name_temp, _ in os.walk(data_path+'temp'): break
        if name_temp != []:
            shutil.move(data_path+'temp/'+name_temp[0], data_path)
        os.rmdir(data_path+'/temp')

def device_kwargs(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device

def backboneSet(args, fs_approach):
    if args.backbone=='Conv4' and args.method in ['RelationNet', 'RelationNet_softmax']:
        args.out_dim =  [64, 19, 19]
        net = Conv4Net_RN()
        args.img_size = 84 
    elif args.backbone=='Conv4': 
        net = Conv4Net() 
        args.out_dim = 1600
        args.img_size = 84 
        
#    elif args.backbone=='ResNet18':                                            ## paper reported number 73.68±0.65
#        net = models.resnet18(pretrained=False)          
#        args.out_dim = 1000
#        args.img_size = 224 
    elif args.backbone=='ResNet18':                                            ## paper reported number 73.68±0.65
        net = models.resnet18(pretrained=False)         
        net = list(net.children())[:-1] 
        net.append(Flatten()) 
        net = torch.nn.Sequential(*net) 
        
        # from backbones.ResNet18 import resnet18
        # net = resnet18(flatten=True)
        args.out_dim = 512
        args.img_size = 224 
        
        
    
        
        
    elif args.backbone=='WideResNet':                                          
        net = wide_resnet()
        args.out_dim = 640
        args.img_size = 80 
        
        
    elif args.backbone=='ResNet12':
        from backbones.ResNet12_embedding import resnet12
        net = resnet12(keep_prob=1.0, avg_pool=False)
        args.out_dim = 16000 
        args.img_size = 84 
        
        
    elif args.backbone=='ResNet12-640': 
        from backbones.res12 import resnet12
        net = resnet12() 
        args.out_dim = 640
         
        args.img_size = 84 
        
        
    else:
        raise "ar: please sepcify a valid backbone_type!"
        
    if args.dataset in ['miniImagenet', 'miniImagenet_forget']:
        args.n_base_class = 64
    elif args.dataset=='CUB':
        args.n_base_class = 100  
        args.n_query = 18
    elif args.dataset=='tieredImageNet':
        args.n_base_class = 351  
        
    elif args.dataset=='FC100':
        args.n_base_class = 60  
    else:    
        raise "ar: sepcify the number of base categories!"    
        
    file_name = str(args.test_n_way)+'way_'+str(args.n_shot)+'shot_'+args.dataset+'_'+args.method+'_'+args.backbone+'_bestModel.tar'     
    
    
#    print('approach|n_way/n_shot|dataset|method|backbone|out_dim|image_size| ', 
#          fs_approach, args.test_n_way, args.n_shot, args.dataset, args.method, 
#          args.backbone, args.out_dim, args.img_size)
            
    return args, net, file_name


