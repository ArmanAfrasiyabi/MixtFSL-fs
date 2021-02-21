#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 02:31:23 2020

@author: ari
"""

from __future__ import print_function 
from methods.transferLearning_clfHeads import softMax, cosMax, arcMax
from torch.autograd import Variable
from torch.optim import  Adam
import numpy as np
import torch 
import torch.nn as nn
import torchvision.models as models
from backbones.shallow_backbone import Conv4Net, Flatten
from backbones.wide_resnet import wide_resnet
from backbones.utils import device_kwargs
from cent_max_.cent_max import euclidean_dist

def clf_optimizer(self, n_class, device, frozen_net, s = 15, m = 0.0001):
    if frozen_net: s=5
    if self.method == 'softMax':
        clf = softMax(self.out_dim, n_class).to(device)
    elif self.method == 'cosMax':
        clf = cosMax(self.out_dim, n_class, s).to(device)
    elif self.method == 'arcMax':
        clf = arcMax(self.out_dim, n_class, s, m).to(device)
    elif self.method == 'centerMax':
        clf = softMax(self.out_dim, n_class).to(device)
    if frozen_net:  
        optimizer = torch.optim.Adam(clf.parameters(), lr = self.lr)  
    else:
        optimizer = Adam([{'params': self.net.parameters()}, 
                          {'params': clf.parameters()}], 
                          lr = self.lr) 
    return clf, optimizer


def clf_fun(self, n_class, device, s = 10, m = 0.2):
    if self.method == 'softMax':
        clf = softMax(self.out_dim, n_class).to(device)
    elif self.method == 'cosMax':
        clf = cosMax(self.out_dim, n_class, s).to(device)
    elif self.method == 'arcMax' or self.method == 'centerMax':
        clf = arcMax(self.out_dim, n_class, s, m).to(device)  
    return clf

class transferLearningFuns(nn.Module):
    def __init__(self, args, net, n_class):
        super(transferLearningFuns, self).__init__()
        
        self.device = device_kwargs(args) 
        self.method = args.method
        self.lr = args.lr
        self.backbone = args.backbone
        
        self.n_way = args.n_way
        
        self.n_epoch = args.n_epoch
        self.n_class = n_class
        self.n_support = args.n_shot
        self.n_query = args.n_query
        
        self.out_dim = args.out_dim
        self.lr = args.lr
        self.net = net.to(self.device)
        self.over_fineTune = args.over_fineTune
        
        self.ft_n_epoch = args.ft_n_epoch
        self.frozen_net = True
        
        self.base_clf, self.optimizer = clf_optimizer(self, self.n_class, self.device, frozen_net = False)
        
    def accuracy_fun_tl(self, data_loader):        # this is typical batch based testing (should be only used for base categories) 
        Acc = 0
        self.net.eval()  
        with torch.no_grad(): 
            for x, y in data_loader:
                x, y = Variable(x).to(self.device), Variable(y).to(self.device)
                logits = self.clf(self.net(x))
                y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
                Acc += np.mean((y_hat == y.data.cpu().numpy()).astype(int))
        return Acc.item()/len(data_loader) 
    
    def accuracy_fun(self, x, n_way): #, p_net):             
        novel_clf = clf_fun(self, self.n_way, self.device, s=15) 
        novel_optimizer = torch.optim.Adam(novel_clf.parameters(), lr = self.lr)   
        x_support   = x[:, :self.n_support, :,:,:].contiguous()
        x_support = x_support.view(n_way * self.n_support, *x.size()[2:]) 
        y_support = torch.from_numpy(np.repeat(range(n_way), self.n_support))
        y_support = Variable(y_support.to(self.device))
        
        with torch.no_grad():
            #z_support  = self.net(p_net(x_support))
            z_support  = self.net(x_support)
        for epoch in range(self.ft_n_epoch):
            loss = novel_clf.loss(novel_clf(z_support), y_support) 
            novel_optimizer.zero_grad()
            loss.backward()
            novel_optimizer.step()    
            
        if self.over_fineTune:
            print('over-ft')
            if self.backbone=='Conv4':
                net = Conv4Net().to(self.device)
            elif self.backbone=='ResNet18':
                net = models.resnet18(pretrained=False)         
                net = list(net.children())[:-1] 
                net.append(Flatten())
                net = torch.nn.Sequential(*net) 
            elif self.backbone=='WideResNet':                                          
                net = wide_resnet()
            else:
                raise "ar: please sepcify a valid backbone_type!" 
            net.load_state_dict(self.net.state_dict())
            net.train()
            novel_optimizer = Adam([{'params': net.parameters()}, 
                                    {'params': novel_clf.parameters()}], 
                                    lr = self.lr) 
            for epoch in range(self.ft_n_epoch):
                z_support  = net(x_support)
                loss = novel_clf.loss(novel_clf(z_support), y_support) 
                novel_optimizer.zero_grad()
                loss.backward()
                novel_optimizer.step()  
            net.eval()  
            x_query = x[:, self.n_support:, :,:,:].contiguous()
            x_query = x_query.view(n_way * self.n_query, *x.size()[2:]) 
            y_query = torch.from_numpy(np.repeat(range(n_way), self.n_query))
            y_query = Variable(y_query.cuda())
            logits = novel_clf(net(x_query))
            y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)  
            return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int))*100
        
        x_query = x[:, self.n_support:, :,:,:].contiguous()
        x_query = x_query.view(n_way * self.n_query, *x.size()[2:]) 
        y_query = torch.from_numpy(np.repeat(range(n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        #logits = novel_clf(self.net(p_net(x_query)))
        logits = novel_clf(self.net(x_query))
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)  
        return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int))*100
        
    # note: this is typical/batch based training
    def train_loop(self, trainLoader): 
        self.net.train() 
        loss_sum = 0
        for i, (x, y) in enumerate(trainLoader):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            loss = self.base_clf.loss(self.base_clf(self.net(x)), y) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()   
        return loss_sum/len(trainLoader)
    
 
    def labeling_fun_P(self, x, y, P, n_centroids=2):
        with torch.no_grad(): 
            [p_net, p_centroids] = P
            p_net.eval()
            yc = torch.zeros_like(y) 
            z = p_net(x)
            for s in range(len(y)):
                low_ = y[s] * n_centroids 
                norm_c_s = p_centroids[y[s]][0].to(self.device)
                scores = euclidean_dist(z[s,:].unsqueeze(0), norm_c_s)    
                yc[s] = torch.argmin(scores) + low_   
            return Variable(yc).to(self.device) 
    
    
    def train_temp_loop(self, trainLoader, P): 
        n_centroids = 2
        n_base_class = 64
        frequency_y_f = np.zeros(n_base_class * n_centroids)  
        self.net.train() 
        loss_sum = 0
        for i, (x, y) in enumerate(trainLoader):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            
            y = self.labeling_fun_P(x, y, P) 
            loss = self.base_clf.loss(self.base_clf(self.net(x)), y) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()   
        return loss_sum/len(trainLoader), frequency_y_f
    
#    #####################################################
#    def train_1loop(self, x, y):
#        loss = self.base_clf.loss(self.base_clf(self.net(x)), y) 
#        self.optimizer.zero_grad()
#        loss.backward()
#        self.optimizer.step()
#        return loss.item() 
#    
#    def test_temp_loop(self, p_net, test_loader, n_way):
#        acc_all = []
#        iter_num = len(test_loader) 
#        p_net.eval()
#        for i, (x,_) in enumerate(test_loader):
#            x = Variable(x).to(self.device) 
#            self.n_query = x.size(1) - self.n_support
#            acc_all.append(self.accuracy_fun(x, n_way, p_net))  
#        acc_all  = np.asarray(acc_all) 
#        teAcc = np.mean(acc_all)
#        acc_std  = np.std(acc_all)
#        conf_interval = 1.96* acc_std/np.sqrt(iter_num)
#        return teAcc, conf_interval
    
    
 
    
    #####################################################
    # note this is episodic testing 
    def test_loop(self, test_loader, n_way):
        acc_all = []
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            x = Variable(x).to(self.device)
            self.n_query = x.size(1) - self.n_support
            acc_all.append(self.accuracy_fun(x, n_way))  
        acc_all  = np.asarray(acc_all) 
        teAcc = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        conf_interval = 1.96* acc_std/np.sqrt(iter_num)
        return teAcc, conf_interval