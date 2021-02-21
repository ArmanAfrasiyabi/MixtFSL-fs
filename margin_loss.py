from __future__ import print_function
from data.tl_dataFunctions import ar_base_DataLaoder 
from utils import tsne_viz_centroids_freq, data_fun 
from backbones.utils import device_kwargs 
from backbones.utils import backboneSet
from backbones.utils import clear_temp
from torch.autograd import Variable
from args_parser import args_parser 
from utils import fun_metaLoader, euclidean_dist
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn
import numpy as np
import torch
import os
import math 


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)


class m_cos_loss(nn.Module):
    def __init__(self, n_class, out_dim, n_centroids, device, T, margin):
        super(m_cos_loss, self).__init__() 
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.components = nn.Parameter(torch.Tensor(n_class*n_centroids, out_dim)) 
        nn.init.kaiming_uniform_(self.components, a=math.sqrt(5))
        # nn.init.xavier_uniform_(self.centroids)   
        self.T = T
        self.margin = margin 
        self.device = device
        self.out_dim = out_dim
        self.n_centroids = n_centroids        
    def forward(self, z, components, y=None):  
        cosine = F.linear(F.normalize(z), F.normalize(components))
        # when test 
        if y is None:
            return cosine/self.T
        phi = cosine - self.margin
        output = torch.where(one_hot(y, cosine.shape[1]).byte(), phi, cosine)
        return output/self.T

    
    
 
    
    def initial_loss(self, f_net, bdata, n_centroids):  
        x, y = Variable(bdata[0]).to(self.device), Variable(bdata[1]).to(self.device)  
        centroids = F.normalize(self.components).view(-1, self.n_centroids, 
                                                 self.out_dim).mean(1)  
        yc, yn, ys = self.labeling(f_net, self.components, x, y, n_centroids)  
        f_net.train()
        z = f_net(x)
        scores_cp = self.forward(z, self.components, ys) 
        scores_cl = self.forward(z, centroids, y) 
        
        loss_cp = self.criterion(scores_cp, ys)
        loss_cl = self.criterion(scores_cl.detach(), y)
        
        return loss_cl, loss_cp, ys 
    
    
    
    # def loss(self, norm_z, norm_c, y): 
    #     cosine_theta = F.linear(norm_z, norm_c)   
    #     sine_theta = torch.sqrt(1.0 - torch.pow(cosine_theta, 2))               
    #     cons_theta_m = cosine_theta * self.cos_m - sine_theta * self.sin_m      
    #     cons_theta_m = torch.where(cosine_theta > 0, cons_theta_m, cosine_theta)
    #     y_1Hot = torch.zeros_like(cons_theta_m)          
    #     y_1Hot.scatter_(1, y.view(-1, 1), 1)            
    #     logits = (y_1Hot * cons_theta_m) + ((1.0 - y_1Hot) * cosine_theta)  
    #     return self.criterion(logits/self.T, y)
    
    
    
    
    def progressive_loss(self, f_net, p_net, p_centroids, bdata, n_centroids):  
        x, y = Variable(bdata[0]).to(self.device), Variable(bdata[1]).to(self.device)   
        yc, yn, ys = self.labeling(p_net, p_centroids.components, x, y, n_centroids)  
        f_net.train() 
        scores_cp = self.forward(f_net(x), self.components, ys)  
        loss_cp = self.criterion(scores_cp, ys) 
        return loss_cp, ys 
    
    
    def labeling(self, net, components, x, y, n_centroids):
        net.eval() 
        norm_z, norm_c = F.normalize(net(x)), F.normalize(components)
        norm_c = norm_c.view(-1, n_centroids, norm_c.shape[1]) 
        with torch.no_grad():
            ys = torch.zeros_like(y)
            for s in range(len(y)):
                norm_z_i = norm_z[s, :].unsqueeze(0)
                norm_c_i = norm_c[y[s],:,:]
                # D_i = euclidean_dist(norm_z_i, norm_c_i)
                D_i = -F.linear(norm_z_i, norm_c_i)
                low_ = y[s] * n_centroids
                ys[s] = torch.argmin(D_i) + low_
            yn = torch.zeros_like(ys)+10000
            k = 0
            for s in range(len(ys)):
                if yn[s] == 10000:
                    yn[ys==ys[s]] = k 
                    k += 1 
            # finding the unique set of the centroid's indexes
            yc = []   
            for x in ys:  
                if x.item() not in yc: 
                    yc.append(x.item())    
            yn = Variable(yn, requires_grad = False).to(self.device)
            yc = Variable(torch.tensor(np.asarray(yc)), requires_grad = False).to(self.device)
            return yc, yn, ys

 
class euc_centroidsLoss(nn.Module):
    def __init__(self, n_class, out_dim, n_centroids, device, T):
        super(euc_centroidsLoss, self).__init__() 
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.centroids = nn.Parameter(torch.Tensor(n_class*n_centroids, out_dim)) 
        nn.init.xavier_uniform_(self.centroids)   
        self.T = T 
        self.device = device
        
    def forward(self, z):  
        norm_z = F.normalize(z)                     
        norm_c = F.normalize(self.centroids)   
        return norm_z, norm_c 
    
    def loss(self, norm_z, norm_c, y): 
        logits = -euclidean_dist(norm_z, norm_c)   
        return self.criterion(logits/self.T, y)
     
    def phase_ii(self, f_net, p_net, p_centroids, bdata, n_centroids): 
        [x, y] = bdata
        x, y = Variable(x).to(self.device), Variable(y).to(self.device)     
        norm_zp, norm_cp = F.normalize(p_net(x)), F.normalize(p_centroids.centroids)    
        yc, yn, ys = self.labeling(norm_zp, norm_cp, y, n_centroids)    
        norm_z, norm_c = self.forward(f_net(x)) 
        loss = self.loss(norm_z, norm_c, ys) 
        return loss, ys 
     
    def phase_i(self, f_net, bdata, n_centroids):  
        [x, y] = bdata
        x, y = Variable(x).to(self.device), Variable(y).to(self.device)   
        norm_z, norm_c = self.forward(f_net(x))  
        yc, yn, ys = self.labeling(norm_z, norm_c, y, n_centroids)   
        proto_c = norm_c.view(-1, n_centroids, norm_c.shape[1]).mean(1)  
        colloboration_loss = self.loss(norm_z, proto_c.detach(), y)   
        competition_loss = self.loss(norm_z, norm_c[yc], yn)  
        return colloboration_loss, competition_loss, ys 
    
    def labeling(self, norm_z, norm_c, y, n_centroids): 
        norm_c = norm_c.view(-1, n_centroids, norm_c.shape[1]) 
        with torch.no_grad():
            ys = torch.zeros_like(y)
            for s in range(len(y)):
                norm_z_i = norm_z[s, :].unsqueeze(0)
                norm_c_i = norm_c[y[s],:,:]
                D_i = -euclidean_dist(norm_z_i, norm_c_i)   
                low_ = y[s] * n_centroids
                ys[s] = torch.argmax(D_i) + low_
            yn = torch.zeros_like(ys)+10000
            k = 0
            for s in range(len(ys)):
                if yn[s] == 10000:
                    yn[ys==ys[s]] = k 
                    k += 1 
            # finding the unique set of the centroid's indexes
            yc = []   
            for x in ys:  
                if x.item() not in yc: 
                    yc.append(x.item())    
            yn = Variable(yn, requires_grad = False).to(self.device)
            yc = Variable(torch.tensor(np.asarray(yc)), requires_grad = False).to(self.device)
            return yc, yn, ys