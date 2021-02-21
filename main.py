from __future__ import print_function
from data.tl_dataFunctions import ar_base_DataLaoder 
from utils import tsne_viz_centroids_freq, data_fun 
from backbones.utils import device_kwargs 
from backbones.utils import backboneSet
from backbones.utils import clear_temp 
from args_parser import args_parser 
from utils import fun_metaLoader, euclidean_dist
from centroid_clustering_loss import cos_centroidsLoss, euc_centroidsLoss
import torch.nn.functional as F
from torch.optim import Adam 
import numpy as np
import torch
import os

torch.cuda.empty_cache()
 
def initial_train(btrLoader, M, optimizer, device, n_centroids, epoch):  
    [f_net, f_centroids] = M   
    l_coll_list, l_comp_list = [], []   
    frequency_y = np.zeros(args.n_base_class * n_centroids)  
    f_net.train(), f_centroids.train()  
    for bdata in btrLoader: 
        l_coll, l_comp, ys = f_centroids.phase_i(f_net, bdata, n_centroids)  
        loss = l_coll + l_comp
        optimizer.zero_grad()   
        loss.backward() 
        optimizer.step()        
        l_coll_list.append(l_coll.item())   
        l_comp_list.append(l_comp.item())    
        frequency_y[ys.cpu().numpy().tolist()] += 1          
    M = [f_net.eval(), f_centroids.eval()]  
    trLoss = [np.average(l_coll_list), np.average(l_comp_list)]
    return trLoss, M, frequency_y
 
    
def steep_following(btrLoader, M, optimizer, device, n_centroids, epoch):  
    [f_net, f_centroids, p_net, p_centroids] = M   
    l_coll_list, l_comp_list = [], []   
    frequency_y = np.zeros(args.n_base_class * n_centroids)  
    f_net.train(), f_centroids.train() 
    p_net.eval(), p_centroids.eval() 
    for bdata in btrLoader: 
        loss, ys = f_centroids.phase_ii(f_net, p_net, p_centroids, bdata, n_centroids)   
        optimizer.zero_grad()   
        loss.backward() 
        optimizer.step()        
        l_coll_list.append(loss.item())   
        l_comp_list.append(loss.item())    
        frequency_y[ys.cpu().numpy().tolist()] += 1         
    M = [f_net.eval(), f_centroids.eval(), p_net.eval(), p_centroids.eval()]  
    trLoss = [np.average(l_coll_list), np.average(l_comp_list)]
    return trLoss, M, frequency_y


def accuracy_fun_tl(args, data_loader, net, device):        
    AccL = []
    net.eval()   
    for x, y in data_loader:
        with torch.no_grad(): 
            [x_support, y_support, x_query, y_query] = data_fun(args, x, device) 
            z_proto = net(x_support.to(device)).view(args.n_way, args.n_shot, -1).mean(1) 
            norm_w = F.normalize(z_proto)  
            norm_z = F.normalize(net(x_query.to(device))) 
            logits = F.linear(norm_z, norm_w)  
            y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)  
            AccL.append(np.mean((y_hat == y_query.cpu().numpy()).astype(int))*100) 
    print_txt = 'ep.%d    %4.2f/%4.2f    %4.2f  %4.2f   T:%4.2f ' 
    teAcc = np.mean(AccL) 
    acc_std  = np.std(AccL) 
    conf_interval = 1.96* acc_std/np.sqrt(len(data_loader))
    net.train()   
    return print_txt, teAcc, conf_interval
 
    
def meta_training_phase(args, f_net, file_name, device, resume, n_centroids, T=0.05, m=0.001, gamma=0.85): 
    maxAcc, first_epoch = 0, 0  
    paciency_exceeded = False
    paciency_th, non_update = 20, 0
    acc_set, conf_set, teacher_update, loss_set, T_set = [], [], [], [], [] 
    clear_temp(args.benchmarks_dir + args.dataset + '/base/') 
    vaLoader = fun_metaLoader(args, n_eposide=100, file_name = '/val.json')
    _, f_net_best, _ = backboneSet(args, fs_approach) 
    _, p_net, _ = backboneSet(args, fs_approach) 
    file_dir = file_name[:-4]+'_'+args.cenType+'_T'+str(T)+'_m'+str(m)+'_n_centroids'+str(n_centroids)+'.tar'   
    f_centroids = cos_centroidsLoss(args.n_base_class, args.out_dim, n_centroids, device, T, m)  
    f_centroids_best = cos_centroidsLoss(args.n_base_class, args.out_dim, n_centroids, device, T, m)   
    p_centroids = cos_centroidsLoss(args.n_base_class, args.out_dim, n_centroids, args.batch_size, T, m)  
    if resume: 
        print('---------------------------------------------') 
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_dir)) 
        first_epoch = checkpoint['epoch']
        maxAcc = checkpoint['maxAcc']  
        print('resume: up to now the best model has:', str(maxAcc), 'accuracy!')
        f_net_best.load_state_dict(checkpoint['f_net_best']) 
        f_centroids_best.load_state_dict(checkpoint['f_centroids_best'])  
        f_net.load_state_dict(checkpoint['f_net_current']) 
        f_centroids.load_state_dict(checkpoint['f_centroids_current'])  
        f_centroids.T = checkpoint['T']
        n_centroids = checkpoint['n_centroids']
        best_frequency_y = checkpoint['best_frequency_y']
        frequency_y = checkpoint['frequency_y']
        print('---------------------------------------------')      
    optimizer = Adam([{'params': f_net.parameters()}, 
                      {'params': f_centroids.parameters()}], lr = args.lr)       
    btrLoader = ar_base_DataLaoder(args, aug=True, shuffle=True)    
    M = [f_net.to(device), f_centroids.to(device)]   
    for epoch in range(first_epoch, args.n_epoch):   
        if non_update>=paciency_th:
            paciency_exceeded = True
            non_update = 0
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_dir)) 
            p_net.load_state_dict(checkpoint['f_net_best']) 
            p_centroids.load_state_dict(checkpoint['f_centroids_best'])  
            maxAcc = checkpoint['maxAcc']  
            print('---------------------------------------------')  
            print('Phase II: steep following: up to now the best model has:', str(maxAcc), 'accuracy!')
            print('---------------------------------------------')   
            M = [f_net.to(device), f_centroids.to(device), p_net.to(device), p_centroids.to(device)] 
            teacher_update.append(epoch) 
            f_centroids.T = f_centroids.T*gamma
            # paciency_th = 10
        if not paciency_exceeded:
            if epoch==0:
                print('---------------------------------------------')  
                print('--         Phase I: initial training!      --')
                print('---------------------------------------------')   
            trLoss, M, frequency_y = initial_train(btrLoader, M, optimizer, device, n_centroids, epoch)  
        else:
            trLoss, M, frequency_y = steep_following(btrLoader, M, optimizer, device, n_centroids, epoch)   
            
        print_txt, vaAcc, vaVar = accuracy_fun_tl(args, vaLoader, M[0], device)  
        if vaAcc > (maxAcc - 0.1):  
            if (vaAcc > maxAcc): maxAcc = vaAcc   
            f_centroids_best.load_state_dict(f_centroids.state_dict()) 
            f_net_best.load_state_dict(f_net.state_dict()) 
            print_txt = print_txt + 'up!'  
            best_frequency_y = frequency_y
            non_update = 0
        else: 
            non_update += 1    
        print(print_txt % (epoch, trLoss[0], trLoss[1], vaAcc, vaVar, f_centroids.T))     
        if epoch+1==15: vaLoader = fun_metaLoader(args, n_eposide=600, file_name = '/val.json')
        acc_set.append(vaAcc)
        conf_set.append(vaVar) 
        T_set.append(f_centroids.T)
        loss_set.append([trLoss]) 
        torch.save({'epoch': epoch, 
                    'f_net_current': M[0].state_dict(), 
                    'f_centroids_current': M[1].state_dict(), 
                    'f_net_best': f_net_best.state_dict(),
                    'f_centroids_best': f_centroids_best.state_dict(),
                    'T': f_centroids.T,
                    'n_centroids': n_centroids,
                    'teacher_update': teacher_update,
                    'frequency_y':frequency_y, 
                    'best_frequency_y':best_frequency_y, 
                    'acc_set': acc_set,
                    'conf_set': conf_set, 
                    'loss_set': loss_set,
                    'T_set': T_set,
                    'maxAcc': maxAcc}, os.path.join(args.checkpoint_dir, file_dir))   


if __name__ == '__main__':
    fs_approach = 'transfer-learning' 
    args = args_parser(fs_approach)
    args.n_shot = 1
    args.backbone = 'ResNet12'
    args, f_net, file_name = backboneSet(args, fs_approach) 
    device = device_kwargs(args)    
    n_centroids = 10
    resume = False 
    n_class = 64  
    T = 0.025
    if args.backbone == 'ResNet12':
        if args.n_shot == 5:
            m = -0.02
        elif args.n_shot == 1:
            m=-0.01 
    elif args.backbone == 'ResNet18':
        if args.n_shot == 5:
            m = -0.02
        elif args.n_shot == 1:
            m= -0.02   
    else:
        m = -0.02  
    gamma = 0.95
    print('---------------------------------------------') 
    print('--                   setup                 --') 
    print('---------------------------------------------')   
    print('  distance/similarity: '+args.cenType) 
    print('  dataset/backbone: '+str(args.dataset)+'/'+args.backbone) 
    print('  n_centroids: '+str(n_centroids)) 
    print('  n_shot: '+str(args.n_shot)) 
    print('  temperature: '+str(T))
    print('  margin: '+str(m))
    print('  gamma(T.Scale): '+str(gamma))
    print('  resume: '+str(resume))  
    meta_training_phase(args, f_net, file_name, device, resume, n_centroids, T, m, gamma)    



 
 
