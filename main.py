from __future__ import print_function
from data.tl_dataFunctions import ar_base_DataLaoder
from utils import tsne_viz_centroids_freq, data_fun
from backbones.utils import device_kwargs
from backbones.utils import backboneSet
from backbones.utils import clear_temp
from args_parser import args_parser
from utils import fun_metaLoader, euclidean_dist
from methods.centroid_clustering_loss import cos_centroidsLoss, euc_centroidsLoss
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
import torch
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import GradualWarmupScheduler


def optimizer_scheduler(opts, f, centroids):
    if opts.backbone == 'Conv4':
        optimizer_ = Adam([{'params': f.parameters()},
                           {'params': centroids.parameters()}], lr=args.lr)
        lr_scheduler_ = None
    elif opts.backbone in ['ResNet12', 'ResNet18']:
        opts.lr = 0.1
        optimizer_ = torch.optim.SGD([{'params': f.parameters(), 'lr': opts.lr},
                                      {'params': centroids.parameters(), 'lr': opts.lr}
                                      ], momentum=0.9, nesterov=True, weight_decay=5e-04)
        lr_scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=25, gamma=0.2)
    print(' ', opts.backbone, 'optimizer set!')
    return f, optimizer_, lr_scheduler_


def initial_train(btrLoader, M, optimizer, n_centroids):
    [f_net, f_centroids] = M
    l_coll_list, l_comp_list = [], []
    frequency_y = np.zeros(args.n_base_class * n_centroids)
    f_net.train(), f_centroids.train()
    for i, bdata in enumerate(btrLoader):
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


def progressive_following(btrLoader, M, optimizer, n_centroids):
    [f_net, f_centroids, p_net, p_centroids] = M
    l_coll_list, l_comp_list = [], []
    frequency_y = np.zeros(args.n_base_class * n_centroids)
    f_net.train(), f_centroids.train()
    p_net.eval(), p_centroids.eval()
    for i, bdata in enumerate(btrLoader):
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


def accuracy_fun_cent(args, data_loader, net, device):
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
            AccL.append(np.mean((y_hat == y_query.cpu().numpy()).astype(int)) * 100)
    print_txt = 'ep.%d    %4.2f/%4.2f    %4.2f  %4.2f   T:%4.2f '
    teAcc = np.mean(AccL)
    acc_std = np.std(AccL)
    conf_interval = 1.96 * acc_std / np.sqrt(len(data_loader))
    net.train()
    return print_txt, teAcc, conf_interval


def fine_tuning(args, net, data, device, T, m):
    clf = cos_centroidsLoss(args.n_way, args.out_dim, 1, device, T, m)
    clf = clf.to(device)
    [x_support, y_support] = data
    optimizer = Adam(clf.parameters(), lr=args.lr)
    with torch.no_grad():
        z_support = net(x_support)
    for _ in range(args.ft_n_epoch):
        norm_z, norm_c = clf(z_support)
        loss = clf.loss(norm_z, norm_c, y_support)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return clf.eval()


def accuracy_fun_tl(args, data_loader, net, device, T_te, m_te):
    Acc_ft, Acc_cn = [], []
    net.eval()
    for i, (x, y) in enumerate(data_loader):
        [x_support, y_support, x_query, y_query] = data_fun(args, x, device)
        # net.train()
        clf = fine_tuning(args, net, [x_support, y_support], device, T_te, m_te)
        with torch.no_grad():
            # net.eval()
            logits = clf.pred(net(x_query.to(device)))
            y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
            Acc_ft.append(np.mean((y_hat == y_query.cpu().numpy()).astype(int)) * 100)

            z_proto = net(x_support.to(device)).view(args.n_way, args.n_shot, -1).mean(1)
            norm_w = F.normalize(z_proto)
            norm_z = F.normalize(net(x_query.to(device)))
            logits = F.linear(norm_z, norm_w)
            y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
            Acc_cn.append(np.mean((y_hat == y_query.cpu().numpy()).astype(int)) * 100)

    print_txt = 'ep.%d    %4.2f/%4.2f  %4.2f(±%4.2f)   %4.2f(±%4.2f)   T:%4.2f '
    teAcc_ft = np.mean(Acc_ft)
    acc_std_ft = np.std(Acc_ft)
    conf_interval_ft = 1.96 * acc_std_ft / np.sqrt(len(data_loader))

    teAcc_cn = np.mean(Acc_cn)
    acc_std_cn = np.std(Acc_cn)
    conf_interval_cn = 1.96 * acc_std_cn / np.sqrt(len(data_loader))
    net.train()
    return print_txt, teAcc_ft, conf_interval_ft, teAcc_cn, conf_interval_cn


def meta_training_phase(args, f_net, vaLoader, file_name, device, resume, n_centroids, T, m, T_te, m_te, gamma):
    maxAcc, first_epoch = 0, 0
    paciency_exceeded = False
    paciency_th, non_update = 20, 0
    acc_set_ft, conf_set_ft, acc_set_cn, conf_set_cn, teacher_update, loss_set, T_set = [], [], [], [], [], [], []
    clear_temp(args.benchmarks_dir + args.dataset + '/base/')
    _, f_net_best, _ = backboneSet(args)
    _, p_net, _ = backboneSet(args)
    file_dir = file_name[:-4] + '_' + args.cenType + '_T' + str(T) + '_m' + str(m) + '_n_centroids' + str(
        n_centroids) + '.tar'
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
        # frequency_y = checkpoint['frequency_y']
        print('---------------------------------------------')
    f_net, optimizer, lr_scheduler = optimizer_scheduler(args, f_net, f_centroids)
    btrLoader = ar_base_DataLaoder(args, aug=True, shuffle=True)
    M = [f_net.to(device), f_centroids.to(device)]
    for epoch in range(first_epoch, args.n_epoch):
        if non_update >= paciency_th:
            paciency_exceeded = True
            non_update = 0
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_dir))
            p_net.load_state_dict(checkpoint['f_net_best'])
            p_centroids.load_state_dict(checkpoint['f_centroids_best'])
            maxAcc = checkpoint['maxAcc']
            print('---------------------------------------------')
            print('Phase II: progressive following: up to now the best model has:', str(maxAcc), 'accuracy!')
            print('---------------------------------------------')
            M = [f_net.to(device), f_centroids.to(device), p_net.to(device), p_centroids.to(device)]
            teacher_update.append(epoch)
            f_centroids.T = f_centroids.T * gamma
        if not paciency_exceeded:
            if epoch == 0:
                print('---------------------------------------------')
                print('--         Phase I: initial training!      --')
                print('---------------------------------------------')
            trLoss, M, frequency_y = initial_train(btrLoader, M, optimizer, n_centroids)
        else:
            M = [f_net.to(device), f_centroids.to(device), p_net.to(device), p_centroids.to(device)]
            trLoss, M, frequency_y = progressive_following(btrLoader, M, optimizer, n_centroids)
            paciency_th = 15
        # print_txt, vaAcc, vaVar = accuracy_fun_tl(args, vaLoader, M[0], device)  
        print_txt, vaAcc_ft, vaVar_ft, vaAcc_cn, vaVar_cn = accuracy_fun_tl(args, vaLoader, M[0], device, T_te, m_te)
        vaAcc = max(vaAcc_ft, vaAcc_cn)
        if vaAcc > (maxAcc - 0.1):
            if (vaAcc > maxAcc): maxAcc = vaAcc
            f_centroids_best.load_state_dict(f_centroids.state_dict())
            f_net_best.load_state_dict(f_net.state_dict())
            print_txt = print_txt + 'up!'
            best_frequency_y = frequency_y
            non_update = 0
        else:
            non_update += 1
        print(print_txt % (epoch, trLoss[0], trLoss[1], vaAcc_ft, vaVar_ft, vaAcc_cn, vaVar_cn, f_centroids.T))
        # if epoch+1==15: vaLoader = fun_metaLoader(args, n_eposide=600, file_name = '/novel.json')    
        # tsne_viz_centroids_freq(args, f_centroids, frequency_y, n_centroids, epoch, vaAcc, 'centViz_em', '.pdf')     
        acc_set_ft.append(vaAcc_ft)
        conf_set_ft.append(vaVar_ft)
        acc_set_cn.append(vaAcc_cn)
        conf_set_cn.append(vaVar_cn)

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
                    'frequency_y': frequency_y,
                    'best_frequency_y': best_frequency_y,
                    'acc_set_ft': acc_set_ft,
                    'conf_set_ft': conf_set_ft,
                    'acc_set_cn': acc_set_cn,
                    'conf_set_cn': conf_set_cn,
                    'loss_set': loss_set,
                    'T_set': T_set,
                    'maxAcc': maxAcc}, os.path.join(args.checkpoint_dir, file_dir))
        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == '__main__':
    fs_approach = 'transfer-learning'
    args = args_parser(fs_approach)
    args.n_shot = 1
    args.backbone = 'ResNet12'
    args, f_net, file_name = backboneSet(args)
    vaLoader = fun_metaLoader(args, n_eposide=600, file_name='/val.json')
    device = device_kwargs(args)
    n_centroids = 15
    resume = False
    T = 0.025
    m = 0.01
    gamma = 0.95
    T_te = 0.2
    m_te = 0.001
    print('---------------------------------------------')
    print('--                   setup                 --')
    print('---------------------------------------------')
    print('  distance/similarity: ' + args.cenType)
    print('  dataset/backbone: ' + str(args.dataset) + '/' + args.backbone)
    print('  n_centroids: ' + str(n_centroids))
    print('  n_way: ' + str(args.test_n_way))
    print('  n_shot: ' + str(args.n_shot))
    print('  temperature: ' + str(T) + '/' + str(T_te))
    print('  gamma(T.Scale): ' + str(gamma))
    print('  margine: ' + str(m) + '/' + str(m_te))
    print('---------------------------------------------')

    print('  resume: ' + str(resume))
    meta_training_phase(args, f_net, vaLoader, file_name, device, resume, n_centroids, T, m,
                        T_te, m_te, gamma)

    teLoader = fun_metaLoader(args, n_eposide=600, file_name='/novel.json')
    file_dir = file_name[:-4] + '_' + args.cenType + '_T' + str(T) + '_m' + str(m) + '_n_centroids' + str(
        n_centroids) + '.tar'
    print('---------------------------------------------')
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_dir))
    first_epoch = checkpoint['epoch']
    maxAcc = checkpoint['maxAcc']
    print('testing: the best model has:', str(maxAcc), 'accuracy on validation set!')
    f_net.load_state_dict(checkpoint['f_net_best'])
    print('---------------------------------------------')

    f_net = f_net.to(device)
    _, teAcc, teVar = accuracy_fun_tl(args, teLoader, f_net.eval(), device, T_te, m_te)
    print_txt = 'ep.%d    %4.2f ± %4.2f             T:%4.2f/%4.2f|%d '
    print(print_txt % (args.seed, teAcc, teVar, T_te, m_te, args.seed))
