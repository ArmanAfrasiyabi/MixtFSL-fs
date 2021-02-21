from __future__ import print_function
from data.tl_dataFunctions import ar_base_DataLaoder
from data.ml_dataFunctions import SetDataManager
from backbones.utils import device_kwargs
from backbones.utils import backboneSet
from backbones.utils import clear_temp
from torch.autograd import Variable
from args_parser import args_parser
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn
import numpy as np
import torch
import os
from scipy.special import softmax
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors







def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)





def tsne_viz_z(z, y, n_centroids, i=0, fileName='z_viz'):
    z = z.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)
    plt.figure(figsize=(16, 16))
    colors = ['gray', 'blue', 'green', 'red', 'black', 'orange']

    plt.title('tsne || Starts: base centroids(c-' + str(n_centroids) + '/64-way) || Circles: novel centroids (5-way)')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)

    for n in range(5):
        z_2D_ = z_2D[y == n]
        plt.scatter(z_2D_[:, 0], z_2D_[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

    plt.savefig('./results/tsne/' + fileName + '_' + str(i) + '.pdf')
    plt.close('all')


def tsne_viz_centroids_z(net, args, tr_centroids, n_centroids, i=0, vaAcc=None, fileName='centroid_viz'):
    args.batch_size = 6000
    trLoader = ar_base_DataLaoder(args, aug=True)
    for i, (x, y_data) in enumerate(trLoader):
        x = Variable(x)
        break
    with torch.no_grad():
        net.cpu()
        z_data = net(x)

    norm_c = F.normalize(tr_centroids.centroids)
    z_data = F.normalize(z_data)

    z_centroids = norm_c[0:n_centroids * args.n_base_class, :].cpu().detach()
    z = torch.cat((z_data, z_centroids), dim=0).numpy()

    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)
    z_2D_data = z_2D[0:z_data.shape[0], :]
    z_2D_centroids = z_2D[z_data.shape[0]:, :]

    plt.figure(figsize=(16, 16))
    plt.rcParams.update({'font.size': 22})
    colors = []
    for key in mcolors.CSS4_COLORS:
        colors.append(key)
    if vaAcc != None:
        plt.title(args.backbone + '/tsne -- circles/starts: base centroids| base embeddings (c-' + str(
            n_centroids) + '/64-way)')
    else:
        plt.title(args.backbone + '/tsne -- circles/starts: base centroids| base embeddings (c-' + str(
            n_centroids) + '/64-way).')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    y = []
    for n in range(n_centroids):
        for c in range(args.n_base_class):
            y.append(c)
    y = np.array(y)
    for n in range(args.n_base_class):
        z_2D_d = z_2D_data[y_data == n]
        z_2D_c = z_2D_centroids[y == n]
        plt.scatter(z_2D_d[:, 0], z_2D_d[:, 1], cmap='viridis', marker="*", s=300, color=colors[n], alpha=0.9,
                    edgecolors='white')
        plt.scatter(z_2D_c[:, 0], z_2D_c[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

    #        if n==1: break

    ### ############################ 
    plt.savefig('./results/tsne/' + fileName + '_z_' + str(i) + '.pdf')
    plt.close('all')


def tsne_viz_centroids_z_query(args, tr_centroids, z_support, z_qeury, y_support, y_qeury, i=0,
                               fileName='centroid_viz'):
    n_centroids = 1

    y_qeury = y_qeury.cpu().detach()
    y_support = y_support.cpu().detach()

    z_centroids = F.normalize(tr_centroids.centroids).cpu().detach()
    z_support = F.normalize(z_support).cpu().detach()
    z_qeury = F.normalize(z_qeury).cpu().detach()

    z = torch.cat((z_support, z_qeury, z_centroids), dim=0).numpy()

    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)

    z_2D_supp = z_2D[0:z_support.shape[0], :]
    z_2D_query = z_2D[z_support.shape[0]:z_support.shape[0] + z_qeury.shape[0], :]
    z_2D_centroids = z_2D[z_support.shape[0] + z_qeury.shape[0]:, :]

    colors = ['red', 'blue', 'black', 'orange', 'green']

    plt.figure(figsize=(16, 16))
    plt.rcParams.update({'font.size': 22})
    #    colors = []
    #    for key in mcolors.CSS4_COLORS:
    #        colors.append(key)

    plt.title(
        args.backbone + '/tsne -- circles/starts: base centroids| base embeddings (c-' + str(n_centroids) + '/64-way)')

    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    y = []
    for n in range(n_centroids):
        for c in range(args.n_way):
            y.append(c)
    y = np.array(y)
    for n in range(args.n_way):
        z_2D_s = z_2D_supp[y_support == n]
        z_2D_q = z_2D_query[y_qeury == n]
        z_2D_c = z_2D_centroids[y == n]
        plt.scatter(z_2D_s[:, 0], z_2D_s[:, 1], cmap='viridis', marker="*", s=300, color=colors[n], alpha=1,
                    edgecolors='black')
        plt.scatter(z_2D_q[:, 0], z_2D_q[:, 1], cmap='viridis', marker="^", s=200, color=colors[n], alpha=1,
                    edgecolors='white')
        plt.scatter(z_2D_c[:, 0], z_2D_c[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

        ### ############################
    plt.savefig('./results/tsne/' + fileName + '_z_' + str(i) + '.pdf')
    plt.close('all')


def tsne_viz_centroids_1(args, tr_centroids, n_centroids, i=0, vaAcc=None, fileName='centroid_viz'):
    #    z = tr_centroids.centroids[0:n_centroids*64,:].cpu().detach().numpy()
    #    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)

    z = tr_centroids.centroids.cpu().detach().numpy()
    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)

    plt.figure(figsize=(16, 16))

    plt.rcParams.update({'font.size': 22})
    colors = []
    for key in mcolors.CSS4_COLORS:
        colors.append(key)
    if vaAcc != None:
        plt.title(
            args.backbone + '/tsne -- circles: base centroids (c-' + str(n_centroids) + '/64-way) -- [vaAcc: ' + str(
                round(vaAcc, 2)) + ']')
    else:
        plt.title(args.backbone + '/tsne-- circles: base centroids (c-' + str(n_centroids) + '/64-way).')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    y = []

    counter = 0
    for c in range(args.n_base_class):
        for n in range(n_centroids):
            y.append(counter)
        counter += 1

    z_2D_centroids = z_2D[n_centroids * 64:, :]
    plt.scatter(z_2D_centroids[:, 0], z_2D_centroids[:, 1], cmap='viridis', marker="*", s=200, color='white', alpha=1,
                edgecolors='black')

    y = np.array(y)
    z_2D_centroids = z_2D[0:n_centroids * 64, :]
    for n in range(args.n_base_class):
        z_2D_ = z_2D_centroids[y == n]
        plt.scatter(z_2D_[:, 0], z_2D_[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

        ### ############################
    plt.savefig('./results/tsne/_' + fileName + '_' + str(i) + '_aug.pdf')
    plt.close('all')


def tsne_viz_centroids_2(args, tr_centroids, n_centroids, i=0, vaAcc=None, fileName='centroid_viz'):
    z = tr_centroids.centroids[0:n_centroids * 64, :].cpu().detach().numpy()
    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)

    plt.figure(figsize=(16, 16))

    plt.rcParams.update({'font.size': 22})
    colors = []
    for key in mcolors.CSS4_COLORS:
        colors.append(key)
    if vaAcc != None:
        plt.title(
            args.backbone + '/tsne -- circles: base centroids (c-' + str(n_centroids) + '/64-way) -- [vaAcc: ' + str(
                round(vaAcc, 2)) + ']')
    else:
        plt.title(args.backbone + '/tsne-- circles: base centroids (c-' + str(n_centroids) + '/64-way).')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    y = [] 
    counter = 0
    for c in range(args.n_base_class):
        for n in range(n_centroids):
            y.append(counter)
        counter += 1

    y = np.array(y)
    for n in range(args.n_base_class):
        z_2D_ = z_2D[y == n]
        plt.scatter(z_2D_[:, 0], z_2D_[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

        ### ############################
    plt.savefig('./results/tsne/' + fileName + '_' + str(i) + '.png')
    plt.close('all')
    
    
def tsne_viz_centroids_3(args, tr_centroids, n_centroids, cent_surv_list, i=0, vaAcc=None, fileName='centroid_viz'):
    y = [] 
    counter = 0
    for c in range(args.n_base_class):
        for n in range(n_centroids):
            y.append(counter)
        counter += 1 
    y = np.array(y)
    
    
#    centroids = tr_centroids.G(tr_centroids.centroids)
    z = tr_centroids.centroids[0:n_centroids * 64, :].cpu().detach().numpy()
    z_ = []
    y_ = []
    for n in range(args.n_base_class):
        cent_surv_list_i = cent_surv_list[n]
        z_.extend(z[y == n][cent_surv_list_i])
        y_.extend(y[y == n][cent_surv_list_i])
    
    z = np.array(z_) 
    y = np.array(y_) 
    
    
    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)

    plt.figure(figsize=(16, 16))

    plt.rcParams.update({'font.size': 22})
    colors = []
    for key in mcolors.CSS4_COLORS:
        colors.append(key)
    if vaAcc != None:
        plt.title(
            args.backbone + '/tsne -- circles: base centroids (c-' + str(n_centroids) + '/64-way) -- [vaAcc: ' + str(
                round(vaAcc, 2)) + ']')
    else:
        plt.title(args.backbone + '/tsne-- circles: base centroids (c-' + str(n_centroids) + '/64-way).')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    
    for n in range(args.n_base_class):
#        cent_surv_list_i = cent_surv_list[n]
        z_2D_ = z_2D[y == n]#[cent_surv_list_i]
        plt.scatter(z_2D_[:, 0], z_2D_[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

        ### ############################
    plt.savefig('./results/tsne/' + fileName + '_' + str(i) + '.png')
    plt.close('all')
    
    
def tsne_viz_centroids_4(args, tr_centroids, g_net, n_centroids, cent_surv_list, i=0, vaAcc=None, fileName='centroid_viz'):
    y = [] 
    counter = 0
    for c in range(args.n_base_class):
        for n in range(n_centroids):
            y.append(counter)
        counter += 1 
    y = np.array(y)
    
    
    centroids = g_net(tr_centroids.centroids)
    z = centroids[0:n_centroids * 64, :].cpu().detach().numpy()
#    z_ = []
#    for n in range(args.n_base_class):
#        cent_surv_list_i = cent_surv_list[n]
#        z_.append(z[y == n][cent_surv_list_i])
    
    
    
    z_2D = TSNE(n_components=2, perplexity=20).fit_transform(z)

    plt.figure(figsize=(16, 16))

    plt.rcParams.update({'font.size': 22})
    colors = []
    for key in mcolors.CSS4_COLORS:
        colors.append(key)
    if vaAcc != None:
        plt.title(
            args.backbone + '/tsne -- circles: base centroids (c-' + str(n_centroids) + '/64-way) -- [vaAcc: ' + str(
                round(vaAcc, 2)) + ']')
    else:
        plt.title(args.backbone + '/tsne-- circles: base centroids (c-' + str(n_centroids) + '/64-way).')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    
    for n in range(args.n_base_class):
        cent_surv_list_i = cent_surv_list[n]
        z_2D_ = z_2D[y == n][cent_surv_list_i]
        plt.scatter(z_2D_[:, 0], z_2D_[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

        ### ############################
    plt.savefig('./results/tsne/' + fileName + '_' + str(i) + '.pdf')
    plt.close('all')
    
    
def centrid_labeling_supervised(net, centroids, x, y, T, n_centroids, device):
    net.eval()
    norm_z, norm_c = centroids.forward(net(x))
    scores = euclidean_dist(norm_z, norm_c)  # [64, 320]
    ye = torch.zeros_like(y)
    for s in range(scores.shape[0]):
        low_ = y[s] * n_centroids
        dist_s = 1-(scores[s, low_:low_ + n_centroids].detach().cpu().numpy())  ## 1- to say that minimum distance
        P = softmax(dist_s/T)
        soft_indx = np.random.choice(range(n_centroids), 1, p=P)[0] 
        ye[s] = soft_indx + low_ 
    net.train()
    return Variable(ye).to(device)

 
def centrid_labeling_supervised_min(net, centroids, x, y, n_centroids, device): 
    net.eval()
    norm_z, norm_c = centroids.forward(net(x))
    scores = euclidean_dist(norm_z, norm_c)  # [64, 320]
    ye = torch.zeros_like(y)
    for s in range(scores.shape[0]):
        low_ = y[s] * n_centroids
        dist_s = (scores[s, low_:low_ + n_centroids].detach().cpu().numpy()) 
        ye[s] = np.argmin(dist_s) + low_ 
    net.train()
    return Variable(ye).to(device)    
    
    
def cooling(Temperature):
    if Temperature > 0.0005:
        Temperature -= temperature_step  ##
    if Temperature < 0:
        Temperature = 0.0005
    return Temperature


def heating(Temperature):
    if Temperature < 1:
        Temperature += 0.05
    if Temperature > 1:
        Temperature = 1
    return Temperature    
from sklearn import svm
from sklearn.neighbors import NearestCentroid 
from sklearn.linear_model import LogisticRegression

def no_fineTuining(args, net, dataLoader, device, n_ft=250):
    Acc_LoGReg, Acc_cnn = [], []
    
    clf_cnn = NearestCentroid() 
#    knn = KNeighborsClassifier(n_neighbors=5) 
    
    net.eval()
    for i, (x, _) in enumerate(dataLoader):
        x = Variable(x).to(device)
        x_support = x[:, :args.n_shot, :, :, :].contiguous()
        x_support = x_support.view(args.test_n_way * args.n_shot, *x.size()[2:])
        y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))
        y_support = Variable(y_support.to(device))

        with torch.no_grad():
            z_train = F.normalize(net(x_support))
            y_train = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))

            n_query = x.size(1) - args.n_shot
            x_query = x[:, args.n_shot:, :, :, :].contiguous()
            x_query = x_query.view(args.test_n_way * n_query, *x.size()[2:])

            z_train = z_train.cpu().numpy()
            y_train = y_train.numpy()
 
            z_test = F.normalize(net(x_query))
            y_test = torch.from_numpy(np.repeat(range(args.test_n_way), n_query))
            z_test = z_test.cpu().numpy()
            y_test = y_test.numpy()
            
             
            clf_cnn.fit(z_train, y_train)
            y_pred = clf_cnn.predict(z_test)
            Acc_cnn.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            
#            clf_logReg = LogisticRegression(random_state=0).fit(z_train, y_train)  
#            y_pred = clf_logReg.predict(z_test)                                     #clf_svm.predict(z_test) 
#            Acc_cnn.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            
#            knn.fit(z_train, y_train)
#            Acc_knn.append(knn.score(z_test, y_test))
#            if i%30==0:
#                print(np.mean(Acc_cnn), np.mean(Acc_knn))

    teAcc1 = np.mean(Acc_cnn)
#    teAcc2 = np.mean(Acc_LoGReg)
    acc_std = np.std(Acc_cnn)
    conf_interval = 1.96 * acc_std / np.sqrt(len(dataLoader))
    return teAcc1, conf_interval



