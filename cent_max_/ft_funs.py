

from __future__ import print_function 
from data.tl_dataFunctions import ar_base_DataLaoder  
from data.ml_dataFunctions import SetDataManager   
from backbones.utils import device_kwargs
from backbones.utils import backboneSet 
from backbones.utils import clear_temp
from torch.autograd import Variable
from args_parser import args_parser
import torch.nn.functional as F 
from torch.optim import  Adam 
from torch import nn 
import numpy as np  
import torch
import os 
from scipy.special import softmax 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors 


from torchvision.utils import save_image
from cent_max_.cent_max import euclidean_dist 





from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors 
from sklearn import metrics
from sklearn.svm import LinearSVC 
from sklearn.pipeline import Pipeline
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import svm


def centrid_labeling_supervised(net, centroids, x, y, T, n_centroids, device):
    net.eval()
    norm_z, norm_c = centroids.forward(net(x))
    scores = -euclidean_dist(norm_z, norm_c)  # [64, 320]
    ye = torch.zeros_like(y)
    for s in range(scores.shape[0]):
        low_ = y[s] * n_centroids
        dist_s = scores[s, low_:low_ + n_centroids].detach().cpu().numpy()  ## 1- to say that minimum distance
        P = softmax(dist_s/T)
        soft_indx = np.random.choice(range(n_centroids), 1, p=P)[0] 
        ye[s] = soft_indx + low_ 
    net.train()
    return Variable(ye).to(device)

def centrid_labeling_supervised_min(net, centroids, cent_died_list, x, y, n_centroids, device): 
    norm_z, norm_c = centroids.forward(net(x))
    scores = euclidean_dist(norm_z, norm_c)  # [64, 320]
    ye = torch.zeros_like(y)
    for s in range(len(cent_died_list)):
        low_ = y[s] * n_centroids
        dist_s = (scores[s, low_:low_ + n_centroids].detach().cpu().numpy()) 
        cent_died_list_s = cent_died_list[s] 
        dist_s[cent_died_list_s] = np.inf 
        ye[s] = np.argmin(dist_s) + low_ 
    return Variable(ye).to(device)  

def fine_tunining(args, net, dataLoader, device, n_ft=250): 
    Acc1, Acc2 = [], []  
    batch_size_ = 7 
    for i, (x,_) in enumerate(dataLoader):  
        x = Variable(x).to(device)
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'test_centroids', file_name))
        te_centroids = checkpoint['te_centroids']    
        
        x_support   = x[:, :args.n_shot, :,:,:].contiguous()
        x_support = x_support.view(args.test_n_way * args.n_shot, *x.size()[2:]) 
        y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))  
        y_support = Variable(y_support.to(device)) 
        
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'test_centroids', file_name))
        te_centroids = checkpoint['te_centroids'] 
        te_optimizer = Adam(te_centroids.parameters(), lr = args.lr)   
        net.eval()
        with torch.no_grad(): 
            z_support = net(x_support)  
        y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))  
        y_support = Variable(y_support.to(device))   
        te_optimizer = Adam(te_centroids.parameters(), lr=args.lr) 
        for iter in range(n_ft):   
            te_optimizer.zero_grad()  
            loss_ = teloss(te_centroids, z_support, y_support)  
            loss_.backward()
            te_optimizer.step()    
        Acc1.append(query_test(args, net.eval(), te_centroids, x))    
    teAcc1 = np.mean(Acc1) 
    acc_std  = np.std(Acc1)
    conf_interval = 1.96* acc_std/np.sqrt(len(dataLoader))
    return teAcc1, conf_interval



def query_test(args, net, centroids, x):
    n_query = x.size(1) - args.n_shot 
    x_query = x[:, args.n_shot:, :,:,:].contiguous()
    x_query = x_query.view(args.test_n_way * n_query, *x.size()[2:]) 
    y_query = torch.from_numpy(np.repeat(range(args.test_n_way), n_query))
    y_query = Variable(y_query.cuda())   
    norm_z, norm_c = centroids.forward(net(x_query))  
    
    dists = euclidean_dist(norm_z, norm_c) 
    
    
    y_hat = np.argmin(dists.data.cpu().numpy(), axis=1)
    return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int))*100





#y_proto = torch.from_numpy(np.repeat(range(n_class), n_centroids))

 
criterion = nn.CrossEntropyLoss()
#criterion2 = nn.MarginRankingLoss()



def teloss(centroids, z, y):
    norm_z, norm_c = centroids.forward(z) 
    
    logits = centroids.s * -euclidean_dist(norm_z, norm_c)
    commitment_loss = criterion(logits, y) 
    
    
    
#    logits = centroids.s * -euclidean_dist(norm_c, norm_c)
    embedding_loss = criterion(norm_c[y], y) 
     
    
    return (commitment_loss + embedding_loss)/2


#commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
#embedding_loss = F.mse_loss(quantized_latents, latents.detach())
 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn import svm
from sklearn.neighbors import NearestCentroid 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB













def cnn_gnb_lrg_RF_svm(args, f_net, dataLoader, device, n_ft=250):
    Acc_LoGReg, Acc_cnn, Acc_RF, Acc_knn, GNB = [], [], [], [], [] 
    clf_cnn = NearestCentroid() 
#    knn = KNeighborsClassifier(n_neighbors=2) 
    svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    RFclf = RandomForestClassifier(max_depth=9, random_state=0)
    gnb = GaussianNB()
    f_net.eval()
    for i, (x, _) in enumerate(dataLoader):
        x = Variable(x).to(device)
        x_support = x[:, :args.n_shot, :, :, :].contiguous()
        x_support = x_support.view(args.test_n_way * args.n_shot, *x.size()[2:])
        y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))
        y_support = Variable(y_support.to(device))

        with torch.no_grad():
            z_train = F.normalize(f_net(x_support))
#            z_train = f_net(x_support)
            y_train = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))

            n_query = x.size(1) - args.n_shot
            x_query = x[:, args.n_shot:, :, :, :].contiguous()
            x_query = x_query.view(args.test_n_way * n_query, *x.size()[2:])

            z_train = z_train.cpu().numpy()
            y_train = y_train.numpy()
 
            z_test = F.normalize(f_net(x_query))
#            z_test = f_net(x_query)
            y_test = torch.from_numpy(np.repeat(range(args.test_n_way), n_query))
            z_test = z_test.cpu().numpy()
            y_test = y_test.numpy()
            
             
            clf_cnn.fit(z_train, y_train)
            y_pred = clf_cnn.predict(z_test)
            Acc_cnn.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            
            clf_logReg = LogisticRegression(random_state=0).fit(z_train, y_train)  
            y_pred = clf_logReg.predict(z_test)                                     #clf_svm.predict(z_test) 
            Acc_LoGReg.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            
            RFclf.fit(z_train, y_train)
            y_pred = RFclf.predict(z_test)
            Acc_RF.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            
            
            svm_clf.fit(z_train, y_train)
            y_pred = svm_clf.predict(z_test)
            Acc_knn.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
#            knn.fit(z_train, y_train)
#            Acc_knn.append(knn.score(z_test, y_test)*100)
            
            
            y_pred = gnb.fit(z_train, y_train).predict(z_test)
            GNB.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
             
#            if i%30==0:
#                print(np.mean(Acc_cnn), np.mean(Acc_knn))

    teAcc0 = np.mean(Acc_cnn)
    teAcc1 = np.mean(GNB)
    teAcc2 = np.mean(Acc_LoGReg)
    teAcc3 = np.mean(Acc_RF)
    teAcc4 = np.mean(Acc_knn)
    
    
#    acc_std = np.std(Acc_cnn)
#    conf_interval = 1.96 * acc_std / np.sqrt(len(dataLoader))
    
    
    
    return [teAcc0, teAcc1, teAcc2, teAcc3, teAcc4]





def cent_nn(args, f_net, dataLoader, device, normalization = True): 
    Acc_cnn, Acc_knn = [], []
    clf_cnn = NearestCentroid()  
    # clf_knn = KNeighborsClassifier(weights='distance', p=2, metric='euclidean', n_neighbors=3) 
    svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    f_net.eval()
    for i, (x, _) in enumerate(dataLoader):
        x = Variable(x).to(device)
        x_support = x[:, :args.n_shot, :, :, :].contiguous()
        x_support = x_support.view(args.test_n_way * args.n_shot, *x.size()[2:])
        y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))
        y_support = Variable(y_support.to(device))

        with torch.no_grad():
            if normalization:
                z_train = F.normalize(f_net(x_support))
            else: 
                z_train = f_net(x_support)
            y_train = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))

            n_query = x.size(1) - args.n_shot
            x_query = x[:, args.n_shot:, :, :, :].contiguous()
            x_query = x_query.view(args.test_n_way * n_query, *x.size()[2:])

            z_train = z_train.cpu().numpy()
            y_train = y_train.numpy()
 
            if normalization:
                z_test = F.normalize(f_net(x_query))
            else:
                z_test = f_net(x_query)
            y_test = torch.from_numpy(np.repeat(range(args.test_n_way), n_query))
            z_test = z_test.cpu().numpy()
            y_test = y_test.numpy()
            
             
            clf_cnn.fit(z_train, y_train)
            y_pred = clf_cnn.predict(z_test)
            Acc_cnn.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            
            # svm_clf.fit(z_train, y_train)
            # y_pred = svm_clf.predict(z_test)
            # Acc_knn.append(np.mean((y_pred == y_test).astype(int)) * 100)
            
            # clf_knn.fit(z_train, y_train)
            # y_pred = clf_knn.predict(z_test)
            # Acc_knn.append(np.mean((y_pred == y_test).astype(int)) * 100) 
            # print(Acc_cnn[-1], Acc_knn[-1]) 
            
    acc_std = np.std(Acc_cnn)
    conf_interval = 1.96 * acc_std / np.sqrt(len(dataLoader)) 
    return np.mean(Acc_cnn),  conf_interval




def innner_centnn(args, data, f_net, device):  
    clf_cnn = NearestCentroid(metric='euclidean')   
    # if args.cenType=='euc':
        # clf_cnn = NearestCentroid(metric='euclidean')   
    # else:
    #     clf_cnn = NearestCentroid(metric='cosine')  
    [x_support, y_support, x_query, y_query] = data 
    f_net.eval() 
    with torch.no_grad():
        z_train = F.normalize(f_net(x_support))  
        z_train = z_train.cpu().numpy() 
        z_test = F.normalize(f_net(x_query))
        z_test = z_test.cpu().numpy() 
        clf_cnn.fit(z_train, y_support.cpu().numpy())
        y_pred = clf_cnn.predict(z_test)
        return np.mean((y_pred == y_query.cpu().numpy()).astype(int)) * 100

            
    

