from __future__ import print_function
# from data.tl_dataFunctions import ar_base_DataLaoder
from data.ml_dataFunctions import SetDataManager #, SetDataManager_base
import copy 
from cent_max_.ft_funs import  cnn_gnb_lrg_RF_svm, cent_nn  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F

def data_fun(args, x, device): 
    x_support   = x[:, :args.n_shot, :,:,:].contiguous()
    x_support = x_support.view(args.test_n_way * args.n_shot, *x.size()[2:]) 
    y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))
    y_support = Variable(y_support.to(device))
    x_query = x[:, args.n_shot:, :,:,:].contiguous()
    x_query = x_query.view(args.test_n_way * args.n_query, *x.size()[2:]) 
    y_query = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_query))
    y_query = Variable(y_query.cuda()) 
    
    x_support = x_support.to(device)
    y_support = y_support.to(device)
    x_query = x_query.to(device)
    y_query = y_query.to(device) 
    return [x_support, y_support, x_query, y_query]
 
    
def innter_acc_fun(args, net, x, device):
    net.eval()
    with torch.no_grad():  
        [x_support, y_support, x_query, y_query] = data_fun(args, x, device) 
        z_proto = net(x_support.to(device)).view(args.n_way, args.n_shot, -1).mean(1) 
        norm_w = F.normalize(z_proto)  
        norm_z = F.normalize(net(x_query.to(device))) 
        if args.cenType == 'cos':
            logits = F.linear(norm_z, norm_w)  
        else:
            logits = -euclidean_dist(norm_z, norm_w)  
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)  
        return np.mean((y_hat == y_query.cpu().numpy()).astype(int))*100
    
     
def accuracy_fun_tl(args, data_loader, net, device):        
    AccL = []
    net.eval()   
    for x, y in data_loader:
        AccL.append(innter_acc_fun(args, net, x, device)) 
    print_txt = 'ep.%d    %4.2f/%4.2f    %4.2f  %4.2f   T:%4.2f ' 
    teAcc = np.mean(AccL) 
    acc_std  = np.std(AccL) 
    conf_interval = 1.96* acc_std/np.sqrt(len(data_loader))
    net.train()   
    return print_txt, teAcc, conf_interval




def tsne_viz_centroids_freq(args, tr_centroids, frequency_y, n_centroids, epoch=0, vaAcc=None, fileName='EM_centViz_', saveform='.pdf'):
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
        frequency_y_ = frequency_y[y==n]
        z_2D_ = z_2D[y == n][frequency_y_>2]
        plt.scatter(z_2D_[:, 0], z_2D_[:, 1], cmap='viridis', marker="o", s=200, color=colors[n], alpha=1,
                    edgecolors='black')

        ### ############################
    plt.savefig('/home/arafr1/scratch/few_shot_lablatory/results/tsne/' + fileName + '_' + str(epoch) + saveform)
    plt.close('all') 
     
    plt.figure(figsize=(16, 16))   
    # major_ticks = np.arange(0, 1, args.n_base_class)
    # plt.set_xticks(major_ticks) 
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.7) 
    x = 0
    for c in range(args.n_base_class):
        frequency_y_ = frequency_y[y==c]
        color_i = colors[c]
        for n in range(n_centroids):
            plt.scatter(x, frequency_y_[n], cmap='viridis', marker="o", s=200, color=color_i, alpha=0.7,
                        edgecolors='black') 
        x += 1
    
    plt.ylabel('Components assigned times in one epoch over the dataset')
    plt.xlabel('Reserved pre-class mixture components (codebook)') 
    plt.savefig('/home/arafr1/scratch/few_shot_lablatory/results/tsne/freq' + str(epoch) + '_suv'+saveform)
    plt.savefig('/home/arafr1/scratch/few_shot_lablatory/results/tsne/freq' + str(epoch) + '_suv.png')
    plt.close('all')  


def fun_metaLoader(args, n_eposide=400, file_name = '/novel.json'): ##   val
    val_file = args.benchmarks_dir + args.dataset + file_name  
    val_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_eposide) 
    return val_datamgr.get_data_loader(val_file, aug=False)

# def fun_metaLoader_base(args, n_eposide, n_way = 64, n_shot = 1, n_query = 1, file_name='/base.json', aug=True): ##   val
#     val_file = args.benchmarks_dir + args.dataset + file_name   
#     tr_datamgr = SetDataManager_base(args.img_size, n_way, n_shot, n_query, n_eposide=n_eposide)
#     return tr_datamgr.get_data_loader(val_file, aug=aug)

 
def prior_update(source, prior):
    prior.load_state_dict(copy.deepcopy(source.state_dict()))   
    prior.eval()   
    return prior


def clf_meta_fun(args, f_net, vaLoader, device, multi_clsssifier):       
    if multi_clsssifier:
        vaAcc_list = cnn_gnb_lrg_RF_svm(args, f_net , vaLoader, device)   
        vaAcc = max(vaAcc_list)   
        print_txt = 'epo %d trL: %4.2f || vaAcc[cnn|gnb|lrg|RF|svm]: || %4.2f%%|%4.2f%%|%4.2f%%|%4.2f%%|%4.2f%% || '
    else:
        vaAcc, vaVar = cent_nn(args, f_net, vaLoader, device)   
        print_txt = 'ep.%d    %4.2f/%4.2f    %4.2f  %4.2f   T:%4.2f '
        vaAcc_list = None 
    return print_txt, vaAcc_list, vaAcc, vaVar 



def centroids_black_list(frequency_y, n_class, deing_th, n_centroids):
    cent_died_list = []
    cent_surv_list = []
    frequency_y_up = []
    num_surv = 0
    k = 0     
    for c in range(n_class):
        curr_c_die_list = []
        curr_c_suv_list = []
        for i in range(n_centroids):  
            if frequency_y[k]<deing_th:
                curr_c_die_list.append(i)
            else:
                num_surv += 1
                frequency_y_up.append(frequency_y[k])  
                curr_c_suv_list.append(i) 
            k += 1   
        cent_died_list.append(curr_c_die_list) 
        cent_surv_list.append(curr_c_suv_list)
        
    return cent_died_list, cent_surv_list, frequency_y_up, num_surv



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







from torch.optim.lr_scheduler import _LRScheduler
#https://github.com/bl0/negative-margin.few-shot/blob/master/lib/utils.py
# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)

 