import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from sklearn.metrics import matthews_corrcoef
from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from datasets.mutag_dataset import Mutag
from models.gnn_ib import GIB
from models.ciga import GNNERM, GNNEnv, CIGA, GSAT, GNNPooling, GALA
from models.groupdro import GroupDRO
from models.losses import GeneralizedCELoss, get_contrast_loss, get_irm_loss
from utils.logger import Logger
from utils.util import args_print, set_seed, split_into_groups, extract_k_hop_neighbor, node_id_list_to_mask, meanDff
from kmeans_pytorch import kmeans, kmeans_predict
from utils.helper import visualize_a_graph, get_viz_idx, plot_and_save_subgraph
import networkx as nx
import torch.optim.lr_scheduler as lr_scheduler
import time
import csv

@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, embeddings = model(batch, return_data="rep")
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    final_report = {}
    unique_classes = np.unique(y_true)
    for class_label in unique_classes:
        class_mask = (y_true == class_label)
        if len(class_mask) == 0:
            class_acc = np.nan
        else:
            class_acc = np.mean(y_true[class_mask] == y_pred[class_mask])
        final_report[class_label] = class_acc

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred, final_report
    else:
        return res_metric, final_report

@torch.no_grad()
def compute_embedding_distance(model, device, loader):
    model.eval()
    dist = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                # pred = model(batch)
                pred, casual_node_id, h = model(batch, return_data='dist')
                num_nodes = batch.num_nodes
                node_counts_per_graph = torch.bincount(batch.batch)

                start_idx = 0
                batch_embedding_dist = []
                for count in node_counts_per_graph:
                    end_idx = start_idx + count.item()
                    edge_mask = (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx) & (batch.edge_index[1] >= start_idx) & (batch.edge_index[1] < end_idx)
                    current_edge_index = batch.edge_index[:, edge_mask]
                    current_casual_node_id = [id for id in casual_node_id if id in torch.unique(current_edge_index).tolist()]
                    current_casual_node_mask = node_id_list_to_mask([id for id in current_casual_node_id], num_nodes)

                    G = nx.Graph()
                    for src, dst in zip(current_edge_index[0], current_edge_index[1]):
                        G.add_edge(src.item(), dst.item())
                        
                    # distance between casual nodes and different hops' spurious nodes
                    inv_size = 4  # 3 layers
                    subgroup_embedding_dist = []
                    neighbor_list = extract_k_hop_neighbor(G, current_casual_node_id, inv_size)
                    for i in range(inv_size):
                        if(len(neighbor_list[i])) != 0:
                            group_node = neighbor_list[i]
                            group_test_mask = node_id_list_to_mask(group_node, num_nodes)
                            subgroup_embedding = h[group_test_mask]
                            embeddings_dist = meanDff(subgroup_embedding.to(device), h[current_casual_node_mask].to(device))
                            subgroup_embedding_dist.append(embeddings_dist.item())                               
                    start_idx = end_idx    
                    batch_embedding_dist.append(subgroup_embedding_dist)
                    
                # batch average
                max_length = max(len(lst) for lst in batch_embedding_dist)
                padded_batch_embedding_dist = [lst + [float('nan')] * (max_length - len(lst)) for lst in batch_embedding_dist]
                padded_tensor = torch.tensor(padded_batch_embedding_dist, dtype=torch.float)
                # 计算每列的平均值，忽略NaN
                batch_dist_averages = torch.nanmean(padded_tensor, dim=0)
                # batch_dist_averages = [sum(column) / len(batch_embedding_dist) for column in zip(*batch_embedding_dist)]       
                print("Batch Embedding Distance Averages:", batch_dist_averages)            
        dist.append(batch_dist_averages)
    dataset_dist_averages = [sum(column) / len(dist) for column in zip(*dist)]       
    print("Dataset Embedding Distance Averages:", dataset_dist_averages)


def main():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning')
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
    # parser.add_argument('--dataset', default='drugood_lbap_core_ic50_assay', type=str)
    parser.add_argument('--dataset', default='drugood_lbap_core_ec50_assay', type=str)
    parser.add_argument('--bias', default='0.2', type=str, help='select bias extend')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')

    # training config
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-e','--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')
    parser.add_argument('--seed', nargs='?', default='[1,2,3,4,5]', help='random seed')
    parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')
    parser.add_argument('-pe','--pretrain_epochs', default=20, type=int, help='pretrain epoch before early stopping')
    parser.add_argument('-pe2','--pretrain_epochs_s2', default=20, type=int, help='pretrain epoch before early stopping')
    
    # model config
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--r', default=0.25, type=float, help='selected ratio')
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='mean', type=str)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--early_stopping', default=5, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric', default='', type=str,
                        help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # Invariant Learning baselines config
    parser.add_argument('--num_envs', default=2, type=int, help='num of envs need to be partitioned')
    parser.add_argument('--env_dis', default='cosine', type=str, help='distance metric used for clustering')
    parser.add_argument('-ep','--eiil_pretrain_iters', default=20, type=int, help='')
    parser.add_argument('--env_prior', default='uniform',type=str, help='')
    parser.add_argument('--eiil', action='store_true')
    parser.add_argument('--irm_p', default=0, type=float, help='penalty weight')
    parser.add_argument('--irm_opt', default='irm', type=str, help='algorithms to use')
   

    # Invariant Graph Learning config
    parser.add_argument('--erm', default=1)  # whether to use normal GNN arch
    parser.add_argument('--groupdro', action='store_true')  # whether to use normal GNN arch
    parser.add_argument('--ginv_opt', default='ginv', type=str)  # which interpretable GNN archs to use
    parser.add_argument('--dir', default=0, type=float)
    parser.add_argument('--grea', default=0, type=float)
    parser.add_argument('--cal', default=0, type=float)
    parser.add_argument('--gil', default=0, type=float)
    parser.add_argument('--disc', default=0, type=float)
    parser.add_argument('--disc_q', default=0.7, type=float)
    parser.add_argument('--contrast_t', default=1.0, type=float, help='temperature prameter in contrast loss')
    # strength of the contrastive reg, \alpha in the paper
    parser.add_argument('-c','--contrast', default=0, type=float)    
    parser.add_argument('--not_norm', action='store_true')  # whether not using normalization for the constrast loss
    parser.add_argument('-c_sam', '--contrast_sampling', default='mul', type=str)
    # contrasting summary from the classifier or featurizer
    # rep:  classifier rep
    # feat: featurizer rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-c_rep', '--contrast_rep', default='rep', type=str)
    # pooling method for the last two options in c_rep
    parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)


    # spurious rep for maximizing I(G_S;Y)
    # rep:  classifier rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)
    # strength of the hinge reg, \beta in the paper
    parser.add_argument('--spu_coe', default=0, type=float) 

    # misc
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('-vt','--visualize_top',default=-1, type=int)
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--save_model', action='store_true')  # save to ./erm_model 
    parser.add_argument('-ea','--save_env_as', action='store_true') 
    parser.add_argument('-nm','--not_mix', action='store_true')  
    parser.add_argument('-mp','--mix_repeats',default=-1,type=int ) 
    parser.add_argument('--DRO', default=True, help='Enable DRO (Default: False)')
    parser.add_argument('--Q', default=True, help='Enable DRO with learnable parameters (Default: False)')
    parser.add_argument('--DRO_method', default=None, help='DRO_method (None, cvar, chisq, cvar_doro, chisq_doro, cvar_group)')
    parser.add_argument('--Q_update', default='distance', help='Q_update (None, variant, group, gradient)')
    parser.add_argument('--D_k', default=2)
    parser.add_argument('--file_name', default='experiment_class6_noise0.15.csv', type=str)
    parser.add_argument('--Distance_loss',default=0.0, type=float)
    parser.add_argument('--Lambda',default=1e-1, type=float)
    parser.add_argument('--Miu',default=1e-1, type=float)
    parser.add_argument('--P_lr',default=3e-5, type=float)

    

    args = parser.parse_args()

    erm_model = None  # used to obtain pesudo labels for sampling in contrastive loss

    args.seed = eval(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)

    criterion = ce_loss
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1
    def group_q_update(q, losses, eta):
        losses = losses.to(q.device)  # Ensure losses are on the same device as q
        q = q * (eta * losses).exp()
        return q
    def variant_q_update(q, pred, labels, num_classes, losses, lr=1e-3, lambda_reg=1e-5):
        prior = torch.zeros(num_classes, dtype=torch.float32)
        for m in range(num_classes):
            mask = (labels == m)
            # compute consistency
            distances = torch.norm(pred[mask], dim=1)
            # var represents consistency
            variance = torch.mean(torch.var(distances, dim=0)).to('cpu')
            variance = torch.round(variance * 1000) / 1000
            prior[m] = variance
        prior = (prior/prior.sum()).to(q.device)
        q = lr*(losses - lambda_reg*(q - prior))
        q = q/q.sum()
        return q
    def gradient_q_update(q, losses, lr=1e-2):
        losses = torch.round(losses * 1000) / 1000
        q = q + lr * losses
        q = q/q.sum()
        return q
    def distance_q_update(q, losses, graph_x, labels, num_classes, D_k=1, lr=1e-3, lamuda=1e-5,  eta=1e-3, miu=1e-2):
        losses = torch.round(losses * 1000) / 1000
        value_array = torch.mean(graph_x, dim=1)
        new_q = q.clone()
        
        for i in range(num_classes):
            maski = (labels == i).cpu().numpy()
            masked_values_i = value_array[maski]
            p_i = torch.mean(masked_values_i, dim=0).item()
            if len(masked_values_i) != 0 and p_i != 0:
                sum_pj = 0
                for j in range(num_classes):
                    if j != i:
                        maskj = (labels == j).cpu().numpy()
                        masked_values_j = value_array[maskj]
                        sum_pj = sum_pj + q[j]*torch.mean(masked_values_j, dim=0).item()
                if sum_pj != 0:
                    if D_k == 1:
                        new_q[i] = q[i].detach() + ((losses[i].item()/miu*p_i*p_i + sum_pj/p_i)/(eta + (num_classes - 1)))*lr
                    else:
                        new_q[i] = q[i].detach() + ((losses[i]-lamuda+miu*p_i*sum_pj)/(eta +((num_classes-1)*miu*p_i*p_i+lamuda)))*lr
                    
                    if new_q[i] < 0:
                        new_q[i] = 1e-6
            
            
        new_q = new_q/new_q.sum()
        return new_q
    """
    DROLOSS
    """
    def calculate_Q_DROloss(pred, graph_x, labels, criterion, q, num_classes, update_methods='variant', eta=1e-3, lambda_reg=1e-1, miu=1e-1, lr=1e-5, D_k=2, loss_distance=0.0):
        # unique_sorted_tensor = torch.unique(labels)
        # label_list = unique_sorted_tensor.sort()[0].tolist()
        n = 0
        losses = torch.ones(num_classes, dtype=torch.float32).to(q.device)
        new_q = q.clone()
        if loss_distance == 0.0:
            for i in range(num_classes):
                mask = (labels == i)
                losses[i] = criterion(pred[mask], labels[mask])
                if torch.isnan(losses[i]).any():
                    losses[i] = 0.0
        else:
            loss_vector = compute_weighted_loss_with_neighborhood(pred=pred, criterion=criterion, graph_x=graph_x, labels=labels, num_classes=num_classes, k=loss_distance)
            for i in range(num_classes):
                mask = (labels == i)
                if mask.any():
                    losses[i] = loss_vector[mask].mean()
                else:
                    losses[i] = 0.0
                
        losses = q.detach() * losses
        weighted_loss = torch.sum(losses)
        if update_methods == 'variant':
            new_q = variant_q_update(q, pred, labels, num_classes, losses, lr, lambda_reg)
        elif update_methods == 'group':
            new_q = group_q_update(q, losses, lr)
        elif update_methods == 'gradient':
            new_q = gradient_q_update(q, losses, lr)
        elif update_methods == 'distance':
            new_q = distance_q_update(q, losses, graph_x, labels, num_classes, D_k, lr, lambda_reg, eta, miu)
            embedding = torch.mean(graph_x, dim=1, keepdim=True)
            group_means = torch.zeros(num_classes, embedding.size(1))
            default_value = torch.zeros(embedding.size(1), device=embedding.device)
            for i in range(num_classes):
                current_class_samples = embedding[labels == i]
                if current_class_samples.size(0)>0:
                    group_means[i] = current_class_samples.mean(dim=0)
                else:
                    group_means[i] = default_value
            diffs = group_means.unsqueeze(1) - group_means.unsqueeze(0)  # (num_classes, num_classes, 1)
            diffs = torch.clamp(diffs, min=-1e6, max=1e6)
            squared_diffs = torch.pow(diffs, 2)  # (num_classes, num_classes, 1)
            squared_diffs_sum = squared_diffs.sum(dim=-1)  # (num_classes, num_classes)
            triu_mask = torch.triu(torch.ones_like(squared_diffs_sum), diagonal=1)  
            upper_triangle_sums = squared_diffs_sum * triu_mask  
            weighted_loss = weighted_loss + lambda_reg*upper_triangle_sums.sum()    
        if torch.isnan(new_q).any()==False:
            q = new_q.clone()
        
        return q, weighted_loss
    def calculate_DROloss(pred, labels, criterion, methods = 'cvar_group' , alpha=0.5, eps=1e-3):
        loss = 0.0
        if criterion == ce_loss:
            loss_vector = F.cross_entropy(pred, labels, reduction='none')
        elif criterion == mse_loss:
            loss_vector = F.mse_loss(pred, labels, reduction='none')
        elif criterion == bce_loss:
            loss_vector = F.binary_cross_entropy_with_logits(pred, labels, reduction='none')
        if methods == 'cvar':
            batch_size = pred.size(0)
            n = int(batch_size * alpha)
            rk = torch.argsort(loss_vector, descending=True)
            loss = loss_vector[rk[:n]].mean()
        elif methods == 'chisq':
            max_l = 10.
            C = math.sqrt(1 + (1 / alpha - 1) ** 2)
            foo = lambda eta: C * math.sqrt((F.relu(loss_vector - eta) ** 2).mean().item()) + eta
            opt_eta = sopt.brent(foo, brack=(0, max_l))
            loss = C * torch.sqrt((F.relu(loss_vector - opt_eta) ** 2).mean()) + opt_eta
        elif methods == 'cvar_doro':
            gamma = eps + alpha * (1 - eps)
            batch_size = len(loss_vector)
            n1 = int(gamma * batch_size)
            n2 = int(eps * batch_size)
            rk = torch.argsort(loss_vector, descending=True)
            loss = loss_vector[rk[n2:n1]].sum() / alpha / (batch_size - n2)  
        elif methods == 'chisq_doro':
            max_l = 10.
            batch_size = len(loss_vector)
            C = math.sqrt(1 + (1 / alpha - 1) ** 2)
            n = int(eps * batch_size)
            rk = torch.argsort(loss_vector, descending=True)
            l0 = loss_vector[rk[n:]]
            foo = lambda eta: C * math.sqrt((F.relu(l0 - eta) ** 2).mean().item()) + eta
            opt_eta = sopt.brent(foo, brack=(0, max_l))
            loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
        elif methods == 'cvar_group':
            unique_sorted_tensor = torch.unique(labels)
            label_list = unique_sorted_tensor.sort()[0].tolist()
            num_class = len(label_list)
            divided_alpha = alpha / num_class
            loss = 0.0
            for i in label_list:
                mask = (labels == i)
                n = int(divided_alpha * mask.size(0))
                rk = torch.argsort(loss_vector[mask], descending=True)
                loss += loss_vector[mask][rk[:n]].mean()
            loss = loss / num_class      
        return loss
    
    def compute_weighted_loss_with_neighborhood(pred, criterion, graph_x, labels, num_classes, k):
        if criterion == ce_loss:
            loss_vector = F.cross_entropy(pred, labels, reduction='none')
        elif criterion == mse_loss:
            loss_vector = F.mse_loss(pred, labels, reduction='none')
        elif criterion == bce_loss:
            loss_vector = F.binary_cross_entropy_with_logits(pred, labels, reduction='none')
        embedding = torch.mean(graph_x, dim=1, keepdim=True)
        lower_bound = embedding - k
        upper_bound = embedding + k
        in_range = (embedding >= lower_bound.T) & (embedding <= upper_bound.T)  # (batch_size, batch_size)
        same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (batch_size, batch_size)
        count_same_class_in_range = (in_range & same_class_mask).sum(dim=1)
        label_counts = torch.zeros(embedding.size(0), dtype=torch.float32, device=graph_x.device)
        for i in range(num_classes):
            label_mask = (labels == i)
            label_counts[label_mask] = label_mask.sum().float()
        label_counts[label_counts == 0] = 1
        weight = count_same_class_in_range/label_counts
        weighted_loss = (loss_vector*weight.detach())
        return weighted_loss
            
    
        
            

    ### automatic dataloading and splitting
    if args.dataset.lower().startswith('drugood'):
        #drugood_lbap_core_ic50_assay.json
        config_path = os.path.join("configs", args.dataset + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(args.root,"DrugOOD")
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
        val_dataset   = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
        test_dataset  = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset, mode="ood_test")
        if args.eval_metric == 'auc':
            evaluator = Evaluator('ogbg-molhiv')
            eval_metric = args.eval_metric = 'rocauc'
        else:
            evaluator = Evaluator('ogbg-ppa')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = 39
        edge_dim = 10
        num_classes = 2
    elif args.dataset.lower().startswith('ogbg'):

        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        if 'ppa' in args.dataset.lower():
            dataset = PygGraphPropPredDataset(root=args.root, name=args.dataset, transform=add_zeros)
            input_dim = -1
            num_classes = dataset.num_classes
        else:
            dataset = PygGraphPropPredDataset(root=args.root, name=args.dataset)
            input_dim = 1
            num_classes = dataset.num_tasks
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            print('using simple feature')
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
        split_idx = dataset.get_idx_split()
        ### automatic evaluator. takes dataset name as input
        evaluator = Evaluator(args.dataset)
        # evaluator = Evaluator('ogbg-ppa')

        train_loader = DataLoader(dataset[split_idx["train"]],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]],
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

        if 'classification' in dataset.task_type:

            def cls_loss(a, b, reduction='mean'):
                return F.binary_cross_entropy_with_logits(a.float(), b.float(), reduction=reduction)

            criterion = cls_loss
        else:

            def mse_loss(a, b, reduction='mean'):
                return F.mse_loss(a.float(), b.float(), reduction=reduction)

            criterion = mse_loss

        eval_metric = dataset.eval_metric
    elif args.dataset.lower() in ['spmotif', 'mspmotif', 'bspmotif', 'pspmotif', 'tspmotif']:
        train_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='train')
        val_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='val')
        test_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='test')
        args.dataset = f'{args.dataset}-{args.bias}'
        input_dim = 7
        num_classes = 6
        evaluator = Evaluator('ogbg-ppa')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset.lower() in ['graph-sst5','graph-sst2']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader,train_dataset,val_dataset,test_dataset\
             = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=True, seed=args.seed,return_set=True)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['graph-twitter']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader,train_dataset,val_dataset,test_dataset \
        = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=False, seed=args.seed,return_set=True)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['cmnist']:
        n_val_data = 5000
        train_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='train')
        test_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='test')
        perm_idx = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(0))
        test_val = test_dataset[perm_idx]
        val_dataset, test_dataset = test_val[:n_val_data], test_val[n_val_data:]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = 7
        num_classes = 6
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['mutag', 'proteins', 'dd', 'nci1', 'nci109']:
        args.root = "data"
        if args.dataset.lower() == 'mutag':
            dataset = Mutag(osp.join(args.root, "TU","Mutagenicity"))
            perm_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
            train_idx = perm_idx[:len(dataset) * 8 // 10].numpy()
            val_idx = perm_idx[len(dataset) * 8 // 10:len(dataset) * 9 // 10].numpy()
            test_idx = perm_idx[len(dataset) * 8 // 10:].numpy()
            # train_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'train_idx.txt'), dtype=np.int64)
            # val_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'val_idx.txt'), dtype=np.int64)
            # test_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'test_idx.txt'), dtype=np.int64)
            def cls_loss(a, b, reduction='mean'):
                return F.binary_cross_entropy_with_logits(a.float(), b.float(), reduction=reduction)

            criterion = cls_loss
        else:
            dataset = TUDataset(osp.join(args.root, "TU"), name=args.dataset.upper())
            train_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'train_idx.txt'), dtype=np.int64)
            val_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'val_idx.txt'), dtype=np.int64)
            test_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'test_idx.txt'), dtype=np.int64)

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = dataset[0].x.size(1)
        num_classes = dataset.num_classes
        evaluator = Evaluator('ogbg-ppa')
    else:
        raise Exception("Invalid dataset name")

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
        'time': [],
    }
    experiment_name = f'{args.dataset}-{args.bias}_{args.ginv_opt}_erm{args.erm}_dir{args.dir}_coes{args.contrast}-{args.spu_coe}_seed{args.seed}_'
    exp_dir = os.path.join('./logs/', experiment_name)
    # os.mkdir(exp_dir)
    logger = Logger.get_logger(filename=exp_dir + '/log.log')
    ResLogFile = os.path.join('logs', args.dataset) + args.commit + "_ResLog.txt"
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")

    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    best_weights = None
    def save_experiment_data(filename, experiment_data):
        file_exists = os.path.isfile(filename)
        fieldnames = experiment_data[0].keys() if experiment_data else []
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for data in experiment_data:
                writer.writerow(data)
    experiment_datas = []
    
    for seed in args.seed:
        erm_model = None
        set_seed(seed)
        if args.DRO and args.Q:
            new_q = torch.ones(num_classes, dtype=torch.float32)
            q = torch.ones(num_classes, dtype=torch.float32)  # 初始化 q
            q = q/num_classes  # 使每个类别的权重为 1/num_classes
            q = q.to(args.device)
        # models and optimizers
        if args.erm:
            args.ginv_opt = 'erm'
            model = GNNERM(input_dim=input_dim,
                           edge_dim=edge_dim,
                           out_dim=num_classes,
                           gnn_type=args.model,
                           num_layers=args.num_layers,
                           emb_dim=args.emb_dim,
                           drop_ratio=args.dropout,
                           graph_pooling=args.pooling,
                           virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.groupdro:
            args.ginv_opt = 'groupdro'
            model = GroupDRO(input_dim=input_dim,
                           edge_dim=edge_dim,
                           out_dim=num_classes,
                           gnn_type=args.model,
                           num_layers=args.num_layers,
                           emb_dim=args.emb_dim,
                           drop_ratio=args.dropout,
                           graph_pooling=args.pooling,
                           virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.eiil:
            args.ginv_opt = 'gnnenv'
            if args.eiil_pretrain_iters < 0:
                args.eiil_pretrain_iters = args.pretrain
                args.pretrain += args.eiil_pretrain_iters
                args.epoch += args.eiil_pretrain_iters
            print(f"Will pretrain {args.eiil_pretrain_iters} epochs for GNNEnv")    
            model = GNNEnv(input_dim=input_dim,
                           edge_dim=edge_dim,
                           out_dim=num_classes,
                           gnn_type=args.model,
                           num_layers=args.num_layers,
                           emb_dim=args.emb_dim,
                           drop_ratio=args.dropout,
                           graph_pooling=args.pooling,
                           virtual_node=args.virtual_node,
                           num_envs=args.num_envs,
                           prior=args.env_prior).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.ginv_opt.lower() in ['asap']:
            args.ginv_opt = 'gnnpooling'
            model = GNNPooling(pooling=args.ginv_opt,
                               ratio=args.r,
                               input_dim=input_dim,
                               edge_dim=edge_dim,
                               out_dim=num_classes,
                               gnn_type=args.model,
                               num_layers=args.num_layers,
                               emb_dim=args.emb_dim,
                               drop_ratio=args.dropout,
                               graph_pooling=args.pooling,
                               virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.ginv_opt.lower() == 'gib':
            model = GIB(ratio=args.r,
                        input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.ginv_opt.lower() == 'gsat':
            model = GSAT(ratio=args.r,
                         input_dim=input_dim,
                         edge_dim=edge_dim,
                         out_dim=num_classes,
                         gnn_type=args.model,
                         num_layers=args.num_layers,
                         emb_dim=args.emb_dim,
                         drop_ratio=args.dropout,
                         graph_pooling=args.pooling,
                         virtual_node=args.virtual_node,
                         c_dim=args.classifier_emb_dim,
                         c_in=args.classifier_input_feat,
                         c_rep=args.contrast_rep,
                         c_pool=args.contrast_pooling,
                         s_rep=args.spurious_rep,
                         pretrain=args.pretrain).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.ginv_opt.lower() == 'gala':
            model = GALA(ratio=args.r,
                         input_dim=input_dim,
                         edge_dim=edge_dim,
                         out_dim=num_classes,
                         gnn_type=args.model,
                         num_layers=args.num_layers,
                         emb_dim=args.emb_dim,
                         drop_ratio=args.dropout,
                         graph_pooling=args.pooling,
                         virtual_node=args.virtual_node,
                         c_dim=args.classifier_emb_dim,
                         c_in=args.classifier_input_feat,
                         c_rep=args.contrast_rep,
                         c_pool=args.contrast_pooling,
                         s_rep=args.spurious_rep).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        else:
            args.ginv_opt = "ciga"
            model = CIGA(ratio=args.r,
                         input_dim=input_dim,
                         edge_dim=edge_dim,
                         out_dim=num_classes,
                         gnn_type=args.model,
                         num_layers=args.num_layers,
                         emb_dim=args.emb_dim,
                         drop_ratio=args.dropout,
                         graph_pooling=args.pooling,
                         virtual_node=args.virtual_node,
                         c_dim=args.classifier_emb_dim,
                         c_in=args.classifier_input_feat,
                         c_rep=args.contrast_rep,
                         c_pool=args.contrast_pooling,
                         s_rep=args.spurious_rep).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        print(model)
        last_train_acc, last_test_acc, last_val_acc = 0, 0, 0
        train_re = []
        test_re = []
        val_re = []
        report_temporal = [{}, {}, {}]
        cnt = 0
        rep_info = []
        Q_info = []
        # generate environment partitions
        if args.num_envs > 1 and not args.eiil:
            env_idx = (torch.sigmoid(torch.randn(len(train_loader.dataset))) > 0.5).long()
            print(f"random env partitions: num env 0: {sum(env_idx==0)} num env 1: {sum(env_idx==1)}")

        if args.contrast > 0:
            model_path = os.path.join('erm_model', args.dataset) + args.commit + f"_seed{seed}.pt"
            if not os.path.exists(model_path):
                print("Playing without ERM model in the path ", model_path)
            else:
                if 'xgnn' in model_path:
                    if 'r1' in args.commit:
                        ea_ratio = 0.1 
                    elif 'r3' in args.commit:
                        ea_ratio = 0.3
                    else:
                        ea_ratio = args.r
                    if 'xgnn2' in model_path:
                        erm_model = GALA(ratio=ea_ratio,
                                        input_dim=input_dim,
                                        edge_dim=edge_dim,
                                        out_dim=num_classes,
                                        gnn_type=args.model,
                                        num_layers=args.num_layers,
                                        emb_dim=args.emb_dim,
                                        drop_ratio=args.dropout,
                                        graph_pooling=args.pooling,
                                        virtual_node=args.virtual_node,
                                        c_dim=args.classifier_emb_dim,
                                        c_in=args.classifier_input_feat,
                                        c_rep=args.contrast_rep,
                                        c_pool=args.contrast_pooling,
                                        s_rep=args.spurious_rep).to(device)
                    else:
                        erm_model = CIGA(ratio=ea_ratio,
                                        input_dim=input_dim,
                                        edge_dim=edge_dim,
                                        out_dim=num_classes,
                                        gnn_type=args.model,
                                        num_layers=args.num_layers,
                                        emb_dim=args.emb_dim,
                                        drop_ratio=args.dropout,
                                        graph_pooling=args.pooling,
                                        virtual_node=args.virtual_node,
                                        c_dim=args.classifier_emb_dim,
                                        c_in=args.classifier_input_feat,
                                        c_rep=args.contrast_rep,
                                        c_pool=args.contrast_pooling,
                                        s_rep=args.spurious_rep).to(device)
                else:
                    erm_model = GNNERM(input_dim=input_dim,
                                    edge_dim=edge_dim,
                                    out_dim=num_classes,
                                    gnn_type=args.model,
                                    num_layers=args.num_layers,
                                    emb_dim=args.emb_dim,
                                    drop_ratio=args.dropout,
                                    graph_pooling=args.pooling,
                                    virtual_node=args.virtual_node).to(device)
                print("Loading model from ", model_path)
                erm_model.load_state_dict(torch.load(model_path, map_location=device))
                erm_model.eval()
                ea_pos = []
                ea_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
                ea_train_loader.dataset.resume_samples()
                reps = []
                with torch.no_grad():
                    for graph in ea_train_loader:
                        graph.to(device)
                        pred, rep = erm_model(graph, return_data="rep")
                        # pred shape: (batch_size, num_classes), rep shape: (batch_size, emb_dim)
                        pred = pred.argmax(-1)
                        ea_pos.append((pred==graph.y).cpu())
                        reps += rep
                reps = torch.stack(reps)    # （training_samples, emb_dim）
                if args.num_envs > 1:
                    pred_envs, cluster_centers = kmeans(
                            X=reps, num_clusters=args.num_envs, distance=args.env_dis, device=device,
                            tqdm_flag=False, seed=seed
                        )
                ea_pos = torch.hstack(ea_pos).bool()
                print(f"EA model training acc: {ea_pos.sum()/ea_pos.size(0)}")
                
                if not args.not_mix:
                    ea_train_loader.dataset.rebalance_samples(ea_pos, args.mix_repeats)
                    ea_train_loader = DataLoader(ea_train_loader.dataset, batch_size=args.batch_size, shuffle=True)
                    ea_pos = []
                    with torch.no_grad():
                        for graph in ea_train_loader:
                            graph.to(device)
                            pred, rep = erm_model(graph,return_data="rep")
                            pred = pred.argmax(-1)
                            ea_pos.append((pred==graph.y).cpu())
                    ea_pos = torch.hstack(ea_pos).bool()
                    print(f"EA model training acc on rebalanced dataset: {ea_pos.sum()/ea_pos.size(0)}")
                    train_loader = DataLoader(ea_train_loader.dataset, batch_size=args.batch_size, shuffle=True)
                else:
                    ea_train_loader = None
        
        start_time = time.time()
        epoch_loss = []
        test_accs = []
        train_accs = []
        val_accs = []
        epoch_loss_val = []
        
        
        for epoch in range(args.epoch):
            all_loss, n_bw = 0, 0
            all_losses = {}
            contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0.
            spu_pred_loss = torch.zeros(1).to(device)
            model.train()
            torch.autograd.set_detect_anomaly(True)
            rep_info_tempo = {i:0 for i in range(num_classes)}
            num_batch = (len(train_loader.dataset) // args.batch_size) + int((len(train_loader.dataset) % args.batch_size) > 0)
            loss_info = []
            rep_dict = {i: 0 for i in range(num_classes)}
            Q_dict = {i: 0 for i in range(num_classes)}
            for step, graph in tqdm(enumerate(train_loader), total=num_batch, desc="Training", disable=args.no_tqdm):
                n_bw += 1
                graph.to(device)
                # ignore nan targets
                # https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
                is_labeled = graph.y == graph.y
                # assert hasattr(graph, 'train_edge_weight_gcn'), "train_edge_weight_gcn is not initialized!"
                if args.dir > 0:
                    # obtain DIR loss and integrate it into cls loss
                    dir_loss, causal_pred, spu_pred, causal_rep = model.get_dir_loss(graph,
                                                                                      graph.y,
                                                                                      criterion,
                                                                                      is_labeled=is_labeled,
                                                                                      return_data='rep')
                    if args.DRO and args.Q:
                        q, pred_loss = calculate_Q_DROloss(causal_pred[is_labeled], graph.x[is_labeled], graph.y[is_labeled], criterion, q, num_classes, args.Q_update)
                        q, spu_loss = calculate_Q_DROloss(spu_pred[is_labeled], graph.x[is_labeled], graph.y[is_labeled], criterion, q, num_classes, args.Q_update)
                    elif args.DRO and args.Q == False:
                        pred_loss = calculate_DROloss(causal_pred[is_labeled], graph.y[is_labeled], criterion, args.DRO_method)
                        spu_loss = calculate_DROloss(spu_pred[is_labeled], graph.y[is_labeled], criterion, args.DRO_method)
                    else:
                        spu_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled])
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                    pred_loss = pred_loss + spu_loss + args.dir * (epoch**1.6) * dir_loss
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    all_losses['dir'] = (all_losses.get('dir', 0) * (n_bw - 1) + dir_loss.item()) / n_bw
                    all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_loss.item()) / n_bw
                elif args.groupdro > 0:
                    causal_pred, causal_rep = model(graph, return_data="rep") 
                    pred_loss = model.get_groupdro_loss(graph, criterion, is_labeled=is_labeled, return_data="rep", return_spu=True)    
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                elif args.grea > 0:
                    # obtain GREA loss and integrate it into cls loss
                    grea_loss, causal_pred, spu_pred, causal_rep = model.get_grea_loss(graph,
                                                                                      graph.y,
                                                                                      criterion,
                                                                                      is_labeled=is_labeled,
                                                                                      return_data='rep')
                    pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    all_losses['grea'] = (all_losses.get('grea', 0) * (n_bw - 1) + grea_loss.item()) / n_bw
                    pred_loss = pred_loss + args.grea * grea_loss
                    if epoch < args.pretrain:
                        pred_loss -= args.grea * grea_loss
                elif args.cal > 0:
                    # obtain GREA loss and integrate it into cls loss
                    cal_loss_co, causal_pred, spu_pred, causal_rep = model.get_cal_loss(graph,
                                                                                      graph.y,
                                                                                      criterion,
                                                                                      is_labeled=is_labeled,
                                                                                      return_data='rep')
                    uniform_target = torch.ones_like(spu_pred, dtype=torch.float).to(device) / model.classifier.num_class
                    # print(F.softmax(spu_pred,-1))
                    cal_loss_c = F.kl_div(F.log_softmax(spu_pred,-1), uniform_target, reduction='batchmean')
                    pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    all_losses['cal_c'] = (all_losses.get('cal_c', 0) * (n_bw - 1) + cal_loss_c.item()) / n_bw
                    all_losses['cal_co'] = (all_losses.get('cal_co', 0) * (n_bw - 1) + cal_loss_co.item()) / n_bw
                    pred_loss = pred_loss + args.cal * (cal_loss_co+cal_loss_c)
                elif args.gil > 0:
                    assert args.num_envs > 0
                    # obtain GIL loss and integrate it into cls loss
                    # fetching representations for GIL is the same as DisC
                    disc_loss, causal_pred, spu_pred, causal_rep, spu_rep = model.get_disc_loss(graph, 
                                                                                                graph.y, 
                                                                                                criterion, 
                                                                                                is_labeled = is_labeled, 
                                                                                                return_data = 'spu')
                    pred_envs, cluster_centers = kmeans(
                        X=spu_rep, num_clusters=args.num_envs, distance=args.env_dis, device=device,
                        tqdm_flag=False,seed=seed
                    )
                    pred_envs = pred_envs.to(device)
                    pred_loss = 0
                    iga_pen = 0
                    env_grads = []
                    grad_mean = 0
                    for env in range(args.num_envs):
                        env_idx = (pred_envs == env) * is_labeled
                        env_loss = criterion(causal_pred[env_idx], graph.y[env_idx])
                        pred_loss += env_loss / args.num_envs 
                        model_optimizer.zero_grad()
                        env_loss.backward(retain_graph=True)
                        cur_grad = []
                        for param in model.classifier.graph_pred_linear.parameters():
                            if param.grad is not None:
                                cur_grad.append(torch.autograd.Variable(param.grad.data.clone().flatten(), requires_grad=False))
                        grad0 = torch.cat(cur_grad)
                        env_grads.append(grad0)
                        grad_mean += grad0 / env_idx.sum()
                    grad_mean = grad_mean.detach()
                    for grad in env_grads:
                        iga_pen += torch.sum((grad - grad_mean) ** 2) / args.num_envs
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.mean().item()) / n_bw
                    pred_loss = pred_loss + args.gil * iga_pen
                    all_losses['gil'] = (all_losses.get('gil', 0) * (n_bw - 1) + iga_pen.item()) / n_bw
                    if epoch < args.pretrain:
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.mean().item()) / n_bw
                elif args.disc > 0:
                    # obtain DisC loss and integrate it into cls loss
                    disc_loss, causal_pred, spu_pred, causal_rep = model.get_disc_loss(graph,
                                                                                      graph.y,
                                                                                      criterion,
                                                                                      is_labeled=is_labeled,
                                                                                      return_data='rep',
                                                                                      grad=True)
                    criterion_ldd = GeneralizedCELoss(q=args.disc_q)
                    spu_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled], reduction='none')
                    causal_pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled], reduction='none')
                    loss_weight = spu_loss / (spu_loss + causal_pred_loss + 1e-8)

                    spu_loss = criterion_ldd(spu_pred[is_labeled], graph.y[is_labeled])
                    causal_pred_loss = causal_pred_loss * (loss_weight.detach())

                    pred_loss = causal_pred_loss.mean() + args.disc * spu_loss.mean()
                    disc_loss = pred_loss
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + causal_pred_loss.mean().item()) / n_bw
                    all_losses['disc'] = (all_losses.get('disc', 0) * (n_bw - 1) + disc_loss.item()) / n_bw
                    all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_loss.mean().item()) / n_bw
                elif args.ginv_opt.lower() == 'gib':
                    # obtain gib loss
                    pred_loss, causal_rep = model.get_ib_loss(graph, return_data="rep")
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                elif args.ginv_opt.lower() == 'gsat':
                    # obtain gsat loss
                    gsat_loss, causal_pred, spu_pred, causal_rep = model.get_gsat_loss(epoch, graph, return_data="rep")
                    pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    all_losses['gsat'] = (all_losses.get('gsat', 0) * (n_bw - 1) + gsat_loss.item()) / n_bw
                    pred_loss = pred_loss + args.irm_p * gsat_loss
                else:
                    # obtain ciga I(G_S;Y) losses
                    if args.spu_coe > 0 and not args.erm:
                        if args.contrast_rep.lower() == "feat":
                            (causal_pred, spu_pred), causal_rep = model(graph, return_data="feat", return_spu=True)
                        else:
                            (causal_pred, spu_pred), causal_rep = model(graph, return_data="rep", return_spu=True)
                        
                        

                        spu_pred_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled], reduction='none')
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled], reduction='none')
                        assert spu_pred_loss.size() == pred_loss.size()
                        # hinge loss
                        spu_loss_weight = torch.zeros(spu_pred_loss.size()).to(device)
                        spu_loss_weight[spu_pred_loss > pred_loss] = 1.0
                        spu_pred_loss = spu_pred_loss.dot(spu_loss_weight) / (sum(spu_pred_loss > pred_loss) + 1e-6)
                        pred_loss = pred_loss.mean()
                        all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_pred_loss.item()) / n_bw
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    else:
                        if args.contrast_rep.lower() == "feat":
                            causal_pred, causal_rep = model(graph, return_data="feat")
                        # erm model losses
                        else:
                            causal_pred, causal_rep = model(graph, return_data="rep")
                            if args.DRO:
                                if args.Q:
                                    q, pred_loss = calculate_Q_DROloss(causal_pred[is_labeled], causal_rep[is_labeled], graph.y[is_labeled], criterion, q, num_classes, args.Q_update, eta=1e-5, lambda_reg=args.Lambda, miu=args.Miu, lr=args.P_lr, D_k=args.D_k, loss_distance=args.Distance_loss)
                                    # print(f"After update: q.requires_grad = {q.requires_grad}")
                                else:
                                    pred_loss = calculate_DROloss(causal_pred[is_labeled], graph.y[is_labeled], criterion, args.DRO_method)
                            else:
                                if args.Distance_loss>0.0:
                                    loss_vector = compute_weighted_loss_with_neighborhood(pred=causal_pred[is_labeled], criterion=criterion, graph_x=causal_rep[is_labeled], labels= graph.y[is_labeled], num_classes=num_classes, k=args.Distance_loss)
                                    pred_loss = loss_vector.mean()
                                else:   
                                    pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                            if pred_loss.item() <= 0:
                                print(f"loss: {pred_loss.item()}, Q: {q}")
                            
                            # dro+ciga
                            # causal_pred, causal_rep, loss_consistency_pred = model.get_groupdro_loss(graph, criterion, is_labeled=is_labeled, return_data="rep", return_spu=True)                         
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                        # pred_loss += loss_consistency_pred
                        # all_losses['consistency'] = (all_losses.get('consistency', 0) * (n_bw - 1) + loss_consistency_pred.item()) / n_bw
                contrast_loss = 0
                contrast_coe = args.contrast
                if args.contrast > 0:
                    # obtain contrast loss
                    # if erm_model is not None and epoch >= args.pretrain_epochs_s2:
                    if erm_model is not None:
                        # obtain the erm predictions to sampling pos/neg pairs in cnc
                        erm_model.eval()
                        with torch.no_grad():
                            # graph= casual + environment 
                            erm_y_pred, reps = erm_model(graph, return_data='rep')
                        if args.num_envs > 1:
                            # print(reps.size())
                            # print(cluster_centers.size())
                            if reps.size(0) < cluster_centers.size(0):
                                print("Insufficient amount of samples for clustering")
                                continue
                            # predict cluster ids for y
                            erm_y_pred = kmeans_predict(
                                reps, cluster_centers, args.env_dis, device=device,
                                tqdm_flag=False
                            )
                        else:
                            erm_y_pred = erm_y_pred.argmax(-1)
                    else:
                        erm_y_pred = None
                    contrast_loss = get_contrast_loss(causal_rep,
                                                      graph.y.view(-1),
                                                      norm=F.normalize if not args.not_norm else None,
                                                      contrast_t=args.contrast_t,
                                                      sampling=args.contrast_sampling,
                                                      y_pred=erm_y_pred)
                    all_losses['contrast'] = (all_losses.get('contrast', 0) * (n_bw - 1) + contrast_loss.item()) / n_bw
                    all_contrast_loss += contrast_loss.item()
                        
                if args.num_envs > 1 and erm_model is None and args.gil <= 0 and args.irm_p > 0:
                    if args.eiil:
                        if epoch < args.eiil_pretrain_iters:
                            # print("Pretraining environment inference module")
                            # training the conditional gnn & domain classifier
                            pred_loss = model.get_env_loss(graph, criterion)
                            all_losses['env'] = (all_losses.get('env', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                            all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                        else:
                            # training the main model
                            with torch.no_grad():
                                model.eval()
                                env_pred, cond_result, data_belong = model.forward_env(graph, criterion)
                                model.train()
                            pred = model(graph)
                            # mean_term = mean_loss(pred, graphs.y.float(), group)
                            _, group_indices, _ = split_into_groups(env_pred)
                            total_loss, total_cnt = 0, 0
                            for i_group in group_indices:
                                pred_group, gt_group = pred[i_group], graph.y[i_group]
                                islabeled = gt_group == gt_group
                                total_loss += criterion(
                                    pred_group[islabeled], gt_group[islabeled]
                                )
                                total_cnt += 1
                            mean_term = total_loss / total_cnt
                            
                            this_loss = criterion(pred[is_labeled], graph.y[is_labeled], reduction='none')
                            dev_term = torch.mean(torch.abs(this_loss - cond_result))
                            pred_loss = mean_term + args.irm_p * dev_term
                            all_losses['var'] = (all_losses.get('var', 0) * (n_bw - 1) + dev_term.item()) / n_bw
                            all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + mean_term.item()) / n_bw
                    else:
                        # indicate invariant learning
                        batch_env_idx = env_idx[step * args.batch_size:step * args.batch_size + graph.y.size(0)]
                        if 'molhiv' in args.dataset.lower():
                            batch_env_idx = batch_env_idx.view(graph.y.shape)
                        causal_pred, labels, batch_env_idx = causal_pred[is_labeled], graph.y[is_labeled], batch_env_idx[is_labeled]
                        if args.irm_opt.lower() == 'eiil':
                            dummy_w = torch.tensor(1.).to(device).requires_grad_()
                            loss = F.nll_loss(causal_pred * dummy_w, labels, reduction='none')
                            env_w = torch.randn(batch_env_idx.size(0)).cuda().requires_grad_()
                            optimizer = torch.optim.Adam([env_w], lr=1e-3)
                            for i in range(20):
                                # penalty for env a
                                lossa = (loss.squeeze() * env_w.sigmoid()).mean()
                                grada = torch.autograd.grad(lossa, [dummy_w], create_graph=True)[0]
                                penaltya = torch.sum(grada**2)
                                # penalty for env b
                                lossb = (loss.squeeze() * (1 - env_w.sigmoid())).mean()
                                gradb = torch.autograd.grad(lossb, [dummy_w], create_graph=True)[0]
                                penaltyb = torch.sum(gradb**2)
                                # negate
                                npenalty = -torch.stack([penaltya, penaltyb]).mean()
                                # step
                                optimizer.zero_grad()
                                npenalty.backward(retain_graph=True)
                                optimizer.step()
                            new_batch_env_idx = (env_w.sigmoid() > 0.5).long()
                            env_idx[step * args.batch_size:step * args.batch_size + graph.y.size(0)][labels] = new_batch_env_idx.to(env_idx.device)
                            irm_loss = get_irm_loss(causal_pred, labels, new_batch_env_idx, criterion=criterion)
                        elif args.irm_opt.lower() == 'ib-irm':
                            ib_penalty = causal_rep.var(dim=0).mean()
                            irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx, criterion=criterion) + ib_penalty / args.irm_p
                            all_losses['ib'] = (all_losses.get('ib', 0) * (n_bw - 1) + ib_penalty.item()) / n_bw
                        elif args.irm_opt.lower() == 'vrex':
                            loss_0 = criterion(causal_pred[batch_env_idx == 0], labels[batch_env_idx == 0])
                            loss_1 = criterion(causal_pred[batch_env_idx == 1], labels[batch_env_idx == 1])
                            irm_loss = torch.var(torch.FloatTensor([loss_0, loss_1]).to(device))
                        else:
                            irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx, criterion=criterion)
                        all_losses['irm'] = (all_losses.get('irm', 0) * (n_bw - 1) + irm_loss.item()) / n_bw
                        pred_loss += irm_loss * args.irm_p
                
                if epoch < args.pretrain:
                    batch_loss = pred_loss
                else:
                    # compile CIGA losses
                    batch_loss = pred_loss + contrast_coe * contrast_loss + args.spu_coe * spu_pred_loss
                model_optimizer.zero_grad()
                batch_loss.backward()
                model_optimizer.step()
                all_loss += batch_loss.item()
                best_weights = deepcopy(model.state_dict())
                loss_info.append(batch_loss.item())
                new_rep = torch.mean(causal_rep, dim=1)
                if args.DRO and args.Q:
                    signal = 0
                    for i in range(num_classes):
                        maski = (graph.y[is_labeled]==i)
                        if len(new_rep[maski])==0:
                            signal = 1
                            break
                    if signal == 0:
                        for i in range(num_classes):
                            maski = (graph.y[is_labeled]==i)
                            rep_dict[i] = torch.mean(new_rep[maski]).item()
                            Q_dict[i] = q[i].item()
                    
                    
                    
                    
            Q_info.append(Q_dict)
            rep_info.append(rep_dict)
            epoch_loss.append(loss_info)
            all_contrast_loss /= n_bw
            all_loss /= n_bw
            epoch_loss_val.append(all_loss)

            model.eval()
            train_acc, train_report = eval_model(model, device, train_loader, evaluator, eval_metric=eval_metric)
            val_acc, val_report = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric)
            test_acc, test_report = eval_model(model, device, test_loader, evaluator, eval_metric=eval_metric)
            if val_acc <= last_val_acc:
                cnt += epoch >= args.pretrain
            else:
                cnt = (cnt + int(epoch >= args.pretrain)) if last_val_acc == 1.0 else 0
                last_train_acc = train_acc
                last_val_acc = val_acc
                last_test_acc = test_acc
                report_temporal = [train_report, test_report, val_report]
                if args.save_model:
                    best_weights = deepcopy(model.state_dict())
            if epoch >= args.pretrain and cnt >= args.early_stopping:
                logger.info("Early Stopping")
                logger.info("+" * 101)
                logger.info("Last: Test_ACC: {:.3f} Train_ACC:{:.3f} Val_ACC:{:.3f} ".format(
                    last_test_acc, last_train_acc, last_val_acc))
                with open(ResLogFile, "a+") as file:
                    now = time.localtime()
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
                    strLog = formatted_time + "Last: Test_ACC: {:.3f} Train_ACC:{:.3f} Val_ACC:{:.3f} ".format(last_test_acc, last_train_acc, last_val_acc) 
                    file.write(strLog +"\n")
                break
            logger.info("Epoch [{:3d}/{:d}]  all_losses:{}  \n"
                        "Test_ACC:{:.3f}  Train_ACC:{:.3f} Val_ACC:{:.3f}".format(
                            epoch, args.epoch, all_losses, test_acc, train_acc, val_acc))
            test_accs.append(test_acc)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_re.append(train_report)
            test_re.append(test_report)
            val_re.append(val_report)

        # if best_weights != None:
        #     print("******compute embedding distance with best_weights******")
        #     compute_embedding_distance(model.load_state_dict(best_weights), device, train_loader)
        #     compute_embedding_distance(model.load_state_dict(best_weights), device, valid_loader)
        #     compute_embedding_distance(model.load_state_dict(best_weights), device, test_loader)
        # else:
        #     print("******compute embedding distance with last_weights******")
        #     compute_embedding_distance(model, device, train_loader)
        #     compute_embedding_distance(model, device, valid_loader)
        #     compute_embedding_distance(model, device, test_loader)
        all_info['test_acc'].append(last_test_acc)
        all_info['train_acc'].append(last_train_acc)
        all_info['val_acc'].append(last_val_acc)
        all_info['time'].append(time.time()-start_time)
        logger.info("=" * 101)
        Q_update = args.Q_update
        if args.Q_update=='distance':
            Q_update = Q_update + str(args.D_k)

        data = {
            'root':args.root,
            'dataset': args.dataset,
            # 'model': args.ginv_opt,
            # 'spurious': args.dir,
            'Distance_loss': args.Distance_loss,
            'Lambda':args.Lambda,
            'P_lr':args.P_lr,
            'seed': seed,
            'DRO': args.DRO,
            'Q': args.Q,
            'num_classes': num_classes,
            # 'DRO_method': args.DRO_method,
            'Q_update': Q_update,
            'class_report': report_temporal,
            'train_report': train_re,
            'test_report': test_re,
            'val_report': val_re,
            'rep_info': rep_info,
            'Q_info': Q_info,
            'test_acc': all_info['test_acc'],
            'train_acc': all_info['train_acc'],
            'val_acc': all_info['val_acc'],
            'test_accs': test_accs,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'epoch_loss_val': epoch_loss_val,
            'epoch_loss': epoch_loss,
            'time': all_info['time'],
        }
        experiment_datas.append(data)
        

        if args.visualize or args.visualize_top>=0:
            plot_and_save_subgraph(args, model, train_loader, eval_model, eval_metric,evaluator,device)

        if args.save_model:
            if args.save_env_as:
                print("Saving last weights..")
                model_path = os.path.join('erm_model', args.dataset) + args.commit + f"_seed{seed}.pt"
                torch.save(model.state_dict(), model_path)
                print("Done..")
            else:
                print("Saving best weights..")
                model_path = os.path.join('erm_model', args.dataset) + args.commit + f"_seed{seed}.pt"
                for k, v in best_weights.items():
                    best_weights[k] = v.cpu()
                torch.save(best_weights, model_path)
                print("Done..")
    name = args.file_name+'.csv'
    save_experiment_data(name, experiment_datas)

    with open(ResLogFile, "a+") as file: 
        now = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
        strLog = formatted_time + "Test ACC:{:.4f}-+-{:.4f}  Train ACC:{:.4f}-+-{:.4f} Val ACC:{:.4f}-+-{:.4f} Runing Time:{:.4f}-+-{:.4f} ".format(
                    torch.tensor(all_info['test_acc']).mean(),
                    torch.tensor(all_info['test_acc']).std(),
                    torch.tensor(all_info['train_acc']).mean(),
                    torch.tensor(all_info['train_acc']).std(),
                    torch.tensor(all_info['val_acc']).mean(),
                    torch.tensor(all_info['val_acc']).std(),
                    torch.tensor(all_info['time']).mean(),
                    torch.tensor(all_info['time']).std()) 
        file.write(strLog +"\n")
    
      
    logger.info("Test ACC:{:.4f}-+-{:.4f}  Train ACC:{:.4f}-+-{:.4f} Val ACC:{:.4f}-+-{:.4f} Runing Time:{:.4f}-+-{:.4f} ".format(
                    torch.tensor(all_info['test_acc']).mean(),
                    torch.tensor(all_info['test_acc']).std(),
                    torch.tensor(all_info['train_acc']).mean(),
                    torch.tensor(all_info['train_acc']).std(),
                    torch.tensor(all_info['val_acc']).mean(),
                    torch.tensor(all_info['val_acc']).std(),
                    torch.tensor(all_info['time']).mean(),
                    torch.tensor(all_info['time']).std()))
    
    def remove_extremes_and_compute_stats(data):
        tensor_data = torch.tensor(data)
        max_value, _ = torch.max(tensor_data, dim=0)
        min_value, _ = torch.min(tensor_data, dim=0)
        mask = (tensor_data != max_value) & (tensor_data != min_value)
        filtered_data = tensor_data[mask]
        mean_value = torch.mean(filtered_data)
        std_value = torch.std(filtered_data, unbiased=True) 
        return mean_value.item(), std_value.item()

    test_acc_mean, test_acc_std = remove_extremes_and_compute_stats(all_info['test_acc'])
    train_acc_mean, train_acc_std = remove_extremes_and_compute_stats(all_info['train_acc'])
    val_acc_mean, val_acc_std = remove_extremes_and_compute_stats(all_info['val_acc'])
    time_mean, time_std = remove_extremes_and_compute_stats(all_info['time'])

    logger.info("Test ACC:{:.4f}±{:.4f}  Train ACC:{:.4f}±{:.4f}  Val ACC:{:.4f}±{:.4f}  Running Time:{:.4f}±{:.4f}".format(
        test_acc_mean, test_acc_std,
        train_acc_mean, train_acc_std,
        val_acc_mean, val_acc_std,
        time_mean, time_std
    ))

    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
