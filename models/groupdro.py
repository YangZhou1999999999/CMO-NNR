import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from models.gnn import GNN
from utils.util import *



class GroupDRO(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GroupDRO, self).__init__()
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)
        self.src_domain = out_dim
        # q
        self.register_buffer("q", torch.Tensor())
        self.groupdro_eta = 0.5
        self.lmbda = 1

    def forward(self, batch, return_data="pred"):
        causal_pred, causal_rep = self.classifier(batch, get_rep=True)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")

    def get_groupdro_loss(self, batched_data, criterion, is_labeled=None, return_data="pred", return_spu=True):
        causal_pred, causal_rep = self.classifier(batched_data, get_rep=True)
        labels = batched_data.y 
        self.device='cpu'
        # Group-DRO
        # if not len(self.q):
        #     self.q = torch.ones(self.src_domain)
        # losses = torch.zeros(self.src_domain)
        
        # for m in range(self.src_domain):
        #     mask = (labels == m)
        #     losses[m] = criterion(causal_pred[mask], labels[mask]) 
        #     self.q[m] *= (self.groupdro_eta * losses[m].data).exp()
        
        # self.q /= self.q.sum()
        # loss_E_pred = torch.dot(losses, self.q)    
               
        # Prior-DRO
        if not len(self.q):
            # Learnable q
            self.q = torch.ones(self.src_domain).to(self.device)
            self.q /= self.q.sum() # Key!

        losses = torch.zeros(self.src_domain).to(self.device)
        prior = torch.zeros(self.src_domain)
        # prior: stability
        for m in range(self.src_domain):
            mask = (labels == m)
            # compute consistency
            distances = torch.norm(causal_pred[mask], dim=1)
            # var represents consistency
            variance = torch.mean(torch.var(distances, dim=0)).to('cpu')
            prior[m] = variance
        prior /= prior.sum()

        for m in range(self.src_domain):
            mask = (labels == m)
            losses[m] = criterion(causal_pred[mask], labels[mask])
            self.q[m] += self.groupdro_eta * (losses[m] - self.lmbda * (self.q[m] - prior[m]))

        self.q = to_tensor(projection_simplex(to_np(self.q)), self.device)

        loss_E_pred = torch.dot(losses, self.q.to(torch.float))
        
        return loss_E_pred

