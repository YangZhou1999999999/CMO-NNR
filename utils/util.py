import torch
import numpy as np

import random
from texttable import Texttable

def split_into_groups(g):
    unique_groups, unique_counts = torch.unique(
        g, sorted=False, return_counts=True
    )
    group_indices = [
        torch.nonzero(g == group, as_tuple=True)[0] for group in unique_groups
    ]
    return unique_groups, group_indices, unique_counts


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5,0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4
    
    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])
        
    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]

    
def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())
    

def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))


def node_id_list_to_mask(node_id_list, num_node):
  i = 0
  bool_list = num_node * [False]
  for i in range(num_node):
    if i in node_id_list:
      bool_list[i] = True
  return bool_list


def extract_k_hop_neighbor(graph, vertex, k):
  neighbor_list = []
  visited = set()
  neighbor_list.append(set(vertex))
  visited.update(set(vertex))
  for i in range(k):
    neighbor_candidate = set()
    for v in neighbor_list[i]:
      to_visit = set([n for n in graph.neighbors(v)])
      neighbor_candidate.update(to_visit)
    next_neigbhor = set()
    for u in neighbor_candidate:
      if u not in visited:
        next_neigbhor.add(u)
        visited.add(u)
    neighbor_list.append(next_neigbhor)
  return neighbor_list

def meanDff(x1, x2):
  return (x1.mean(0) - x2.mean(0)).norm(p=2)

# utils for GroupDRO
def to_np(x):
    return x.detach().cpu().numpy()

def to_tensor(x, device="cpu"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

def projection_simplex(v, z=1):
    """
    Old implementation for test and benchmark purposes.
    The arguments v and z should be a vector and a scalar, respectively.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w