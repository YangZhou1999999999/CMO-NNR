from BA3_loc import *
import torch
import numpy as np
import random
from tqdm import tqdm
from texttable import Texttable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)
global_b = '0.4' # The global_b controls the spurious correlation strength
label_noise = 0.01  # the label_noise controls the invariant correlation strength(1-label_noise)

motif_names = ['diamond', 'house', 'varcycle', 'crane', 'crossgrid', 'star'] 
# 'dircycle',
motif_ids = [0, 1, 2, 3, 4, 5]
base_ids = [0, 1, 2, 3, 4, 5]
num_train_group = [3000, 3000, 3000, 300, 3000, 3000]
num_val_group = [1000, 1000, 1000, 1000, 1000, 1000]
num_test_group = [1000, 1000, 1000, 1000, 1000, 1000]

# motif_names = ['diamond', 'house', 'varcycle', 'crane', 'crossgrid'] 
# # 'dircycle',
# motif_ids = [0, 1, 2, 3, 4]
# base_ids = [0, 1, 2, 3, 4]
# num_train_group = [3000, 3000, 300, 3000, 3000]
# num_val_group = [1000, 1000, 1000, 1000, 1000]
# num_test_group = [1000, 1000, 1000, 1000, 1000]
# num_train_perc = 3000
# num_val_perc = 1000
# num_test_perc = 1000
test_bias = 1/len(motif_names)  # the test_bias controls the invariant correlation strength
# clique cycle diamond tree ba wheel ladder house fan varcycle crane dircycle crossgrid star path


def generate_graph(graph_type, basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph Generator:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        graph_type        :  The type of graph to generate ('clique', 'cycle', 'diamond', 'tree', 'ba').
        nb_shapes         :  The number of shapes that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.
        draw              :  Whether to draw the graph.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [[graph_type]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    start=0,
                                                    rdm_basis_plugins=True)
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def generate_data(motif_id, bias, label_noise, base_list):
    label = motif_id
    p = [(1 - bias) / (len(base_list) - 1)] * len(base_list)
    p[motif_id] = bias
    base_num = np.random.choice(base_list, p=p)

    if base_num == 0:
        base = 'tree'
        width_basis = np.random.choice(range(3, 4))
    if base_num == 1:
        base = 'cycle'
        width_basis = np.random.choice(range(8, 12))
    if base_num == 2:
        base = 'path'
        width_basis = np.random.choice(range(5, 10))
    if base_num == 3:
        base = 'wheel'
        width_basis = np.random.choice(range(5, 10))
    if base_num == 4:
        base = 'ladder'
        width_basis = np.random.choice(range(5, 10))
    if base_num == 5:
        base = 'fan'
        width_basis = np.random.choice(range(5, 10))
    
    # the label_noise controls the invariant correlation strength
    if random.random() < label_noise:
        motif_id = random.randint(0, len(base_list) - 1)
        
        # gen_name = f"get_{motif_names[motif_id]}"
    graph_type = motif_names[motif_id]
    G, role_id, name = generate_graph(graph_type=graph_type,basis_type=base,nb_shapes=1,width_basis=width_basis,feature_generator=None,m=3,draw=False)
    return G, role_id, name, label, base_num, motif_id
###########################
# All Training data
###########################

edge_index_list = []
label_list = []
ground_truth_list = []
role_id_list = []
pos_list = []
base_list = []

bias = float(global_b)
motif_id_list = []
cnt_cau_spu = 0
num_train_perc = 10 
e_mean = []
n_mean = []
train_names = []
val_names = []
test_names = []

train_data_distribution = {f"{i}-{j}": 0 for i in motif_ids for j in range(len(motif_names))}
val_data_distribution = {f"{i}-{j}": 0 for i in motif_ids for j in range(len(motif_names))}
test_data_distribution = {f"{i}-{j}": 0 for i in motif_ids for j in range(len(motif_names))}
# environment 1: The global_b controls the spurious correlation strength

for i in motif_ids:
    e_mean = []
    n_mean = []
    for _ in tqdm(range(num_train_group[i])):
        G, role_id, name, label, base_num, motif_id = generate_data(i, bias, label_noise, base_ids)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(label)
        base_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
        motif_id_list.append(motif_id)
        train_names.append(str(label) + '_' + str(base_num) + '_' + str(motif_id))
        train_data_distribution[f"{motif_id}-{base_num}"] += 1
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

if not os.path.exists(f'./data9/tSPMotif-{global_b}/'):
    os.makedirs(f'./data9/tSPMotif-{global_b}/')
if not os.path.exists(f'./data9/tSPMotif-{global_b}/raw'):
    os.makedirs(f'./data9/tSPMotif-{global_b}/raw')
np.save(f'./data9/tSPMotif-{global_b}/raw/train.npy', np.array((edge_index_list, label_list, base_list, ground_truth_list, role_id_list, pos_list, motif_id_list), dtype=object))



###########################
# All Validation data
###########################

edge_index_list = []
label_list = []
ground_truth_list = []
role_id_list = []
pos_list = []
base_list = []

bias = max(float(global_b) - 0.2, 1.0 / 6)

for i in motif_ids:
    e_mean = []
    n_mean = []
    for _ in tqdm(range(num_val_group[i])):
        G, role_id, name, label, base_num, motif_id = generate_data(i, bias, label_noise, base_ids)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(label)
        base_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
        motif_id_list.append(motif_id)
        val_names.append(str(label) + '_' + str(base_num) + '_' + str(motif_id))
        val_data_distribution[f"{motif_id}-{base_num}"] += 1
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))


np.save(f'./data9/tSPMotif-{global_b}/raw/val.npy', np.array((edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list), dtype=object))



###########################
# All Test data
###########################

edge_index_list = []
label_list = []
ground_truth_list = []
role_id_list = []
pos_list = []
base_list = []

bias = test_bias

for i in motif_ids:
    e_mean = []
    n_mean = []
    for _ in tqdm(range(num_test_group[i])):
        G, role_id, name, label, base_num, motif_id = generate_data(i, bias, label_noise, base_ids)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(label)
        base_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
        motif_id_list.append(motif_id)
        test_names.append(str(label) + '_' + str(base_num) + '_' + str(motif_id))
        test_data_distribution[f"{motif_id}-{base_num}"] += 1
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))
np.save(f'./data9/tSPMotif-{global_b}/raw/test.npy', np.array((edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list), dtype=object))

file_path = f'./data9/tSPMotif-{global_b}/names.txt'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
# Write the names to the text file
with open(file_path, 'w') as f:
    # Write train data distribution
    f.write("label_noise:\n")
    f.write(f"{label_noise}\n\n")
    f.write("Train Data Distribution:\n")
    f.write(f"{train_data_distribution}\n\n")
    
    # Write val data distribution
    f.write("\nVal Data Distribution:\n")
    f.write(f"{val_data_distribution}\n\n")
    
    # Write test data distribution
    f.write("\nTest Data Distribution:\n")
    f.write(f"{test_data_distribution}\n\n")
    