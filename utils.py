from torch.utils.data import Dataset
import torch
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
import math
from numpy.linalg import norm
from sklearn import preprocessing
from torch import Tensor
from pathlib import Path
from torch.nn import init
import logging
import time
import random
from numpy import linalg as LA
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon, LINKXDataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.seed import seed_everything    
from Hetero_dataset import HeteroDataset
from LINKX_dataset import LINKXDataset

logger = None

seeds=[ 8073, 49184, 94208, 1681, 25443,   27880, 75161, 84677,
       32340, 38995, 78096, 37432, 70984,   841, 62755, 23832, 49295,
       63475, 30897]


def degree(row,num_nodes):
    out = torch.zeros((num_nodes, ), dtype=row.dtype)
    one = torch.ones((row.size(0), ), dtype=out.dtype)
    return out.scatter_add_(0, row, one)

def setup_logger(name):
    global logger
    logger = logging.getLogger(name)

def set_logger(args,logger,dt,name="edge"):
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}/")
    print(
        f"****** log in: {args.analysis_path}/{args.dataset}/{name}/{dt}_Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_axis_{args.axis_num}_r_{args.r}_edge_idx_{args.edge_idx_start}_seed_{args.seed}.log ******"
    )
    file_handler = logging.FileHandler(
        f"{args.analysis_path}/{args.dataset}/{name}/{dt}_Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_axis_{args.axis_num}_r_{args.r}_edge_idx_{args.edge_idx_start}_seed_{args.seed}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

def random_planetoid_splits(
    data, num_classes, percls_trn=20, val_lb=500, test_lb=1000, Flag=0
):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    if Flag == 0:
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        all_index = torch.randperm(data.y.shape[0])
        data.val_mask = index_to_mask(all_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            all_index[val_lb : (val_lb + test_lb)], size=data.num_nodes
        )
        data.train_mask = index_to_mask(
            all_index[(val_lb + test_lb) :], size=data.num_nodes
        )
    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def load_data(path,dataset,self_loop=True,undirected=True):
    if dataset in ["cora", "citeseer", "pubmed"]:
        data = Planetoid(root=path, name=dataset, split="full")
        data = data[0]
        if undirected:
            data.edge_index = to_undirected(data.edge_index)
    elif dataset in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]:
        data = PygNodePropPredDataset(name=dataset, root=path)
        split_idx = data.get_idx_split()
        data = data[0]
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask[split_idx["valid"]] = True
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze(-1)
        # logger.info(f"original edge_index: {data.edge_index.shape}")
        if undirected:
            data.edge_index = to_undirected(data.edge_index)
    elif dataset in ["computers", "photo"]:
        origin_data = Amazon(path, dataset)
        data = origin_data[0]
        data = random_planetoid_splits(
            data, num_classes=origin_data.num_classes, val_lb=500, test_lb=1000, Flag=1
        )
        if undirected:
            data.edge_index = to_undirected(data.edge_index)
    elif dataset in [
        "penn94",
        "genius",
        "wiki",
        "pokec",
        "arxiv-year",
        "twitch-gamer",
        "snap-patents",
        "twitch-de",
        "deezer-europe",
    ]:
        data = LINKXDataset(root=path, name=dataset)
        if dataset != "arxiv-year" and dataset != "snap-patents":
            if undirected:
                data.data["edge_index"] = to_undirected(data.data["edge_index"])
        data = data[0]
    elif dataset in ["questions", "minesweeper", "tolokers"]:
        data = HeteroDataset(
            root=path, name=dataset, transform=ToUndirected()
        )
        data = data[0]
    else:
        raise ("Error: Not supported dataset yet.")
    if self_loop:
        edge_index, _ = add_remaining_self_loops(data.edge_index)
    else:
        edge_index=data.edge_index
    edge_index = edge_index.numpy().astype(np.int32)
    # logger.debug(f"edge_index: {edge_index[:,:10]}")
    return data,edge_index

def get_prop_weight(weight_mode,prop_step,decay):
    weights = []
    if weight_mode == "decay":
        weight = 1.0
        for _ in range(prop_step):
            weights.append(decay * weight)
            weight *= 1 - decay
    elif weight_mode == "avg":
        for _ in range(prop_step):
            weights.append(float(1) / prop_step)
    elif weight_mode == "test":
        weights.extend([0 for _ in range(prop_step - 1)])
        weights.append(1)
    elif weight_mode == "hetero":
        for i in range(prop_step):
            weights.append(pow(-1, i))
    return weights

def preprocess_data(X, axis_num=1,ord=2):
    """
    input:
        X: (n,d), torch.Tensor
    """
    X_np = X.numpy()
    scaler = preprocessing.StandardScaler().fit(X_np)
    X_scaled = scaler.transform(X_np)
    X_norm = norm(X_scaled,ord=ord, axis=axis_num)
    X_scaled = X_scaled / X_norm.max()
    X_scaled = X_scaled.astype(np.float64)
    X_scaled = np.nan_to_num(X_scaled)
    return torch.from_numpy(X_scaled)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def com_accuracy(y_pred, y):
    pred = y_pred.data.max(1)[1]
    pred = pred.reshape(pred.size(0), 1)
    correct = pred.eq(y.data).cpu().sum()
    accuracy = correct.to(dtype=torch.long) / len(y)
    return accuracy


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def get_idx_split(name,label,split_type='random', train_prop=.6, valid_prop=.2):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
        train_mask = torch.zeros(label.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(label.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(label.shape[0], dtype=torch.bool)
        train_mask[train_idx] = 1
        val_mask[valid_idx] = 1
        test_mask[test_idx] = 1
        return train_mask, val_mask, test_mask

def get_split(data,X,train_mode,Y_binary,dataset_name="None"):
    if dataset_name in ["wiki"]:
        train_mask, val_mask, test_mask = get_idx_split(dataset_name,data.y)
    else:
        if len(data.train_mask.shape) > 1:
            # hetero datasets, multi split
            train_mask = data.train_mask[:, 0].clone().detach()
            val_mask = data.val_mask[:, 0].clone().detach()
            test_mask = data.test_mask[:, 0].clone().detach()
        else:
            train_mask = data.train_mask.clone().detach()
            val_mask = data.val_mask.clone().detach()
            test_mask = data.test_mask.clone().detach()
    X_train, X_val, X_test = (X[train_mask], X[val_mask], X[test_mask])

    # label prepare
    if train_mode == "binary":
        if "+" in Y_binary:
            # two classes are specified
            class1 = int(Y_binary.split("+")[0])
            class2 = int(Y_binary.split("+")[1])
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y == class2] = -1
        else:
            # one vs rest
            class1 = int(Y_binary)
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y != class1] = -1
        y_train, y_val, y_test = (
            Y[train_mask],
            Y[val_mask],
            Y[test_mask],
        )
    else:
        y_train = F.one_hot(data.y[train_mask],num_classes=data.y.max().item()+1) * 2 - 1
        y_train = y_train.float()
        y_val = data.y[val_mask]
        y_test = data.y[test_mask]
    return X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask

def get_split_large(data,train_mode,Y_binary,dataset_name="None"):
    if dataset_name in ["wiki"]:
        train_mask, val_mask, test_mask = get_idx_split(dataset_name,data.y)
    else:
        if len(data.train_mask.shape) > 1:
            # hetero datasets, multi split
            train_mask = data.train_mask[:, 0].clone().detach()
            val_mask = data.val_mask[:, 0].clone().detach()
            test_mask = data.test_mask[:, 0].clone().detach()
        else:
            train_mask = data.train_mask.clone().detach()
            val_mask = data.val_mask.clone().detach()
            test_mask = data.test_mask.clone().detach()

    # label prepare
    if train_mode == "binary":
        if "+" in Y_binary:
            # two classes are specified
            class1 = int(Y_binary.split("+")[0])
            class2 = int(Y_binary.split("+")[1])
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y == class2] = -1
        else:
            # one vs rest
            class1 = int(Y_binary)
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y != class1] = -1
        y_train, y_val, y_test = (
            Y[train_mask],
            Y[val_mask],
            Y[test_mask],
        )
    else:
        y_train = F.one_hot(data.y[train_mask],num_classes=data.y.max().item()+1) * 2 - 1
        y_train = y_train.float()
        y_val = data.y[val_mask]
        y_test = data.y[test_mask]
    return y_train, y_val, y_test, train_mask, val_mask, test_mask

def get_deep_split(data,X,train_mode,Y_binary,dataset_name="None"):
    if dataset_name in ["wiki"]:
        train_mask, val_mask, test_mask = get_idx_split(dataset_name,data.y)
    else:
        if len(data.train_mask.shape) > 1:
            # hetero datasets, multi split
            train_mask = data.train_mask[:, 0].clone().detach()
            val_mask = data.val_mask[:, 0].clone().detach()
            test_mask = data.test_mask[:, 0].clone().detach()
        else:
            train_mask = data.train_mask.clone().detach()
            val_mask = data.val_mask.clone().detach()
            test_mask = data.test_mask.clone().detach()
    X_train, X_val, X_test = (X[train_mask], X[val_mask], X[test_mask])

    y_train = data.y[train_mask]
    y_val = data.y[val_mask]
    y_test = data.y[test_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask

def get_deep_split_large(data,train_mode,Y_binary,dataset_name="None"):
    if dataset_name in ["wiki"]:
        train_mask, val_mask, test_mask = get_idx_split(dataset_name,data.y)
    else:
        if len(data.train_mask.shape) > 1:
            # hetero datasets, multi split
            train_mask = data.train_mask[:, 0].clone().detach()
            val_mask = data.val_mask[:, 0].clone().detach()
            test_mask = data.test_mask[:, 0].clone().detach()
        else:
            train_mask = data.train_mask.clone().detach()
            val_mask = data.val_mask.clone().detach()
            test_mask = data.test_mask.clone().detach()
    y_train = data.y[train_mask]
    y_val = data.y[val_mask]
    y_test = data.y[test_mask]

    return  y_train, y_val, y_test, train_mask, val_mask, test_mask

def check_propagation(groundtruth, result):
    print(groundtruth.shape, result.shape)
    assert len(groundtruth) == len(result)
    l1error = np.sum(np.abs(groundtruth - result)) / len(groundtruth)
    maxl1error = max(
        [
            np.sum(np.abs(groundtruth[i] - result[i]))
            for i in range(groundtruth.shape[0])
        ]
    )
    maxerror = max(
        [
            np.max(np.abs(groundtruth[i] - result[i]))
            for i in range(groundtruth.shape[0])
        ]
    )
    index = np.unravel_index(
        np.argmax(np.abs(groundtruth - result)),
        (groundtruth.shape[0], groundtruth.shape[1]),
    )
    print("max error at: ", index)
    print("max error: ", groundtruth[index], result[index])
    print("max l1-error: ", maxl1error)
    print("max error: ", maxerror)
    return maxl1error, maxerror


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_budget(std, eps, c):
    return std * eps / c


def get_worst_Gbound_edge(deg1,deg2,train_size,feat_dim,lam,rmax,num_nodes,prop_step):
    c=1
    c_1=1
    gamma_1=1/4
    gamma_2=1/4
    epsilon_1=math.sqrt(num_nodes)*prop_step*rmax
    multi1=c*gamma_1/lam*feat_dim+c_1*math.sqrt(feat_dim*train_size)
    multi2=epsilon_1+2*gamma_1*feat_dim/train_size*(4/math.sqrt(deg1)+4/math.sqrt(deg2))
    return multi1*multi2

def get_worst_Gbound_node(degs,train_size,feat_dim,lam,rmax,num_nodes,prop_step):
    c=1
    c_1=1
    gamma_1=1/4
    gamma_2=1/4
    epsilon_1=math.sqrt(num_nodes)*prop_step*rmax
    degsum=0
    for _deg in degs:
        degsum+=4/math.sqrt(_deg)
    multi1=c*gamma_1/lam*feat_dim+c_1*math.sqrt(feat_dim*train_size)
    multi2=epsilon_1+2*gamma_1*feat_dim/train_size*degsum
    return multi1*multi2

def get_worst_Gbound_feat(_deg,train_size,feat_dim,lam,rmax,num_nodes,prop_step):
    c=1
    c_1=1
    gamma_1=1/4
    gamma_2=1/4
    epsilon_1=math.sqrt(num_nodes)*prop_step*rmax
    multi1=c*gamma_1/lam*feat_dim+c_1*math.sqrt(feat_dim*train_size)
    multi2=epsilon_1+2*gamma_1*feat_dim/train_size*math.sqrt(_deg)
    return multi1*multi2

def get_c(delta):
    return np.sqrt(2 * np.log(1.5 / delta))
