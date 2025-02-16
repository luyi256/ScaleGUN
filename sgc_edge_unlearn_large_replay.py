from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression
import gc
# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional

from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import propagation
from torch.nn import init
from sgc_utils import *

from sklearn import preprocessing
from numpy.linalg import norm
import struct
import logging
import pytz
from torch_geometric.utils import to_undirected
from datetime import datetime
from torch_geometric.seed import seed_everything


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


seeds = [8073, 49184, 94208,  1681,  25443, 27880, 75161, 84677,
         32340, 38995, 78096, 37432, 70984,   841, 62755, 23832, 49295,
         63475, 30897]

if __name__ == "__main__":
    del_path = "/mnt_1/lu_yi/data/unlearning_data/"
    num_threads = 40
    parser = argparse.ArgumentParser(
        description="Training a removal-enabled linear model [edge]"
    )
    parser.add_argument("--analysis_path", default="./analysis/")
    parser.add_argument("--path", default="/mnt_1/lu_yi/data/unlearning_data/")
    parser.add_argument(
        "--data_dir", type=str, default="/mnt_1/lu_yi/data/", help="data directory"
    )
    parser.add_argument(
        "--result_dir", type=str, default="result", help="directory for saving results"
    )
    parser.add_argument("--dataset", type=str, default="cora", help="dataset")
    parser.add_argument("--lam", type=float, default=1e-2,
                        help="L2 regularization")
    parser.add_argument(
        "--std",
        type=float,
        default=1e-1,
        help="standard deviation for objective perturbation",
    )
    parser.add_argument(
        "--num_removes", type=int, default=1, help="number of data points to remove"
    )
    parser.add_argument(
        "--num_batch_removes", type=int, default=1000, help="Number of repeated trails."
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="number of optimization steps"
    )
    parser.add_argument(
        "--train_mode", type=str, default="ovr", help="train mode [ovr/binary]"
    )
    parser.add_argument(
        "--train_sep",
        action="store_true",
        default=False,
        help="train binary classifiers separately",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="verbosity in optimizer"
    )
    # New arguments below
    parser.add_argument(
        "--device", type=int, default=1, help="nonnegative int for cuda id, -1 for cpu"
    )
    parser.add_argument(
        "--prop_step",
        type=int,
        default=3,
        help="number of steps of graph propagation/convolution",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=1.0,
        help="we use D^{-a}AD^{-(1-a)} as propagation matrix",
    )
    parser.add_argument(
        "--XdegNorm",
        type=bool,
        default=False,
        help="Apply our degree normaliztion trick",
    )
    parser.add_argument(
        "--weight_mode",
        default="test",
        type=str,
        choices=["decay", "avg", "test", "hetero"],
    )
    parser.add_argument("--decay", default=0.1, type=float)
    parser.add_argument(
        "--add_self_loops",
        type=bool,
        default=True,
        help="Add self loops in propagation matrix",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LBFGS",
        help="Choice of optimizer. [LBFGS/Adam]",
    )
    parser.add_argument("--lr", type=float, default=1, help="Learning rate")
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="Weight decay factor for Adam"
    )
    parser.add_argument(
        "--featNorm", type=bool, default=True, help="Row normalize feature to norm 1."
    )
    parser.add_argument(
        "--GPR", action="store_true", default=False, help="Use GPR model"
    )
    parser.add_argument(
        "--balance_train",
        action="store_true",
        default=False,
        help="Subsample training set to make it balance in class size.",
    )
    parser.add_argument(
        "--ceu",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--Y_binary",
        type=str,
        default="0",
        help="In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="data",
        help="Data dependent noise or worst case noise [data/worst].",
    )
    parser.add_argument(
        "--removal_mode", type=str, default="node", help="[feature/edge/node]."
    )
    parser.add_argument(
        "--eps", type=float, default=1.0, help="Eps coefficient for certified removal."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-4,
        help="Delta coefficient for certified removal.",
    )
    parser.add_argument("--disp", type=int, default=100,
                        help="Display frequency.")
    parser.add_argument(
        "--trails", type=int, default=1, help="Number of repeated trails."
    )
    parser.add_argument(
        "--fix_random_seed",
        action="store_true",
        default=False,
        help="Use fixed random seed for removal queue.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Use fixed random seed for removal queue.",
    )
    parser.add_argument(
        "--compare_gnorm",
        action="store_true",
        default=False,
        help="Compute norm of worst case and real gradient each round.",
    )
    # parser.add_argument(
    #     "--compare_retrain",
    #     action="store_true",
    #     default=False,
    #     help="Compare acc with retraining each round.",
    # )
    parser.add_argument("--axis_num", default=1, type=int)
    parser.add_argument("--edge_idx_start", default=0, type=int)
    parser.add_argument("--attack_dim", default=100, type=int)
    parser.add_argument(
        "--test_MI",
        action="store_true",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
    )
    # Use this if turning into .py code
    args = parser.parse_args()
    name = "sgc_unlearn_nf_large_replay"

    seed_everything(seeds[args.seed])
    tz = pytz.timezone("Asia/Shanghai")
    dt = datetime.now(tz).strftime("%m%d_%H%M")

    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s - %(message)s", handlers=[])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}/")
    print(
        f"******log in: {args.analysis_path}/{args.dataset}/{name}/{dt}_Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_axis_{args.axis_num}_r_{args.r}_edge_idx_{args.edge_idx_start}.log ******"
    )
    file_handler = logging.FileHandler(
        f"{args.analysis_path}/{args.dataset}/{name}/{dt}_Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_axis_{args.axis_num}_r_{args.r}_edge_idx_{args.edge_idx_start}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result/")
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model/")
    logger.info(args)
    tot_cost_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_prop_{args.prop_step}_r_{args.r}_{args.removal_mode}_cost"
    unlearn_cost_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_prop_{args.prop_step}_r_{args.r}_{args.removal_mode}_unlearn_cost"
    update_cost_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_prop_{args.prop_step}_r_{args.r}_{args.removal_mode}_update_cost"
    acc_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_prop_{args.prop_step}_r_{args.r}_{args.removal_mode}_acc"
    f_tot_cost = open(tot_cost_path+".txt", "ab")
    f_unlearn_cost = open(unlearn_cost_path+".txt", "ab")
    f_update_cost = open(update_cost_path+".txt", "ab")
    f_acc = open(acc_path+".txt", "ab")
    log_prefix = f"{args.analysis_path}/{args.dataset}/{name}_model/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_std_{args.std}_prop_{args.prop_step}_r_{args.r}_{args.removal_mode}_edge_idx_{args.edge_idx_start}"

    # Use this if running using notebook
    # args = parser.parse_args([])

    # this script is only for feature/node removal
    assert args.removal_mode in ['feature', 'node']
    # dont compute norm together with retrain
    # assert not (args.compare_gnorm and args.compare_retrain)

    if args.device > -1:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    ######
    # Load the data
    logger.info("=" * 10 + "Loading data" + "=" * 10)
    logger.info(f"Dataset: {args.dataset}")
    # read data from PyG datasets (cora, citeseer, pubmed)
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(args.data_dir, args.dataset, split="full")
        data = dataset[0]
    elif args.dataset in ["ogbn-arxiv", "ogbn-products"]:
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_dir)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask[split_idx["valid"]] = True
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze(-1)
    elif args.dataset in ["computers", "photo"]:
        # path = osp.join(args.data_dir, "data", args.dataset)
        dataset = Amazon(args.data_dir, args.dataset)
        data = dataset[0]
        data = random_planetoid_splits(
            data, num_classes=dataset.num_classes, val_lb=500, test_lb=1000, Flag=1
        )
    else:
        raise ("Error: Not supported dataset yet.")

    # save the degree of each node for later use
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    row = data.edge_index[0]
    deg = degree(row)
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]-num_nodes
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    # print(data.x[-5:, -5:])
    if args.delta < 0:
        args.delta = 1/num_nodes
        logger.info(f"delta: {args.delta}")

    node_idx_start = args.edge_idx_start
    node_file = (
        del_path + "/" + args.dataset + "/" + args.dataset + "_del_nodes.npy"
    )
    del_nodes = np.load(node_file)
    np.random.shuffle(del_nodes)
    del_nodes = del_nodes[node_idx_start:node_idx_start +
                          args.num_removes*args.num_batch_removes]

    # process features

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    if args.replay:
        # find the minimal class
        if args.dataset == 'ogbn-products':
            min = data.num_nodes
            min_c = -1
            for c in range(num_classes):
                cnt = (data.y == c).sum()
                if cnt < args.num_removes*args.num_batch_removes:
                    continue
                if cnt < min:
                    min = cnt
                    min_c = c
            print(f'Choose class {min_c}, with {min} nodes')
            c_nodes = (data.y == min_c).nonzero().flatten().numpy()
            np.random.shuffle(c_nodes)
            del_nodes = c_nodes[
                node_idx_start:node_idx_start + args.num_removes*args.num_batch_removes]
        padding = torch.zeros(
            (data.num_nodes, args.attack_dim), dtype=torch.float)
        feat = torch.cat([feat, padding], dim=1)
        feat[del_nodes, -args.attack_dim:] = 1
        data.y[del_nodes] = num_classes
        num_classes += 1
        feat_dim += args.attack_dim
    feat = feat.T
    if args.dataset in ["ogbn-arxiv", "ogbn-products", "pokec"]:
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()
    # save a copy of X for removal
    # print("ATTEN!!!", X[:, 1000].cpu().numpy())

    # process labels
    if args.train_mode == "binary":
        if "+" in args.Y_binary:
            # two classes are specified
            class1 = int(args.Y_binary.split("+")[0])
            class2 = int(args.Y_binary.split("+")[1])
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y == class2] = -1
            interested_data_mask = (data.y == class1) + (data.y == class2)
            train_mask = data.train_mask * interested_data_mask
            val_mask = data.val_mask * interested_data_mask
            test_mask = data.test_mask * interested_data_mask
        else:
            # one vs rest
            class1 = int(args.Y_binary)
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y != class1] = -1
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        y_train, y_val, y_test = (
            Y[train_mask].to(device),
            Y[val_mask].to(device),
            Y[test_mask].to(device),
        )
    else:
        # multiclass classification
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        y_train = F.one_hot(data.y) * 2 - 1
        y_train = y_train[data.train_mask].float().to(device)
        y_val = data.y[data.val_mask].to(device)
        y_test = data.y[data.test_mask].to(device)

    assert args.noise_mode == "data"

    if args.compare_gnorm:
        # if we want to compare the residual gradient norm of three cases, we should not add noise
        # and make budget very large
        b_std = 0
    else:
        if args.noise_mode == "data":
            b_std = args.std
        elif args.noise_mode == "worst":
            b_std = args.std  # change to worst case sigma
        else:
            raise ("Error: Not supported noise model.")

    #############
    # initial training with graph
    logger.info("=" * 10 + "Training on full dataset with graph" + "=" * 10)

    edge_index, _ = add_remaining_self_loops(data.edge_index)
    edge_index = edge_index.cpu().numpy().astype(np.int32)

    # Propagation = MyGraphConv(
    #     K=args.prop_step,
    #     add_self_loops=args.add_self_loops,
    #     alpha=args.r,
    #     XdegNorm=args.XdegNorm,
    #     GPR=args.GPR,
    # ).to(device)

    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)
    del_path = "/mnt_1/lu_yi/data/unlearning_data/"
    g.init_graph(del_path, args.dataset, edge_index,
                 args.prop_step, args.r, weights, num_threads, feat_dim)
    start_train = time.perf_counter()
    origin_embedding = np.copy(feat.numpy())
    prop_time = g.PowerMethod(origin_embedding)
    logger.info(f"initial prop time: {prop_time}")
    del edge_index
    gc.collect()
    init_finish_time = time.perf_counter()
    # print("ATTEN!!!", X[:, 1000].cpu().numpy())
    X = torch.FloatTensor(origin_embedding.T)
    X_train = X[train_mask].to(device)
    X_val = X[val_mask].to(device)
    X_test = X[test_mask].to(device)
    # result_txt = f"{args.path}/{args.dataset}/{args.dataset}_{args.prop_step+1}_{args.r}_sgc_result.txt"
    # np.savetxt(result_txt, X.cpu().t().numpy(), fmt="%.10f")
    # np.save(result_txt.replace(".txt", ".npy"), X.cpu().t().numpy())

    # logger.info(
    #     "Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(
    #         X_train.shape[0],
    #         X_val.shape[0],
    #         X_test.shape[0],
    #         data.edge_index.shape[1],
    #         X_train.shape[1],
    #     )
    # )

    # train removal-enabled linear model
    # logger.info(
    #     f"With graph, train mode:{args.train_mode}, optimizer: {args.optimizer}"
    # )

    weight = None
    # in our case weight should always be None
    assert weight is None
    opt_grad_norm = 0

    if args.train_mode == "ovr":
        b = b_std * torch.randn(feat_dim, num_classes).float().to(device)
        logger.info(f"b:{b}")
        # print(b.shape)
        if args.train_sep:
            # train K binary LR models separately
            w = torch.zeros(b.size()).float().to(device)
            for k in range(num_classes):
                if weight is None:
                    w[:, k] = lr_optimize(
                        X_train,
                        y_train[:, k],
                        args.lam,
                        b=b[:, k],
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                else:
                    w[:, k] = lr_optimize(
                        X_train[weight[:, k].gt(0)],
                        y_train[:, k][weight[:, k].gt(0)],
                        args.lam,
                        b=b[:, k],
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
        else:
            # train K binary LR models jointly
            w = ovr_lr_optimize(
                X_train,
                y_train,
                args.lam,
                weight,
                b=b,
                num_steps=args.num_steps,
                verbose=args.verbose,
                opt_choice=args.optimizer,
                lr=args.lr,
                wd=args.wd,
            )
        # record the opt_grad_norm
        train_finish_time = time.perf_counter()
        for k in range(y_train.size(1)):
            opt_grad_norm += (
                lr_grad(w[:, k], X_train, y_train[:, k], args.lam).norm().cpu()
            )
    else:
        b = b_std * torch.randn(X_train.size(1)).float().to(device)
        w = lr_optimize(
            X_train,
            y_train,
            args.lam,
            b=b,
            num_steps=args.num_steps,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            lr=args.lr,
            wd=args.wd,
        )
        train_finish_time = time.perf_counter()
        opt_grad_norm = lr_grad(w, X_train, y_train, args.lam).norm().cpu()

    logger.info("Time elapsed: %.2fs" % (time.perf_counter() - start_train))
    checkpt_file = log_prefix+"_init.pt"
    # torch.save(w, checkpt_file)
    embed_file = log_prefix+"_init"
    # np.save(embed_file, X.cpu().numpy())
    unlearn_cost = [train_finish_time-start_train,]
    update_cost = [prop_time,]
    removal_times = [train_finish_time-start_train+prop_time,]
    if args.train_mode == "ovr":
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
        print('Val accuracy = %.4f' % val_acc)
        print('Test accuracy = %.4f' % test_acc)
    else:
        val_acc = lr_eval(w, X_val, y_val)
        test_acc = lr_eval(w, X_test, y_test)
        print('Val accuracy = %.4f' % val_acc)
        print('Test accuracy = %.4f' % test_acc)
    acc_removal = [[val_acc.item()], [test_acc.item()]]

    # if args.test_MI:
    #     MI_ori_pred = torch.empty(
    #         (0, num_classes), dtype=torch.float).to(device)
    #     for i in range(args.num_batch_removes):
    #         nodes = del_nodes[
    #             node_idx_start
    #             + i * args.num_removes: node_idx_start
    #             + args.num_removes * (i + 1),
    #         ].T.tolist()
    #         MI_ori_pred = torch.cat([MI_ori_pred, predict(
    #             w, X[nodes].to(device))], dim=0)
    #     MI_path = f"./analysis/{args.dataset}/{name}_MI/"
    #     check_dir(MI_path)
    #     np.save(f'{MI_path}/MI_pred_{name}_ori_{args.num_batch_removes}_{args.num_removes}_{args.seed}.npy',
    #             MI_ori_pred.cpu().numpy())
    # if args.replay:
    #     all_pred = predict_class(w, X.to(device))
    #     res = all_pred == num_classes - 1
    #     res_deleted = res[del_nodes]
    #     logger.info(
    #         f"Ori Replay test false: {res_deleted.sum()} {res_deleted.shape[0]} Ori Total test false: {res.sum()} {res.shape[0]}")
    #     print(
    #         f"Ori Replay test false: {res_deleted.sum()} {res_deleted.shape[0]} Ori Total test false: {res.sum()} {res.shape[0]}")
    #     replay_path = f"./analysis/{args.dataset}/{name}_MI/"
    #     check_dir(replay_path)
    #     np.savez(f"{replay_path}/replay_pred_ori_{name}_{args.seed}.npz",
    #              all_pred=all_pred.cpu().numpy(), false_pred=res.cpu().numpy(), del_nodes=del_nodes)

    X_train_old = X_train.clone().detach().to(device)

    del X_train, X_val, X_test, X

    ###########
    # budget for removal
    c_val = get_c(args.delta)
    if args.compare_gnorm:
        budget = 1e5
    else:
        if args.train_mode == "ovr":
            budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
        else:
            budget = get_budget(b_std, args.eps, c_val)
    gamma = 1 / 4  # pre-computed for -logsigmoid loss
    logger.info(f"Budget: {budget}")

    ##########
    # our removal
    # grad_norm_approx is the data dependent upper bound of residual gradient norm
    grad_norm_approx = torch.zeros(
        (args.num_batch_removes, args.trails)).float()

    grad_norm_worst = torch.zeros(
        (args.num_batch_removes, args.trails)
    ).float()  # worst case norm bound
    grad_norm_real = torch.zeros(
        (args.num_batch_removes, args.trails)
    ).float()  # true norm
    # graph retrain
    removal_times_graph_retrain = torch.zeros(
        (args.num_batch_removes, args.trails)
    ).float()
    acc_graph_retrain = torch.zeros(
        (2, args.num_batch_removes, args.trails)).float()

    y_train_old = y_train.clone().detach().to(device)
    for trail_iter in range(args.trails):
        logger.info(f"********** {trail_iter} **********")
        # if args.fix_random_seed:
        #     np.random.seed(trail_iter)
        # get a random permutation for edge indices for each trail
        # perm = torch.from_numpy(np.random.permutation(data.edge_index.shape[1]))

        # Note that all edges are used in training, so we just need to decide the order to remove edges
        # the number of training samples will always be m

        w_approx = w.clone().detach().to(device)  # copy the parameters to modify
        num_retrain = 0
        grad_norm_approx_sum = 0
        # perm_idx = 0
        # start the removal process
        logger.info("=" * 10 + "Testing our edge removal" + "=" * 10)
        for i in range(args.num_batch_removes):
            # First, check if this is a self-loop or an edge already deleted
            # while (
            #     data.edge_index[0, perm[perm_idx]] == data.edge_index[1, perm[perm_idx]]
            # ) or (not edge_mask[perm[perm_idx]]):
            #     perm_idx += 1
            # edge_mask[perm[perm_idx]] = False
            # source_idx = data.edge_index[0, perm[perm_idx]]
            # dst_idx = data.edge_index[1, perm[perm_idx]]
            # # find the other undirected edge
            # rev_edge_idx = (
            #     torch.logical_and(
            #         data.edge_index[0] == dst_idx, data.edge_index[1] == source_idx
            #     )
            #     .nonzero()
            #     .squeeze(-1)
            # )
            # if rev_edge_idx.size(0) > 0:
            #     edge_mask[rev_edge_idx] = False
            nodes = del_nodes[
                node_idx_start
                + i * args.num_removes: node_idx_start
                + args.num_removes * (i + 1),
            ].T.tolist()
            start = time.perf_counter()
            if args.removal_mode == "node":
                g.UpdateStructNodes(nodes)
                if args.replay:
                    feat[:, nodes] = 0
                retrain_embedding = np.copy(feat.numpy())
                g.PowerMethod(retrain_embedding)
            else:
                feat[:, nodes] = 0
                retrain_embedding = np.copy(feat.numpy())
                g.PowerMethod(retrain_embedding)
            start_unlearn = time.perf_counter()
            X_new = torch.FloatTensor(retrain_embedding.T)
            train_mask[nodes] = False
            X_rem = X_new[train_mask].to(device)
            y_train = F.one_hot(
                data.y[train_mask], num_classes=data.y.max().item()+1) * 2 - 1
            y_train = y_train.float().to(device)

            # note that the removed data point should still not be used in computing K or H
            # removal_queue[(i+1):] are the remaining training idx
            K = get_K_matrix(X_rem).to(device)
            spec_norm = sqrt_spectral_norm(K)

            if args.train_mode == 'ovr':
                # removal from all one-vs-rest models

                for k in range(num_classes):
                    y_rem = y_train[:, k]
                    H_inv = lr_hessian_inv(
                        w_approx[:, k], X_rem, y_rem, args.lam)
                    # grad_i is the difference
                    grad_old = lr_grad(
                        w_approx[:, k], X_train_old, y_train_old[:, k], args.lam)
                    grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, args.lam)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    # update w here. If beta exceed the budget, w_approx will be retrained
                    w_approx[:, k] += Delta
                    # data dependent norm
                    grad_norm_approx[i, trail_iter] += (
                        Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                    if args.compare_gnorm:
                        grad_norm_real[i, trail_iter] += lr_grad(
                            w_approx[:, k], X_rem, y_rem, args.lam).norm().cpu()
                        # if args.removal_mode == 'node':
                        #     # grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(args.lam, X_rem.shape[0],args.prop_step,deg[removal_queue[i]]).cpu()
                        #     pass
                        # elif args.removal_mode == 'feature':
                        #     grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(args.lam, X_rem.shape[0],
                        #                                                                deg[removal_queue[i]]).cpu()
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(feat_dim,
                                            num_classes).float().to(device)
                    w_approx = ovr_lr_optimize(X_rem, y_train, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose,
                                               opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                finish_time = time.perf_counter()
                X_val_new = X_new[val_mask].to(device)
                X_test_new = X_new[test_mask].to(device)
                acc_removal[0].append(ovr_lr_eval(
                    w_approx, X_val_new, y_val).item())
                acc_removal[1].append(ovr_lr_eval(
                    w_approx, X_test_new, y_test).item())
            else:
                # removal from a single binary logistic regression model

                y_rem = y_train
                H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, args.lam)
                # grad_i should be the difference
                grad_old = lr_grad(w_approx, X_train_old,
                                   y_train_old, args.lam)
                grad_new = lr_grad(w_approx, X_rem, y_rem, args.lam)
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                w_approx += Delta
                grad_norm_approx[i, trail_iter] += (
                    Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                if args.compare_gnorm:
                    grad_norm_real[i, trail_iter] += lr_grad(
                        w_approx, X_rem, y_rem, args.lam).norm().cpu()
                    # if args.removal_mode == 'node':
                    #     grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(args.lam, X_rem.shape[0],
                    #                                                             args.prop_step,
                    #                                                             deg[removal_queue[i]]).cpu()
                    # elif args.removal_mode == 'feature':
                    #     grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(args.lam, X_rem.shape[0],
                    #                                                                deg[removal_queue[i]]).cpu()

                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(feat_dim).float().to(device)
                    w_approx = lr_optimize(X_rem, y_rem, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose,
                                           opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                acc_removal[0, i, trail_iter] = lr_eval(
                    w_approx, X_val_new, y_val)
                acc_removal[1, i, trail_iter] = lr_eval(
                    w_approx, X_test_new, y_test)

            removal_times.append(finish_time - start)
            unlearn_cost.append(finish_time-start_unlearn)
            update_cost.append(start_unlearn-start)
            # Remember to replace X_old with X_new
            X_train_old = X_rem.clone().detach()
            y_train_old = y_train.clone().detach()
            if i % args.disp == 0:
                logger.info(
                    f"Iteration {i}: Edge del = {nodes[0]}, grad_norm_approx_sum: {grad_norm_approx_sum}, grad_norm_real: {grad_norm_real[i, trail_iter]}, Val acc = {acc_removal[0][i+1]} Test acc = {acc_removal[1][i+1]}, avg unlearn cost:{unlearn_cost[i+1]}, tot cost:{removal_times[i+1]}, num_retrain: {num_retrain}"
                )

        #######
        # retrain each round with graph
        #######
        # guo removal

        #######
        # retrain each round without graph

    logger.info("update cost: %.6fs" %
                (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" %
                (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" %
                (sum(removal_times[1:]) / (len(removal_times)-1)))
    if args.test_MI:
        MI_pred = torch.empty(
            (0, num_classes), dtype=torch.float).to(device)
        for i in range(args.num_batch_removes):
            nodes = del_nodes[
                node_idx_start
                + i * args.num_removes: node_idx_start
                + args.num_removes * (i + 1),
            ].T.tolist()
            MI_pred = torch.cat([MI_pred, predict(
                w_approx, X_new[nodes].to(device))], dim=0)
        MI_path = f"./analysis/{args.dataset}/{name}_MI/"
        check_dir(MI_path)
        np.save(f'{MI_path}/MI_pred_{name}_{args.num_batch_removes}_{args.num_removes}_{args.seed}.npy',
                MI_pred.cpu().numpy())
        exit()
    if args.replay:
        all_pred = predict_class(w_approx, X_new.to(device))
        res = all_pred == num_classes - 1
        res_deleted = res[del_nodes]
        logger.info(
            f"Replay test false: {res_deleted.sum()} {res_deleted.shape[0]} Total test false: {res.sum()} {res.shape[0]}")
        print(
            f"Replay test false: {res_deleted.sum()} {res_deleted.shape[0]} Total test false: {res.sum()} {res.shape[0]}")
        replay_path = f"./analysis/{args.dataset}/{name}_MI/"
        check_dir(replay_path)
        np.savez(f"{replay_path}/replay_pred_{name}_{args.seed}.npz", all_pred=all_pred.cpu().numpy(),
                 false_pred=res.cpu().numpy(), del_nodes=del_nodes)
        exit()
    np.save(tot_cost_path+f"_{node_idx_start}", removal_times)
    np.save(unlearn_cost_path+f"_{node_idx_start}", unlearn_cost)
    np.save(update_cost_path+f"_{node_idx_start}", update_cost)
    np.save(acc_path+f"_{node_idx_start}", acc_removal[1])
    np.savetxt(f_tot_cost, removal_times, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")

    # torch.save(
    #     {
    #         "grad_norm_approx": grad_norm_approx,
    #         "removal_times": removal_times,
    #         "acc_removal": acc_removal,
    #         "grad_norm_worst": grad_norm_worst,
    #         "grad_norm_real": grad_norm_real,
    #         "removal_times_graph_retrain": removal_times_graph_retrain,
    #         "acc_graph_retrain": acc_graph_retrain,
    #     },
    #     save_path,
    # )
