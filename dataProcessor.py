from torch_geometric.datasets import Planetoid, Amazon
import argparse
import numpy as np
import sklearn.preprocessing
from torch_geometric.utils import to_undirected
import struct
from utils import *
from ogb.nodeproppred import PygNodePropPredDataset
import random
import gc
from linear_unlearn_utils import *
from numpy.linalg import norm
from torch_geometric.utils import add_remaining_self_loops
import math
import time
from torch_geometric.transforms import ToUndirected
from Hetero_dataset import HeteroDataset
from LINKX_dataset import LINKXDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def common(path, dataset, result_path, normalized_dim):
    start = time.time()
    if normalized_dim == "column":
        dim = 0
    else:
        dim = 1
    print("normalized dim:", dim)

    load_time = time.time()
    print("load time:", load_time - start)

    # print("save del edges.....")
    data, _ = load_data(path, dataset)
    # perm = torch.from_numpy(np.random.permutation(data.edge_index.shape[1])).to(device)
    # data.edge_index=data.edge_index.to(device)
    # edge_index=data.edge_index[:,perm]
    # np.save(
    #     f"{result_path}/{dataset}/{dataset}_del_edges.npy", edge_index.cpu().numpy()
    # )
    #
    train_idx = torch.arange(data.x.shape[0])[data.train_mask]
    perm = torch.from_numpy(np.random.permutation(train_idx.shape[0]))
    np.save(f"{result_path}/{dataset}/{dataset}_del_nodes.npy",
            train_idx[perm])
    perm = perm[: 2 * args.num_del_edges]
    edge_index = data.edge_index[:, perm]
    edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    cnt = 0
    for e in range(edge_index.shape[1]):
        if edge_mask[e] == False:
            continue
        if edge_index[0][e] == edge_index[1][e]:
            edge_mask[e] = False
            continue
        source_idx = edge_index[0][e].item()
        dst_idx = edge_index[1][e].item()
        # find the other undirected edge
        rev_edge_idx = (
            torch.logical_and(
                edge_index[0] == dst_idx, edge_index[1] == source_idx)
            .nonzero()
            .squeeze(-1)
        )
        if rev_edge_idx.shape[0] > 0:
            edge_mask[rev_edge_idx] = False
        cnt += 1
        if cnt >= args.num_del_edges and args.num_del_edges != -1:
            break
    edge_mask = edge_mask[: e + 1]
    edge_index = edge_index[:, : e + 1]
    edge_index = edge_index[:, edge_mask]
    check_dir(f"{result_path}/{dataset}")
    np.save(
        f"{result_path}/{dataset}/{dataset}_del_edges.npy", edge_index.cpu().numpy()
    )
    # np.savetxt(
    #     f"{result_path}/{dataset}/{dataset}_del_edges.txt",
    #     edge_index.cpu().numpy(),
    #     fmt="%d",
    # )

    del_time = time.time()
    print("del time:", del_time - load_time)
    if args.del_only:
        return

    # print("save edges.....")

    # f = open(
    #     f"{result_path}/{dataset}/{dataset}.edges", "wb", buffering=1024 * 1024 * 128
    # )
    # for i in range(edge_index.shape[1]):
    #     m = struct.pack("II", edge_index[0][i], edge_index[1][i])
    #     f.write(m)
    # f.close()

    # edges_time=time.time()
    # print("edges time:",edges_time-label_time)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    print("save attr.....")
    num_edges = edge_index.shape[1]
    f = open(f"{result_path}/{dataset}/{dataset}.attr", "w")
    f.write("%d %d %d" % (data.num_nodes, num_edges, data.num_features))
    print("num_nodes:", data.num_nodes)
    print("num_edges:", num_edges)
    print("num_features:", data.num_features)
    f.close()

    print("finish saving data")


# def common_random(path, dataset, result_path):
#     if dataset in ["cora"]:
#         data = Planetoid(root="./data/", name=dataset, split="full")
#         data = data[0]

#         print("save features.....")
#         feat = data.x.numpy()
#         feat = np.array(feat, dtype=np.float64)
#         scaler = sklearn.preprocessing.StandardScaler()
#         scaler.fit(feat)
#         feat = scaler.transform(feat)
#         for i in range(feat.shape[1]):
#             # print sum
#             print(f"dimension: {i}, sum:{np.sum(feat[:, i])}")
#         print(feat[:, 0])
#         check_dir(f"{result_path}/{dataset}")
#         np.save(f"{result_path}/{dataset}/{dataset}_feat.npy", feat)

#         print("save labels.....")
#         train_idx = [index for index, value in enumerate(data.train_mask) if value]
#         val_idx = [index for index, value in enumerate(data.val_mask) if value]
#         test_idx = [index for index, value in enumerate(data.test_mask) if value]
#         all_idx = train_idx + val_idx + test_idx
#         total_num = len(all_idx)
#         np.random.shuffle(all_idx)
#         train_idx = np.array(all_idx[: math.floor(total_num * 0.7)], dtype=np.int32)
#         val_idx = np.array(
#             all_idx[math.floor(total_num * 0.7) : math.floor(total_num * 0.8)],
#             dtype=np.int32,
#         )
#         test_idx = np.array(all_idx[math.floor(total_num * 0.8) :], dtype=np.int32)

#         labels = data.y
#         train_labels = labels[train_idx].numpy().astype(np.int32)
#         val_labels = labels[val_idx].numpy().astype(np.int32)
#         test_labels = labels[test_idx].numpy().astype(np.int32)

#         np.savez(
#             f"{result_path}{dataset}/{dataset}_labels_random.npz",
#             train_idx=train_idx,
#             val_idx=val_idx,
#             test_idx=test_idx,
#             train_labels=train_labels,
#             val_labels=val_labels,
#             test_labels=test_labels,
#         )
#         data.edge_index = to_undirected(data.edge_index, data.num_nodes)
#         f = open(f"{result_path}/{dataset}/{dataset}.edges", "wb")
#         for i in range(data.edge_index.shape[1]):
#             m = struct.pack("II", data.edge_index[0][i], data.edge_index[1][i])
#             f.write(m)
#         f.close()

#         f = open(f"{result_path}/{dataset}/{dataset}.attr", "w")
#         f.write("%d %d %d" % (data.num_nodes, data.num_edges, data.num_features))
#         f.close()

#         print("finish saving data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="penn94")
    parser.add_argument("--path", type=str, default="/mnt_1/lu_yi/data/")
    parser.add_argument(
        "--result_path", type=str, default="/mnt_1/lu_yi/data/unlearning_data/"
    )
    parser.add_argument("--feature_only", type=bool, default=False)
    parser.add_argument("--del_only", default=False, action="store_true")
    parser.add_argument("--normalized_dim", type=str, default="column")
    parser.add_argument(
        "--num_del_edges",
        type=int,
        default=10000,
        help="the number of edges to be removed, -1 for all edges",
    )
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--attr", default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    common(args.path, args.dataset, args.result_path,
           args.normalized_dim)
