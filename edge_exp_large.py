import numpy as np
import argparse
import propagation
import torch
from linear_unlearn_utils import *
from utils import *
import torch.nn.functional as F
import time
import gc
from datetime import datetime
import optuna
import logging
import pytz
import psutil
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon, LINKXDataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_remaining_self_loops
from Hetero_dataset import HeteroDataset
from LINKX_dataset import LINKXDataset
from argparser import argparser
name="edge_large"
torch.set_printoptions(precision=10)
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(message)s", handlers=[]
)
setup_logger(name)
setup_unlearn_logger(name)

def main():
    args = argparser()
    seed_everything(seeds[args.seed])
    tz = pytz.timezone("Asia/Shanghai")
    dt = datetime.now(tz).strftime("%m%d_%H%M")
    set_logger(args, logger,dt,name=name)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result/")
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model/")
    logger.info(args)
    tot_cost_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_cost"
    unlearn_cost_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_unlearn_cost"
    update_cost_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_update_cost"
    acc_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_acc"
    log_prefix=f"{args.analysis_path}/{args.dataset}/{name}_model/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_edge_idx_{args.edge_idx_start}"
    if args.dev > -1:
        device = torch.device("cuda:" + str(args.dev))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")

    start = time.perf_counter()
    data,edge_index=load_data(args.path, args.dataset)

    weights=get_prop_weight(args.weight_mode, args.prop_step,args.decay)
    
    data.y = data.y.long()
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    y_train, y_val, y_test,train_mask, val_mask, test_mask=get_split_large(data,args.train_mode,args.Y_binary,dataset_name=args.dataset)

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    del data
    column_sum_avg=feat.abs().sum(axis=0).mean()
    logger.info(f"column_sum_avg: {column_sum_avg}")
    args.rmax=args.rmax*column_sum_avg
    logger.debug(f"feat: {feat[:5,:5]}")
    feat = feat.T
    origin_embedding = np.copy(feat.numpy())
    if args.dataset in ["ogbn-arxiv", "ogbn-products","pokec"]:
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage = mem_info.rss
    logger.info(f"Memory Usage before prop: {memory_usage/1024/1024/1024} GB")

    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(del_path,args.dataset,origin_embedding,edge_index.T,args.prop_step,args.r,weights,args.num_threads,args.rmax)
    logger.info(f"initial prop time: {prop_time}" )

    mem_info = process.memory_info()
    memory_usage = mem_info.rss
    logger.info(f"After prop, Memory Usage: {memory_usage/1024/1024/1024} GB")

    del edge_index
    gc.collect()
    init_finish_time = time.perf_counter()
    logger.info("init cost: %.6fs" % (init_finish_time - start))

    # if origin_embedding.shape[0] == feat_dim:  # feature dimension
    X = torch.FloatTensor(origin_embedding.T)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    logger.debug(f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
    del X
    logger.info(
        "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
        )
    )

    assert args.noise_mode == "data"

    if args.compare_gnorm:
        b_std = 0
    else:
        if args.noise_mode == "data":
            b_std = args.std
        elif args.noise_mode == "worst":
            b_std = args.std  # change to worst case sigma
        else:
            raise ("Error: Not supported noise model.")

    weight = None
    logger.info("--------------------------")
    logger.info("Training...")
    train_time = time.perf_counter()
    if args.train_mode == "ovr":
        b = b_std * torch.randn(feat_dim, num_classes).float().to(device)
    else:  # binary classification
        b = b_std * torch.randn(feat_dim).float().to(device)
    best_reg_lambda, best_lr, best_wd = args.lam, args.lr, args.wd
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    seed_everything(seeds[args.seed])
    w = train(X_train,y_train,args,best_reg_lambda,best_lr,best_wd,b,device,weight=weight,X_val=X_val,y_val=y_val)
    train_finish_time = time.perf_counter()
    
    update_cost = []
    unlearn_cost = []
    tot_cost = []
    acc_removal = [[], []]
    update_cost.append(prop_time)
    unlearn_cost.append(train_finish_time - train_time)
    logger.info("first train cost: %.6fs" % (train_finish_time - train_time))
    tot_cost.append(train_finish_time - train_time+prop_time)
    mem_info = process.memory_info()
    memory_usage = mem_info.rss
    logger.info(f"Train Memory Usage: {memory_usage/1024/1024/1024} GB")
    
    opt_grad_norm = 0.0
    if args.train_mode == "ovr":
        grad_old = lr_grad_handloader(w,X_train,y_train,args.train_batch,best_reg_lambda,feat_dim,num_classes,)
        for k in range(num_classes):
            opt_grad_norm += grad_old[k].norm().cpu()
    else:
        grad_old = lr_grad(w, X_train, y_train, best_reg_lambda)
        opt_grad_norm = grad_old.norm().cpu()
    logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
    
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    if args.train_mode == "ovr":
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
    else:
        val_acc = lr_eval(w, X_val, y_val)
        test_acc = lr_eval(w, X_test, y_test)
    print("Validation accuracy: %.4f" % val_acc)
    print("Test accuracy: %.4f" % test_acc)
    logger.info("Validation accuracy: %.4f" % val_acc)
    logger.info("Test accuracy: %.4f" % test_acc)
    acc_removal[0].append(val_acc.item())
    acc_removal[1].append(test_acc.item())
    checkpt_file=log_prefix+"_init.pt"
    torch.save(w, checkpt_file)
    # exit(-1)

    # remove
    logger.info("start to remove edges...")
    logger.info("*" * 20)

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
    logger.debug(f"Budget: {budget}")

    start_time = time.perf_counter()
    grad_norm_approx = torch.zeros(args.num_batch_removes).float()
    grad_norm_worst = torch.zeros(args.num_batch_removes).float()
    grad_norm_real = torch.zeros(args.num_batch_removes).float()
    
    grad_norm_approx_sum = 0.0
    num_retrain = 0

    # obtain delete edges
    edge_idx_start = args.edge_idx_start
    edge_file = del_path + "/" + args.dataset + "/" + args.dataset + f"_del_edges{args.del_postfix}.npy"
    del_edges = np.load(edge_file)
    if del_edges.shape[1]==2:
        del_edges=del_edges.T

    w_approx = w.clone().detach().to(device)
    del X_train
    del X_val
    del X_test
    gc.collect()

    for i in range(args.num_batch_removes):
        edges = del_edges[
            :,
            edge_idx_start
            + i * args.num_removes : edge_idx_start
            + args.num_removes * (i + 1),
        ].T.tolist()
        return_time = g.UpdateEdges(edges, origin_embedding, args.num_threads, args.rmax)
        update_finish_time = time.perf_counter()
        update_cost.append(return_time)
        X_new = torch.FloatTensor(origin_embedding.T)
        X_new_train = X_new[train_mask]

        if args.train_mode == "ovr":
            spec_norm, H_inv, grad_new = unlearn_step1(
                w_approx,
                X_new_train,
                y_train,
                best_reg_lambda,
                args.train_batch,
                feat_dim,
                num_classes,
                device,
            )
            Delta = torch.bmm(H_inv, (grad_old - grad_new).unsqueeze(2)).squeeze(2).t()
            w_approx = w_approx + Delta
            Delta_p, grad_old = unlearn_step2(w_approx,X_new_train,y_train,best_reg_lambda,Delta,args.train_batch,feat_dim,num_classes,device)
            grad_norm_approx[i] += np.sum(
                [
                    (Delta[:,k].norm() * Delta_p[:,k].norm() * spec_norm * gamma).cpu()
                    for k in range(num_classes)
                ]
            )
            if args.compare_gnorm:
                for k in range(num_classes):
                    grad_norm_real[i] += grad_old[k].norm().cpu()
                grad_norm_worst[i] += get_worst_Gbound_edge()
            if grad_norm_approx_sum + grad_norm_approx[i] > budget:
                logger.info(
                    f"grad_norm_approx: {grad_norm_approx_sum + grad_norm_approx[i]}, retraining..."
                )
                grad_norm_approx_sum = 0.0
                b = b_std * torch.randn(feat_dim, num_classes).float().to(device)
                X_val_new = X_new[val_mask].to(device)
                seed_everything(seeds[args.seed])
                w_approx = ovr_lr_optimize_handloader(
                    X_new_train,
                    y_train,
                    best_reg_lambda,
                    batch_size=args.train_batch,
                    init_method=args.init_method,
                    weight=weight,
                    b=b,
                    num_steps=args.epochs,
                    verbose=False,
                    opt_choice=args.optimizer,
                    patience=args.patience,
                    lr=best_lr,
                    wd=best_wd,
                    X_val=X_val_new,
                    y_val=y_val,
                )
                grad_old=lr_grad_handloader(w_approx,X_new_train,y_train,args.train_batch,best_reg_lambda,feat_dim,num_classes,)
                del X_val_new
                num_retrain += 1
            else:
                grad_norm_approx_sum += grad_norm_approx[i]

            remove_finish_time = time.perf_counter()
            X_val_new = X_new[val_mask].to(device)
            acc_removal[0].append(ovr_lr_eval(w_approx, X_val_new, y_val).item())
            X_test_new = X_new[test_mask].to(device)
            acc_removal[1].append(ovr_lr_eval(w_approx, X_test_new, y_test).item())
            del X_new, X_new_train, X_val_new, X_test_new,H_inv,Delta,Delta_p,grad_new
        else:
            X_rem = X_new[train_mask].to(device)
            y_rem = y_train.to(device)
            H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, args.lam)
            # grad_i should be the difference
            grad_new = lr_grad(w_approx, X_rem, y_rem, args.lam)
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_rem.mv(Delta)
            w_approx += Delta
            grad_norm_approx[i] += (
                Delta.norm() * Delta_p.norm() * spec_norm * gamma
            ).cpu()
            grad_old = lr_grad(w_approx, X_rem, y_rem, args.lam)
            if args.compare_gnorm:
                grad_norm_real[i] += (
                    lr_grad(w_approx, X_rem, y_rem, args.lam).norm().cpu()
                )
                grad_norm_worst[i] += get_worst_Gbound_edge(
                    args.lam, X_rem.shape[0], args.prop_step
                )
            if grad_norm_approx_sum + grad_norm_approx[i] > budget:
                # retrain the model
                grad_norm_approx_sum = 0
                b = b_std * torch.randn(X_new.size(1)).float().to(device)
                w_approx = lr_optimize(
                    X_rem,
                    y_rem,
                    args.lam,
                    b=b,
                    num_steps=args.epochs,
                    verbose=False,
                    opt_choice=args.optimizer,
                    lr=args.lr,
                    wd=args.wd,
                )
                num_retrain += 1
            else:
                grad_norm_approx_sum += grad_norm_approx[i]
            remove_finish_time = time.perf_counter()
            acc_removal[0].append(lr_eval(w_approx, X_val_new, y_val).item())
            acc_removal[1].append(lr_eval(w_approx, X_test_new, y_test).item())
        unlearn_cost.append(remove_finish_time - update_finish_time)
        tot_cost.append(remove_finish_time - update_finish_time + return_time)
        checkpt_file=log_prefix+f"_{i}.pt"
        torch.save(w_approx, checkpt_file)
        embed_file=log_prefix+f"_{i}"
        # np.save(embed_file, X_new.cpu().numpy())
        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Edge del = {edges[0]}, grad_norm_approx_sum: {grad_norm_approx_sum},  grad_norm_real: {grad_norm_real[i]}, Val acc = {acc_removal[0][i+1]} Test acc = {acc_removal[1][i+1]}, avg update cost: {update_cost[i+1]}, avg unlearn cost:{unlearn_cost[i+1]}, avg tot cost:{tot_cost[i+1]}, num_retrain: {num_retrain}"
            )
            np.save(acc_path, acc_removal[1])
            np.save(tot_cost_path, tot_cost)
            np.save(unlearn_cost_path, unlearn_cost)
            np.save(update_cost_path, update_cost)
            mem_info = process.memory_info()
            memory_usage = mem_info.rss
        if i==0:
            logger.info(f"Unlearn Memory Usage: {memory_usage/1024/1024/1024} GB")
        
    end_time = time.perf_counter()
    logger.info("tot cost: %.6fs" % (end_time - start_time))

    logger.info("update cost: %.6fs" % (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" % (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" % (sum(tot_cost[1:]) / (len(tot_cost)-1)))
    np.savetxt(acc_path+".txt", acc_removal[1])
    np.savetxt(tot_cost_path+".txt", tot_cost)
    np.savetxt(unlearn_cost_path+".txt", unlearn_cost)
    np.savetxt(update_cost_path+".txt", update_cost)

if __name__ == "__main__":
    main()