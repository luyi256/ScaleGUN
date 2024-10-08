import numpy as np
import propagation
import torch
from linear_unlearn_utils import *
from utils import *
import time
import gc
from datetime import datetime
import optuna
import logging
import pytz
import copy
from argparser import argparser
name = "edge"
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
    set_logger(args, logger, dt, name=name)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result/")
    logger.info(args)
    statistics_prefix = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}"
    tot_cost_path = f"{statistics_prefix}_cost"
    unlearn_cost_path = f"{statistics_prefix}_unlearn_cost"
    update_cost_path = f"{statistics_prefix}_update_cost"
    acc_path = f"{statistics_prefix}_acc"
    f_tot_cost = open(tot_cost_path+".txt", "ab")
    f_unlearn_cost = open(unlearn_cost_path+".txt", "ab")
    f_update_cost = open(update_cost_path+".txt", "ab")
    f_acc = open(acc_path+".txt", "ab")

    if args.dev > -1:
        device = torch.device("cuda:" + str(args.dev))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")

    start = time.perf_counter()
    data, edge_index = load_data(args.path, args.dataset)
    num_nodes = data.x.shape[0]
    num_edges = edge_index.shape[1]-num_nodes  # sub the self-loop
    if args.delta < 0:
        args.delta = 1/num_edges
        logger.info(f"delta: {args.delta}")

    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    column_sum_avg = feat.abs().sum(axis=0).mean()
    logger.info(f"column_sum_avg: {column_sum_avg}")
    args.rmax = args.rmax*column_sum_avg
    feat = feat.T
    origin_embedding = np.copy(feat.numpy())
    if args.dataset in ["ogbn-arxiv", "ogbn-products", "pokec"]:
        # transpose due to the discrepancy between Eigen and Python
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()
    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(del_path, args.dataset, origin_embedding,
                                  edge_index.T, args.prop_step, args.r, weights, args.num_threads, args.rmax)
    logger.info(f"initial prop time: {prop_time}")
    row = torch.from_numpy(edge_index[0]).long()
    deg = degree(row, feat.shape[1])
    del edge_index
    gc.collect()
    init_finish_time = time.perf_counter()

    X = torch.FloatTensor(origin_embedding.T)
    # logger.debug(
    #     f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
    data.y = data.y.long()
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1

    X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(
        data, X, args.train_mode, args.Y_binary)
    del X
    del data
    logger.info(
        "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
        )
    )
    train_size = X_train.shape[0]

    if args.compare_gnorm:
        b_std = 0
    else:
        b_std = args.std
    logger.info("--------------------------")
    logger.info("Training...")
    train_time = time.perf_counter()
    if args.train_mode == "ovr":
        b = b_std * torch.randn(feat_dim, num_classes).float().to(device)
    else:  # binary classification
        b = b_std * torch.randn(feat_dim).float().to(device)
    best_reg_lambda, best_lr, best_wd = args.lam, args.lr, args.wd
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    # logger.info(f"b:{b}")
    if args.train_mode == "ovr":
        w = ovr_lr_optimize(
            X_train,
            y_train,
            best_reg_lambda,
            weight=None,
            b=b,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            lr=best_lr,
            wd=best_wd,
            # X_val=X_val,
            # y_val=y_val,
        )
    else:
        w = lr_optimize(
            X_train,
            y_train,
            best_reg_lambda,
            b=b,
            num_steps=args.epochs,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            lr=args.lr,
            wd=args.wd,
        )
    train_finish_time = time.perf_counter()
    accum_un_grad_norm = 0.0
    opt_grad_norm = 0.0
    accum_un_grad_norm_arr = torch.zeros(args.num_batch_removes).float()
    accum_un_worst_grad_norm_arr = torch.zeros(args.num_batch_removes).float()
    if args.train_mode == "ovr":
        for k in range(y_train.size(1)):
            opt_grad_norm += lr_grad(w[:, k], X_train,
                                     y_train[:, k], best_reg_lambda).norm().cpu()
    else:
        grad_old = lr_grad(w, X_train, y_train, best_reg_lambda)
        opt_grad_norm = grad_old.norm().cpu()
    accum_un_worst_grad_norm = 0.0
    logger.info("init cost: %.6fs" % (init_finish_time - start))
    logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
    accum_un_worst_grad_norm_arr[0] = accum_un_grad_norm

    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    if args.train_mode == "ovr":
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
    else:
        val_acc = lr_eval(w, X_val, y_val)
        test_acc = lr_eval(w, X_test, y_test)
    logger.info("Validation accuracy: %.4f" % val_acc)
    logger.info("Test accuracy: %.4f" % test_acc)
    update_cost = [prop_time,]
    unlearn_cost = [train_finish_time - train_time,]
    tot_cost = [train_finish_time - train_time+prop_time,]
    acc_removal = [[val_acc.item()], [test_acc.item()]]
    logger.info("first train cost: %.6fs" % (train_finish_time - train_time))

    # remove
    logger.info("start to remove edges...")
    logger.info("*" * 20)

    ###########
    # budget for removal
    c_val = get_c(args.delta)
    if args.compare_gnorm or args.no_retrain:
        budget = 1e9
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
    edge_file = del_path + "/" + args.dataset + "/" + \
        args.dataset + f"_del_edges{args.del_postfix}.npy"
    del_edges = np.load(edge_file)
    logger.info(f"read del_edges from {edge_file}")
    if del_edges.shape[1] == 2:
        del_edges = del_edges.T

    w_approx = w.clone().detach().to(device)
    X_train_old = X_train.clone().detach().to(device)
    del X_train
    del X_val
    del X_test
    gc.collect()

    for i in range(args.num_batch_removes):
        edges = del_edges[
            :,
            edge_idx_start
            + i * args.num_removes: edge_idx_start
            + args.num_removes * (i + 1),
        ].T.tolist()
        return_time = g.UpdateEdges(
            edges, origin_embedding, args.num_threads, args.rmax)
        update_cost.append(return_time)
        residue = np.zeros(feat_dim)
        g.GetResidueSum(residue)
        column_sum_norm = LA.norm(residue, 2)
        X_new = torch.FloatTensor(origin_embedding.T)
        X_train_new = X_new[train_mask].to(device)
        update_finish_time = time.perf_counter()
        K = get_K_matrix(X_train_new)
        spec_norm = sqrt_spectral_norm(K)
        if args.compare_gnorm:
            groundtruth = np.copy(feat.numpy())
            g.PowerMethod(groundtruth)
            X_groundtruth = torch.FloatTensor(groundtruth.T)
            X_groundtruth_train = X_groundtruth[train_mask].to(device)
        if args.train_mode == "ovr":
            for k in range(y_train.size(1)):
                y_rem = y_train[:, k]
                H_inv = lr_hessian_inv(
                    w_approx[:, k], X_train_new, y_rem, best_reg_lambda
                )
                grad_old = lr_grad(
                    w_approx[:, k], X_train_old, y_rem, best_reg_lambda)
                grad_new = lr_grad(
                    w_approx[:, k], X_train_new, y_rem, best_reg_lambda)
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                w_approx[:, k] += Delta
                Delta_p = X_train_new.mv(Delta)
                # here, grad_norm_approx store the norm induced by unlearning, that is, the second term of data-dependent bound
                grad_norm_approx[i] += (Delta.norm() *
                                        Delta_p.norm() * spec_norm * gamma).cpu()
                if args.compare_gnorm:
                    grad_gt_k = lr_grad(
                        w_approx[:, k], X_groundtruth_train, y_rem, best_reg_lambda)
                    grad_norm_real[i] += grad_gt_k.norm().cpu()
            if args.compare_gnorm:
                approximation_worst_norm, unlearning_worst_norm = get_worst_Gbound_edge(
                    deg[edges[0][0]], deg[edges[0][1]], train_size, feat_dim, args.lam, args.rmax, num_nodes, args.prop_step)
                accum_un_worst_grad_norm += unlearning_worst_norm * \
                    y_train.size(1)
                grad_norm_worst[i] = y_train.size(
                    1)*approximation_worst_norm+accum_un_worst_grad_norm
                accum_un_worst_grad_norm_arr[i] = accum_un_worst_grad_norm
            approximation_norm = column_sum_norm*2*y_train.shape[1]
            accum_un_grad_norm += grad_norm_approx[i]
            accum_un_grad_norm_arr[i] = accum_un_grad_norm
            grad_norm_approx[i] = approximation_norm + accum_un_grad_norm
            if grad_norm_approx[i] > budget:
                logger.info(
                    f"The {i}-th removal, grad_norm_approx: {grad_norm_approx[i]}, approximation_norm: {approximation_norm}, retraining..."
                )
                accum_un_grad_norm = 0.0
                b = b_std * torch.randn(feat_dim,
                                        num_classes).float().to(device)
                w_approx = ovr_lr_optimize(
                    X_train_new,
                    y_train,
                    best_reg_lambda,
                    weight=None,
                    b=b,
                    verbose=args.verbose,
                    opt_choice=args.optimizer,
                    lr=best_lr,
                    wd=best_wd,
                    # X_val=X_val,
                    # y_val=y_val,
                )
                num_retrain += 1
            remove_finish_time = time.perf_counter()
            X_val_new = X_new[val_mask].to(device)
            acc_removal[0].append(ovr_lr_eval(
                w_approx, X_val_new, y_val).item())
            X_test_new = X_new[test_mask].to(device)
            acc_removal[1].append(ovr_lr_eval(
                w_approx, X_test_new, y_test).item())
        else:
            X_train_new = X_new[train_mask].to(device)
            y_train = y_train.to(device)
            H_inv = lr_hessian_inv(w_approx, X_train_new, y_train, args.lam)
            # grad_i should be the difference
            grad_old = lr_grad(w_approx, X_train_old, y_train, args.lam)
            grad_new = lr_grad(w_approx, X_train_new, y_train, args.lam)
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_train_new.mv(Delta)
            w_approx += Delta
            grad_norm_approx[i] += (
                Delta.norm() * Delta_p.norm() * spec_norm * gamma
            ).cpu()
            grad_old = lr_grad(w_approx, X_train_new, y_train, args.lam)
            if args.compare_gnorm:
                grad_norm_real[i] = (
                    lr_grad(w_approx, X_groundtruth_train,
                            y_train, args.lam).norm().cpu()
                )
                approximation_worst_norm, unlearning_worst_norm = get_worst_Gbound_edge(
                    deg[edges[0][0]], deg[edges[0][1]], train_size, feat_dim, args.lam, args.rmax, num_nodes, args.prop_step)
                accum_un_worst_grad_norm += unlearning_worst_norm
                grad_norm_worst[i] = accum_un_worst_grad_norm + \
                    approximation_worst_norm
                approximation_norm = column_sum_norm*2
                accum_un_grad_norm += grad_norm_approx[i]
                grad_norm_approx[i] = approximation_norm + accum_un_grad_norm
            if grad_norm_approx[i] > budget:
                # retrain the model
                accum_un_grad_norm = 0
                b = b_std * torch.randn(X_new.size(1)).float().to(device)
                w_approx = lr_optimize(
                    X_train_new,
                    y_train,
                    args.lam,
                    b=b,
                    num_steps=args.epochs,
                    verbose=False,
                    opt_choice=args.optimizer,
                    lr=args.lr,
                    wd=args.wd,
                )
                num_retrain += 1

            remove_finish_time = time.perf_counter()
            acc_removal[0].append(lr_eval(w_approx, X_val_new, y_val).item())
            acc_removal[1].append(lr_eval(w_approx, X_test_new, y_test).item())
        unlearn_cost.append(remove_finish_time - update_finish_time)
        tot_cost.append(remove_finish_time - update_finish_time+return_time)
        X_train_old = X_train_new.clone().detach()
        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Edge del = {edges[0]}, grad_norm_approx = {grad_norm_approx[i]}, Val acc = {acc_removal[0][i+1]} Test acc = {acc_removal[1][i+1]}, avg update cost: {update_cost[i+1]}, avg unlearn cost:{unlearn_cost[i+1]}, avg tot cost:{tot_cost[i+1]}, num_retrain: {num_retrain}"
            )
    end_time = time.perf_counter()

    logger.info("update cost: %.6fs" %
                (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" %
                (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" % (sum(tot_cost[1:]) / (len(tot_cost)-1)))
    logger.info("tot cost: %.6fs" % (end_time - start_time))
    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")
    if args.compare_gnorm:
        grad_norm_approx = grad_norm_approx.cpu().numpy()
        grad_norm_real = grad_norm_real.cpu().numpy()
        grad_norm_worst = grad_norm_worst.cpu().numpy()
        accum_un_grad_norm_arr = accum_un_grad_norm_arr.cpu().numpy()
        accum_un_worst_grad_norm_arr = accum_un_worst_grad_norm_arr.cpu().numpy()
        np.savetxt(statistics_prefix+"_approx.txt",
                   grad_norm_approx, delimiter=",")
        np.savetxt(statistics_prefix+"_real.txt",
                   grad_norm_real, delimiter=",")
        np.savetxt(statistics_prefix+"_worst.txt",
                   grad_norm_worst, delimiter=",")
        np.savetxt(statistics_prefix+"_app_real.txt",
                   accum_un_grad_norm_arr, delimiter=",")
        np.savetxt(statistics_prefix+"_app_worst.txt",
                   accum_un_worst_grad_norm_arr, delimiter=",")


if __name__ == "__main__":
    main()
