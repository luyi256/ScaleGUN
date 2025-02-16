import numpy as np
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
from argparser import argparser
name = "node_feat_test"
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
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model/")
    logger.info(args)
    statistics_prefix = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_removal_mode_{args.removal_mode}"
    tot_cost_path = f"{statistics_prefix}_cost"
    unlearn_cost_path = f"{statistics_prefix}_unlearn_cost"
    update_cost_path = f"{statistics_prefix}_update_cost"
    acc_path = f"{statistics_prefix}_acc"
    f_tot_cost = open(tot_cost_path+".txt", "a")
    f_unlearn_cost = open(unlearn_cost_path+".txt", "a")
    f_update_cost = open(update_cost_path+".txt", "a")
    f_acc = open(acc_path+".txt", "a")
    if args.dev > -1:
        device = torch.device("cuda:" + str(args.dev))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")

    start = time.perf_counter()
    data, edge_index = load_data(args.path, args.dataset)
    num_nodes = data.x.shape[0]
    # num_edges = edge_index.shape[1]-num_nodes  # sub the self-loop
    if args.delta < 0:
        args.delta = 1/num_nodes
        logger.info(f"delta: {args.delta}")

    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)

    data.y = data.y.long()
    data_y = data.y
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1

    del_path = os.path.join(args.path, args.del_path_suffix)
    node_idx_start = args.edge_idx_start
    node_file = (
        del_path + "/" + args.dataset + "/" + args.dataset + "_del_nodes.npy"
    )
    del_nodes = np.load(node_file)
    np.random.shuffle(del_nodes)
    del_nodes = del_nodes[node_idx_start:node_idx_start +
                          args.num_removes*args.num_batch_removes]
    feat = preprocess_data(data.x, axis_num=args.axis_num)
    if args.replay:
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
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split_large(
        data, args.train_mode, args.Y_binary, dataset_name=args.dataset)
    del data
    # column_sum_avg = feat.abs().sum(axis=0).mean()
    # logger.info(f"column_sum_avg: {column_sum_avg}")
    # args.rmax = args.rmax*column_sum_avg
    logger.debug(f"feat: {feat[:5,:5]}")
    feat = feat.T
    origin_embedding = np.copy(feat.numpy())
    if args.dataset in ["ogbn-arxiv", "ogbn-products", "pokec"]:
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()

    # process = psutil.Process(os.getpid())
    # mem_info = process.memory_info()
    # memory_usage = mem_info.rss
    # logger.info(f"Memory Usage before prop: {memory_usage/1024/1024/1024} GB")

    prop_time = g.init_push_graph(del_path, args.dataset, origin_embedding,
                                  edge_index.T, args.prop_step, args.r, weights, args.num_threads, args.rmax, feat_dim)
    logger.info(f"initial prop time: {prop_time}")
    # ! test with pre-generated embedding
    # prop_time = 0
    # origin_embedding = np.load(
    #     f"{del_path}/{args.dataset}/ogbn-papers100M_embedding.npy")
    # !

    # mem_info = process.memory_info()
    # memory_usage = mem_info.rss
    # logger.info(f"After prop, Memory Usage: {memory_usage/1024/1024/1024} GB")

    # np.save(
    #     f"{del_path}/{args.dataset}/{args.dataset}_{args.rmax}_embedding.npy", origin_embedding)
    # groundtruth=np.copy(feat.numpy())
    # g.PowerMethod(groundtruth)
    # check_propagation(groundtruth,origin_embedding)
    del edge_index
    gc.collect()
    init_finish_time = time.perf_counter()
    logger.info("init cost: %.6fs" % (init_finish_time - start))

    # if origin_embedding.shape[0] == feat_dim:  # feature dimension
    X = torch.FloatTensor(origin_embedding.T)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    # embed_file = log_prefix+"_init"
    # np.save(embed_file, X.numpy())
    logger.debug(
        f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
    del X
    logger.info(
        "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
        )
    )
    data_prepare_time = time.perf_counter()

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
    w = train(X_train, y_train, args, best_reg_lambda, best_lr,
              best_wd, b, device, weight=weight, X_val=X_val, y_val=y_val)
    train_finish_time = time.perf_counter()

    update_cost = []
    unlearn_cost = []
    tot_cost = []
    acc_removal = [[], []]
    update_cost.append(prop_time)
    unlearn_cost.append(train_finish_time - train_time)
    logger.info("first train cost: %.6fs" % (train_finish_time - train_time))
    tot_cost.append(train_finish_time - train_time+prop_time)
    # mem_info = process.memory_info()
    # memory_usage = mem_info.rss
    # logger.info(f"Train Memory Usage: {memory_usage/1024/1024/1024} GB")
    accum_un_grad_norm = 0.0
    # only the error caused by unlearning
    accum_un_grad_norm_arr = torch.zeros(args.num_batch_removes).float()
    accum_un_worst_grad_norm_arr = torch.zeros(args.num_batch_removes).float()
    accum_un_worst_grad_norm = 0.0
    opt_grad_norm = 0.0
    if args.train_mode == "ovr":
        grad_old = lr_grad_handloader(
            w, X_train, y_train, args.train_batch, best_reg_lambda, feat_dim, num_classes,)
        for k in range(num_classes):
            opt_grad_norm += grad_old[k].norm().cpu()
    else:
        grad_old = lr_grad(w, X_train, y_train, best_reg_lambda)
        opt_grad_norm = grad_old.norm().cpu()
    accum_un_worst_grad_norm = 0.0
    logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
    accum_un_grad_norm_arr[0] = accum_un_grad_norm
    accum_un_worst_grad_norm_arr[0] = accum_un_grad_norm
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
    # np.save(f"{del_path}/{args.dataset}/{args.dataset}_{args.weight_mode}_{args.rmax}_embedding.npy", origin_embedding)
    # checkpt_file = log_prefix+"_init.pt"
    # torch.save(w, checkpt_file)
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

    # grad_norm_approx_sum = 0.0
    num_retrain = 0

    # obtain delete nodes
    w_approx = w.clone().detach().to(device)
    del X_train
    del X_val
    del X_test
    gc.collect()
    MI_pred = torch.empty((0, num_classes), dtype=torch.float).to(device)

    for i in range(args.num_batch_removes):
        nodes = del_nodes[
            node_idx_start
            + i * args.num_removes: node_idx_start
            + args.num_removes * (i + 1),
        ].T.tolist()
        if args.removal_mode == "node":
            return_time = g.UpdateNodes(
                nodes, origin_embedding, args.num_threads, args.rmax)
            if args.replay:
                origin_embedding[:, nodes] = 0.0
        else:
            return_time = g.UpdateFeatures(
                nodes, origin_embedding, args.num_threads, args.rmax)
        train_mask[nodes] = False
        residue = np.zeros(feat_dim)
        g.GetResidueSum(residue)
        column_sum_norm = LA.norm(residue, 2)
        update_cost.append(return_time)
        X_new = torch.FloatTensor(origin_embedding.T)
        X_new_train = X_new[train_mask]
        train_size = train_mask.sum().item()
        y_train = F.one_hot(data_y[train_mask],
                            num_classes=data_y.max().item()+1) * 2 - 1
        y_train = y_train.float().to(device)
        update_finish_time = time.perf_counter()
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
            Delta = torch.bmm(
                H_inv, (grad_old - grad_new).unsqueeze(2)).squeeze(2).t()
            w_approx = w_approx + Delta
            Delta_p, grad_old = unlearn_step2(
                w_approx, X_new_train, y_train, best_reg_lambda, Delta, args.train_batch, feat_dim, num_classes, device)
            grad_norm_approx[i] += np.sum(
                [
                    (Delta[:, k].norm() * Delta_p[:, k].norm()
                     * spec_norm * gamma).cpu()
                    for k in range(num_classes)
                ]
            )
            approximation_norm = column_sum_norm*2*y_train.shape[1]
            # logger.debug(f"approximation_norm: {approximation_norm}")
            accum_un_grad_norm += grad_norm_approx[i]
            grad_norm_approx[i] = approximation_norm + accum_un_grad_norm
            # todo: add compare_gnorm
            # if args.compare_gnorm:
            #     for k in range(num_classes):
            #         grad_norm_real[i] += grad_old[k].norm().cpu()
            # grad_norm_worst[i] += get_worst_Gbound_edge()
            if grad_norm_approx[i] > budget:
                logger.info(
                    f"grad_norm_approx: {grad_norm_approx[i]}, retraining..."
                )
                accum_un_grad_norm = 0.0
                b = b_std * torch.randn(feat_dim,
                                        num_classes).float().to(device)
                X_val_new = X_new[val_mask].to(device)
                # seed_everything(seeds[args.seed])
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
                grad_old = lr_grad_handloader(
                    w_approx, X_new_train, y_train, args.train_batch, best_reg_lambda, feat_dim, num_classes,)
                # for k in range(num_classes):
                #     accum_un_grad_norm += grad_old[k].norm().cpu()
                del X_val_new
                num_retrain += 1

            remove_finish_time = time.perf_counter()
            X_val_new = X_new[val_mask].to(device)
            acc_removal[0].append(ovr_lr_eval(
                w_approx, X_val_new, y_val).item())
            X_test_new = X_new[test_mask].to(device)
            acc_removal[1].append(ovr_lr_eval(
                w_approx, X_test_new, y_test).item())
            if i == args.num_batch_removes-1:
                pass
            else:
                del X_new, X_new_train, X_val_new, X_test_new, H_inv, Delta, Delta_p, grad_new
        # else:
        #     X_rem = X_new[train_mask].to(device)
        #     y_rem = y_train.to(device)
        #     H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, args.lam)
        #     # grad_i should be the difference
        #     grad_new = lr_grad(w_approx, X_rem, y_rem, args.lam)
        #     grad_i = grad_old - grad_new
        #     Delta = H_inv.mv(grad_i)
        #     Delta_p = X_rem.mv(Delta)
        #     w_approx += Delta
        #     grad_norm_approx[i] += (
        #         Delta.norm() * Delta_p.norm() * spec_norm * gamma
        #     ).cpu()
        #     grad_old = lr_grad(w_approx, X_rem, y_rem, args.lam)
        #     if args.compare_gnorm:
        #         grad_norm_real[i] += (
        #             lr_grad(w_approx, X_rem, y_rem, args.lam).norm().cpu()
        #         )
        #         grad_norm_worst[i] += get_worst_Gbound_edge(
        #             args.lam, X_rem.shape[0], args.prop_step
        #         )
        #     if grad_norm_approx_sum + grad_norm_approx[i] > budget:
        #         # retrain the model
        #         grad_norm_approx_sum = 0
        #         b = b_std * torch.randn(X_new.size(1)).float().to(device)
        #         w_approx = lr_optimize(
        #             X_rem,
        #             y_rem,
        #             args.lam,
        #             b=b,
        #             num_steps=args.epochs,
        #             verbose=False,
        #             opt_choice=args.optimizer,
        #             lr=args.lr,
        #             wd=args.wd,
        #         )
        #         num_retrain += 1
        #     else:
        #         grad_norm_approx_sum += grad_norm_approx[i]
            # remove_finish_time = time.perf_counter()
            # acc_removal[0].append(lr_eval(w_approx, X_val_new, y_val).item())
            # acc_removal[1].append(lr_eval(w_approx, X_test_new, y_test).item())
        unlearn_cost.append(remove_finish_time - update_finish_time)
        tot_cost.append(remove_finish_time - update_finish_time + return_time)
        # checkpt_file=log_prefix+f"_{i}.pt"
        # torch.save(w_approx, checkpt_file)
        # embed_file = log_prefix+f"_{i}"
        # np.save(embed_file, X_new.cpu().numpy())
        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Edge del = {nodes[0]}, grad_norm_approx_sum: {grad_norm_approx[i]},  grad_norm_real: {grad_norm_real[i]}, Val acc = {acc_removal[0][i+1]} Test acc = {acc_removal[1][i+1]}, avg update cost: {update_cost[i+1]}, avg unlearn cost:{unlearn_cost[i+1]}, avg tot cost:{tot_cost[i+1]}, num_retrain: {num_retrain}"
            )
            np.save(acc_path, acc_removal[1])
            np.save(tot_cost_path, tot_cost)
            np.save(unlearn_cost_path, unlearn_cost)
            np.save(update_cost_path, update_cost)
            # mem_info = process.memory_info()
            # memory_usage = mem_info.rss
        # if i == 0:
        #     logger.info(
        #         f"Unlearn Memory Usage: {memory_usage/1024/1024/1024} GB")

    end_time = time.perf_counter()
    logger.info("tot cost: %.6fs" % (end_time - start_time))

    logger.info("update cost: %.6fs" %
                (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" %
                (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" % (sum(tot_cost[1:]) / (len(tot_cost)-1)))
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
        return
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
        return
    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")


# def objective(trial,args,X_val,y_val):
#     reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True)
#     lr = trial.suggest_float("lr", 1e-5, 1.0, log=True)
#     if args.optimizer == "Adam":
#         wd = trial.suggest_float("wd", 1e-5, 1.0, log=True)
#         w = train(reg_lambda, lr, wd)
#     else:
#         w = train(reg_lambda, lr, args.wd)
#     if args.train_mode == "ovr":
#         val_acc = ovr_lr_eval(w, X_val, y_val)
#     else:
#         val_acc = lr_eval(w, X_val, y_val)
#     return val_acc

if __name__ == "__main__":
    main()
