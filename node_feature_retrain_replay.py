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
from argparser import argparser
name = "node_feat_retrain_test"
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
    tot_cost_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_{args.removal_mode}_cost"
    unlearn_cost_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_{args.removal_mode}_unlearn_cost"
    update_cost_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_{args.removal_mode}_update_cost"
    acc_path = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_{args.removal_mode}_acc"
    f_tot_cost = open(tot_cost_path+".txt", "ab")
    f_unlearn_cost = open(unlearn_cost_path+".txt", "ab")
    f_update_cost = open(update_cost_path+".txt", "ab")
    f_acc = open(acc_path+".txt", "ab")
    log_prefix = f"{args.analysis_path}/{args.dataset}/{name}_model/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_{args.removal_mode}_edge_idx_{args.edge_idx_start}"

    if args.dev > -1:
        device = torch.device("cuda:" + str(args.dev))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")

    start = time.perf_counter()
    data, edge_index = load_data(args.path, args.dataset)
    data.y = data.y.long()
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    # obtain delete nodes
    node_idx_start = args.edge_idx_start
    del_path = os.path.join(args.path, args.del_path_suffix)
    node_file = (
        del_path + "/" + args.dataset + "/" + args.dataset + "_del_nodes.npy"
    )
    del_nodes = np.load(node_file)
    np.random.shuffle(del_nodes)
    del_nodes = del_nodes[node_idx_start:node_idx_start +
                          args.num_removes*args.num_batch_removes]

    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)
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
    # column_sum_avg = feat.abs().sum(axis=0).mean()
    # logger.info(f"column_sum_avg: {column_sum_avg}")
    # args.rmax = args.rmax*column_sum_avg
    feat = feat.T
    origin_embedding = np.copy(feat.numpy())
    if args.dataset in ["ogbn-arxiv", "ogbn-products", "pokec"]:
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()

    g.init_graph(del_path, args.dataset, edge_index.T,
                 args.prop_step, args.r, weights, args.num_threads, feat_dim)
    prop_time = g.PowerMethod(origin_embedding)
    logger.info(f"initial prop time: {prop_time}")
    # groundtruth=np.copy(feat.numpy())
    # g.PowerMethod(groundtruth)
    # check_propagation(groundtruth,origin_embedding)
    del edge_index
    gc.collect()
    init_finish_time = time.perf_counter()

    # if origin_embedding.shape[0] == feat_dim:  # feature dimension
    X = torch.FloatTensor(origin_embedding.T)
    logger.debug(
        f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")

    X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(
        data, X, args.train_mode, args.Y_binary, args.rand)
    # embed_file = log_prefix+f"_init"
    # np.save(embed_file,X.numpy())
    # del data
    logger.info(
        "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
        )
    )
    data_prepare_time = time.perf_counter()

    weight = None
    logger.info("--------------------------")
    logger.info("Training...")
    train_time = time.perf_counter()
    if args.train_mode == "ovr":
        b = torch.zeros(feat_dim, num_classes).float().to(device)
    else:  # binary classification
        b = torch.zeros(feat_dim).float().to(device)
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
    opt_grad_norm = 0.0
    if args.train_mode == "ovr":
        for k in range(y_train.size(1)):
            opt_grad_norm += (
                lr_grad(w[:, k], X_train, y_train[:, k],
                        best_reg_lambda).norm().cpu()
            )
    else:
        grad_old = lr_grad(w, X_train, y_train, best_reg_lambda)
        opt_grad_norm = grad_old.norm().cpu()
    logger.info("init cost: %.6fs" % (init_finish_time - start))
    logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
    checkpt_file = log_prefix+f"_init.pt"
    # torch.save(w, checkpt_file)

    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    if args.train_mode == "ovr":
        train_acc = ovr_lr_eval(w, X_train, data.y[train_mask].to(device))
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
    else:
        train_acc = lr_eval(w, X_train, data.y[train_mask].to(device))
        val_acc = lr_eval(w, X_val, y_val)
        test_acc = lr_eval(w, X_test, y_test)
    logger.info("Train accuracy: %.4f" % train_acc)
    logger.info("Validation accuracy: %.4f" % val_acc)
    logger.info("Test accuracy: %.4f" % test_acc)
    update_cost = [prop_time,]
    unlearn_cost = [train_finish_time - train_time,]
    tot_cost = [train_finish_time - train_time+prop_time,]
    acc_removal = [[val_acc.item()], [test_acc.item()]]
    logger.info("first train cost: %.6fs" % (train_finish_time - train_time))
    if args.rand:  # store the logits to train the attack model
        train_logits = predict(w, X_train)
        test_logits = predict(w, X_test)
        MI_path = f"./analysis/{args.dataset}/{name}_MI/"
        check_dir(MI_path)
        np.save(f'{MI_path}/MI_pred_train_logits_{args.seed}.npy',
                train_logits.cpu().numpy())
        np.save(f'{MI_path}/MI_pred_test_logits_{args.seed}.npy',
                test_logits.cpu().numpy())
        print("Finish output logits")
        exit()
        # remove
    logger.info("start to remove edges...")
    logger.info("*" * 20)

    start_time = time.perf_counter()

    num_retrain = 0

    if args.test_MI:
        MI_ori_pred = torch.empty(
            (0, num_classes), dtype=torch.float).to(device)
        for i in range(args.num_batch_removes):
            nodes = del_nodes[
                node_idx_start
                + i * args.num_removes: node_idx_start
                + args.num_removes * (i + 1),
            ].T.tolist()
            MI_ori_pred = torch.cat([MI_ori_pred, predict(
                w, X[nodes].to(device))], dim=0)
        MI_path = f"./analysis/{args.dataset}/{name}_MI/"
        check_dir(MI_path)
        np.save(f'{MI_path}/MI_pred_{name}_ori_{args.num_batch_removes}_{args.num_removes}_{args.seed}.npy',
                MI_ori_pred.cpu().numpy())
    if args.replay:
        all_pred = predict_class(w, X.to(device))
        res = all_pred == num_classes - 1
        res_deleted = res[del_nodes]
        logger.info(
            f"Ori Replay test false: {res_deleted.sum()} {res_deleted.shape[0]} Ori Total test false: {res.sum()} {res.shape[0]}")
        print(
            f"Ori Replay test false: {res_deleted.sum()} {res_deleted.shape[0]} Ori Total test false: {res.sum()} {res.shape[0]}")
        replay_path = f"./analysis/{args.dataset}/{name}_MI/"
        check_dir(replay_path)
        np.savez(f"{replay_path}/replay_pred_ori_{name}_{args.seed}.npz",
                 all_pred=all_pred.cpu().numpy(), false_pred=res.cpu().numpy(), del_nodes=del_nodes)

    w_approx = w.clone().detach().to(device)
    del X
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
        start = time.perf_counter()
        if args.removal_mode == "node":
            g.UpdateStructNodes(nodes)
            feat[:, nodes] = 0
            retrain_embedding = np.copy(feat.numpy())
            g.PowerMethod(retrain_embedding)
        else:
            feat[:, nodes] = 0
            retrain_embedding = np.copy(feat.numpy())
            g.PowerMethod(retrain_embedding)
        start_unlearn = time.perf_counter()
        update_cost.append(start_unlearn-start)
        train_mask[nodes] = False

        # groundtruth=np.copy(feat.numpy())
        # g.PowerMethod(groundtruth)
        # check_propagation(groundtruth,origin_embedding)
        # test_node=edges[0][0]
        # logger.debug(f"test node: {test_node}, check propagation: {origin_embedding[:10,test_node]}")

        X_new = torch.FloatTensor(retrain_embedding.T)
        X_train_new = X_new[train_mask].to(device)
        y_train = F.one_hot(data.y[train_mask],
                            num_classes=data.y.max().item()+1) * 2 - 1
        y_train = y_train.float().to(device)
        if args.train_mode == "ovr":
            w_approx = ovr_lr_optimize(
                X_train_new,
                y_train,
                best_reg_lambda,
                weight=None,
                b=torch.zeros(feat_dim, num_classes).float().to(device),
                verbose=args.verbose,
                opt_choice=args.optimizer,
                lr=best_lr,
                wd=best_wd,
                # X_val=features_val,
                # y_val=y_val,
            )
            remove_finish_time = time.perf_counter()
            X_val_new = X_new[val_mask].to(device)
            acc_removal[0].append(ovr_lr_eval(
                w_approx, X_val_new, y_val).item())
            X_test_new = X_new[test_mask].to(device)
            acc_removal[1].append(ovr_lr_eval(
                w_approx, X_test_new, y_test).item())
        else:
            w_approx = lr_optimize(
                X_train_new.to(device),
                y_train.to(device),
                best_reg_lambda,
                b=torch.zeros(feat_dim).float().to(device),
                num_steps=args.epochs,
                verbose=args.verbose,
                opt_choice=args.optimizer,
                lr=args.lr,
                wd=args.wd,
            )
            remove_finish_time = time.perf_counter()
            X_val_new = X_new[val_mask].to(device)
            acc_removal[0].append(lr_eval(w_approx, X_val_new, y_val).item())
            X_test_new = X_new[test_mask].to(device)
            acc_removal[1].append(lr_eval(w_approx, X_test_new, y_test).item())
        # checkpt_file = log_prefix+f"_{i}.pt"
        # torch.save(w_approx, checkpt_file)
        # embed_file = log_prefix+f"_{i}"
        # np.save(embed_file,X_new.numpy())
        unlearn_cost.append(remove_finish_time - start_unlearn)
        tot_cost.append(remove_finish_time - start)
        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Node del = {nodes[0]}, Val acc = {acc_removal[0][i+1]} Test acc = {acc_removal[1][i+1]}, avg update cost: {update_cost[i+1]}, avg unlearn cost:{unlearn_cost[i+1]}, avg tot cost:{tot_cost[i+1]}, num_retrain: {num_retrain}"
            )
    end_time = time.perf_counter()
    logger.info("update cost: %.6fs" %
                (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" %
                (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" % (sum(tot_cost[1:]) / (len(tot_cost)-1)))
    logger.info("tot cost: %.6fs" % (end_time - start_time))
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

    np.save(tot_cost_path+f"_{args.edge_idx_start}", tot_cost)
    np.save(unlearn_cost_path+f"_{args.edge_idx_start}", unlearn_cost)
    np.save(update_cost_path+f"_{args.edge_idx_start}", update_cost)
    np.save(acc_path+f"_{args.edge_idx_start}", acc_removal[1])
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
