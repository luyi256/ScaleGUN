import numpy as np
import argparse
import propagation
import uuid
import torch
from utils import *
from deep_unlearn_utils import *    
from deep_unlearn_utils import _get_fmin_loss_fn, _get_fmin_grad_fn
import torch.nn.functional as F
import time
import gc
from datetime import datetime
import optuna
import logging
import pytz
from model import ClassMLP
from ogb.nodeproppred import Evaluator
from argparser import argparser
from torch.autograd import grad
from scipy.optimize import fmin_cg
import random
torch.set_printoptions(precision=10)
name="deep_edge"
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG, format="%(filename)s - %(levelname)s - %(message)s", handlers=[]
)
logger_name=name
setup_logger(name)
setup_unlearn_logger(name)
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def main():
    args = argparser()
    seed_everything(seeds[args.seed])
    tz = pytz.timezone("Asia/Shanghai")
    dt = datetime.now(tz).strftime("%m%d_%H%M")
    set_logger(args, logger,dt,name)
    logger.info(args)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result")
    tot_cost_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_drop_{args.dropout}_hidden_{args.hidden}_cost"
    unlearn_cost_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_drop_{args.dropout}_hidden_{args.hidden}_unlearn_cost"
    update_cost_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_drop_{args.dropout}_hidden_{args.hidden}_update_cost"
    acc_path=f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_drop_{args.dropout}_hidden_{args.hidden}_acc"
    f_tot_cost=open(tot_cost_path+".txt","ab")
    f_unlearn_cost=open(unlearn_cost_path+".txt","ab")
    f_update_cost=open(update_cost_path+".txt","ab")
    f_acc=open(acc_path+".txt","ab")
    origin_rmax=args.rmax

    if args.dev > -1:
        device = torch.device("cuda:" + str(args.dev))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")
    torch.cuda.set_device(device)

    start = time.perf_counter()
    data,edge_index=load_data(args.path, args.dataset)

    weights=get_prop_weight(args.weight_mode, args.prop_step,args.decay)

    data.y = data.y.long()
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    y_train, y_val, y_test,train_mask, val_mask, test_mask=get_deep_split_large(data,args.train_mode,args.Y_binary)

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
    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(
        del_path,
        args.dataset,
        origin_embedding,
        edge_index.T,
        args.prop_step,
        args.r,
        weights,
        args.num_threads,
        args.rmax,
    )
    del edge_index
    gc.collect()


    init_finish_time = time.perf_counter()
    logger.info("init cost: %.6fs" % (init_finish_time - start))

    X = torch.FloatTensor(origin_embedding.T)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model")
    logger.debug(f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    del X
    
    logger.info(
        "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
        )
    )

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

    logger.info("--------------------------")
    logger.info("Training...")

    model = ClassMLP(feat_dim, args.hidden, num_classes, args.layer, args.dropout).to(
        device
    )
    noises=[]
    for param in model.parameters():
        if param.requires_grad:
            noises.append(b_std*torch.randn(param.shape).float().to(device))
    seed_everything(seeds[args.seed])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.dataset in ["ogbn-arxiv", "ogbn-products","ogbn-papers100M"]:
        evaluator = Evaluator(name=args.dataset)
    else:
        evaluator = None
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model")
    checkpt_file = f"{args.analysis_path}/{args.dataset}/{name}_model/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{origin_rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_edge_idx_{args.edge_idx_start}_init.pt"
    model.reset_parameters()
    train_time=train_model(model, device, X_train,y_train, args.train_batch, optimizer, args.epochs,X_val, y_val,evaluator,checkpt_file,args.patience,noises=noises)
    model.load_state_dict(torch.load(checkpt_file))
    test_acc = test(model, device, X_test,y_test, args.test_batch, evaluator)
    val_acc = test(model, device, X_val,y_val, args.test_batch, evaluator)
    logger.info(f"Train cost: {train_time:.2f}s")
    logger.info(f"Test accuracy:{100*test_acc:.2f}%")
    
    acc_removal = [[val_acc,], [test_acc,]]
    old_grad = cal_grad_handloader(model, device, X_train,y_train,args.test_batch,retain=False )
    logger.info(f"old_grad: {old_grad[0].norm()}")
    update_cost = [prop_time,]
    unlearn_cost = [train_time,]
    tot_cost = [train_time+prop_time,]
    logger.info("start to remove edges...")
    logger.info("*" * 20)

    start_time = time.perf_counter()
    # obtain delete edges
    edge_idx_start = args.edge_idx_start
    edge_file = del_path + "/" + args.dataset + "/" + args.dataset + f"_del_edges{args.del_postfix}.npy"
    del_edges = np.load(edge_file)
    logger.info(f"read del_edges from {edge_file}")
    if del_edges.shape[1]==2:
        del_edges=del_edges.T

    del X_train
    del X_val
    del X_test
    del noises
    gc.collect()

    for i in range(args.num_batch_removes):
        edges = del_edges[
            :,
            edge_idx_start
            + i * args.num_removes : edge_idx_start
            + args.num_removes * (i + 1),
        ].T.tolist()
        return_time = g.UpdateEdges(edges, origin_embedding, args.num_threads, args.rmax)
        update_cost.append(return_time)
        X_new = torch.FloatTensor(origin_embedding.T)
        embed_file=f"{args.analysis_path}/{args.dataset}/{name}_model/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{origin_rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_edge_idx_{args.edge_idx_start}_{i}"
        # np.save(embed_file,X_new)
        update_finish_time = time.perf_counter()
        X_new_train=X_new[train_mask]
        X_new_val=X_new[val_mask]
        X_new_test=X_new[test_mask]
        del X_new
        model_params = [p for p in model.parameters() if p.requires_grad]
        new_grad = cal_grad_handloader(model, device, X_new_train, y_train,args.test_batch,retain=True)
        vs = tuple(old_grad[k] - new_grad[k] for k in range(len(old_grad)))
        del old_grad
        inverse_hvs = []
        status = []
        cg_loss = []
        for j, (v, p) in enumerate(zip(vs, model_params)):
            fmin_loss_fn = _get_fmin_loss_fn(
                v, p=p, _grad=new_grad[j], device=device, damping=args.damping
            )
            fmin_grad_fn = _get_fmin_grad_fn(
                v, p=p, _grad=new_grad[j], device=device, damping=args.damping
            )
            result = fmin_cg(
                f=fmin_loss_fn,
                x0=v.view(-1).detach().cpu().numpy(),
                fprime=fmin_grad_fn,
                gtol=1e-4,
                disp=False,
                full_output=True,
                maxiter=200,
            )
            inverse_hvs.append(torch.tensor(result[0]).view(v.shape))
            status.append(result[4])
            cg_loss.append(result[1])
            del result
        idx = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data = inverse_hvs[idx].to(device)+p.data
            idx += 1
        del model_params, inverse_hvs, new_grad, vs
        gc.collect()
        torch.cuda.empty_cache()
        # print(f"after one remove, GPU memory: {torch.cuda.memory_allocated()/1024/1024}")
        old_grad = cal_grad_handloader(model, device, X_new_train, y_train,args.test_batch,retain=False)
        # old_grad=cal_grad_data(model,device,X_new_train,y_train)
        test_acc = test(model, device, X_new_test, y_test,args.test_batch,evaluator)
        val_acc = test(model, device, X_new_val,y_val,args.test_batch, evaluator)
        remove_finish_time = time.perf_counter()
        acc_removal[0].append(val_acc)
        acc_removal[1].append(test_acc)
        checkpt_file = f"{args.analysis_path}/{args.dataset}/{name}_model/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{origin_rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_batch_{args.train_batch}_edge_idx_{args.edge_idx_start}_{i}.pt"
        torch.save(model.state_dict(), checkpt_file)
        del X_new_train, X_new_val, X_new_test
        
        unlearn_cost.append(remove_finish_time - update_finish_time)
        tot_cost.append(remove_finish_time - update_finish_time+return_time)
        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Edge del = {edges[0]}, Val acc = {acc_removal[0][i]} Test acc = {acc_removal[1][i]}, avg update cost: {np.mean(update_cost[1:])}, avg unlearn cost:{np.mean(unlearn_cost[1:])}, avg tot cost:{np.mean(tot_cost[1:])}"
            )
    end_time = time.perf_counter()

    logger.info("update cost: %.6fs" % (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" % (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" % (sum(tot_cost) / (len(tot_cost)-1)))
    logger.info("tot cost: %.6fs" % (end_time - start_time))
    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")
    np.save(tot_cost_path+f"_{edge_idx_start}", tot_cost)
    np.save(unlearn_cost_path+f"_{edge_idx_start}", unlearn_cost)
    np.save(update_cost_path+f"_{edge_idx_start}",update_cost )
    np.save(acc_path+f"_{edge_idx_start}", acc_removal[1])
    print(acc_path+".txt")

if __name__ == "__main__":
    main()
