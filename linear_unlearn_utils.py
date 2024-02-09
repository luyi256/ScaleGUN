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

logger = None


def setup_unlearn_logger(name):
    global logger
    logger = logging.getLogger(name)

def train(X_train, y_train, args, reg_lambda, lr, wd, b, device, weight=None,X_val=None,y_val=None):
    if args.train_mode == "ovr":
        w = ovr_lr_optimize_handloader(
            X_train,
            y_train,
            reg_lambda,
            batch_size=args.train_batch,
            init_method=args.init_method,
            weight=weight,
            b=b,
            num_steps=args.epochs,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            patience=args.patience,
            lr=lr,
            wd=wd,
            X_val=X_val,
            y_val=y_val,
        )
    else:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        w = lr_optimize(
            X_train,
            y_train,
            reg_lambda,
            b=b,
            num_steps=args.epochs,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            lr=lr,
            wd=wd,
        )

    return w

# K = X^T * X for fast computation of spectral norm
def get_K_matrix_dataloader(dataloader, feat_dim, device):
    xtx_accumulator = torch.zeros(feat_dim, feat_dim).to(device)
    for x, _ in dataloader:
        X = x.to(device)
        xtx_accumulator += X.t().mm(X)
    return xtx_accumulator


def get_K_matrix_handloader(_X, batch_size, feat_dim, device):
    xtx_accumulator = torch.zeros(feat_dim, feat_dim).to(device)
    idx = 0
    num_data = _X.shape[0]
    while idx < num_data:
        X = _X[idx : idx + batch_size].to(device)
        idx += batch_size
        xtx_accumulator += X.t().mm(X)
    return xtx_accumulator


def get_K_matrix(X):
    K = X.t().mm(X)
    return K


def sqrt_spectral_norm(A, num_iters=100):
    """
    return:
        sqrt of maximum eigenvalue/spectral norm
    """
    x = torch.randn(A.size(0)).float().to(A.device)
    for i in range(num_iters):
        x = A.mv(x)
        x_norm = x.norm()
        x /= x_norm
    max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
    return math.sqrt(max_lam)


# hessian of loss wrt w for binary classification
def lr_hessian_inv_dataloader(w, dataloader, lam, batch_size=50000):
    """
    The hessian here is computed wrt sum.
    input:
        w: (d,n_c)
        X: (n,d)
        y: (n,n_c)
        lambda: scalar
        batch_size: int
    return:
        hessian: a list of n_c (d,d)
    """
    device = w.device
    H = torch.zeros(w.size(1), w.size(0), w.size(0)).to(device)
    total_num = 0
    for x, y in dataloader:
        X = x.to(device)
        for k in range(y.size(1)):
            Y = y[:, k].to(device)
            z = torch.sigmoid(Y * X.mv(w[:, k]))
            D = z * (1 - z)
            H[k] += X.t().mm(D.unsqueeze(1) * X)
        total_num += X.size(0)
    for k in range(w.size(1)):
        H[k] += lam * total_num * torch.eye(w.size(0)).float().to(device)
        H[k] = H[k].inverse()
    return H


def lr_hessian_inv_handloader(w, _X, _y, lam, batch_size):
    """
    The hessian here is computed wrt sum.
    input:
        w: (d,n_c)
        X: (n,d)
        y: (n,n_c)
        lambda: scalar
        batch_size: int
    return:
        hessian: a list of n_c (n_c,d,d)
    """
    device = w.device
    H = torch.zeros(w.size(1), w.size(0), w.size(0)).to(device)
    total_num = 0
    idx = 0
    num_data = _X.shape[0]
    while idx < num_data:
        X = _X[idx : idx + batch_size].to(device)
        y = _y[idx : idx + batch_size].to(device)
        idx += batch_size
        for k in range(y.size(1)):
            Y = y[:, k]
            z = torch.sigmoid(Y * X.mv(w[:, k]))
            D = z * (1 - z)
            H[k] += X.t().mm(D.unsqueeze(1) * X)
        total_num += X.size(0)
    for k in range(w.size(1)):
        H[k] += lam * total_num * torch.eye(w.size(0)).float().to(device)
        H[k] = H[k].inverse()
    return H


def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    """
    The hessian here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
        batch_size: int
    return:
        hessian: (d,d)
    """
    z = torch.sigmoid(y * X.mv(w))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(X.device)).inverse()


def unlearn_step1(w_approx, _X, _y, lam, batch_size, feat_dim, num_classes, device):
    # get K matrix
    xtx_accumulator = torch.zeros(feat_dim, feat_dim).to(device)
    # get Hessian
    H_inv = torch.zeros(num_classes, feat_dim, feat_dim).to(device)
    # get lr gradient
    grad_new = torch.zeros(num_classes, feat_dim).to(device)
    idx = 0
    num_data = _X.shape[0]
    while idx < num_data:
        X = _X[idx : idx + batch_size].to(device)
        y = _y[idx : idx + batch_size].to(device)
        for k in range(num_classes):
            Y = y[:, k]
            z = torch.sigmoid(Y * X.mv(w_approx[:, k]))
            grad_new[k] += X.t().mv((z - 1) * Y)
            D = z * (1 - z)
            H_inv[k] += X.t().mm(D.unsqueeze(1) * X)
        xtx_accumulator += X.t().mm(X)
        idx += batch_size
    spec_norm = sqrt_spectral_norm(xtx_accumulator)
    for k in range(num_classes):
        H_inv[k] += lam * num_data * torch.eye(feat_dim).float().to(device)
        H_inv[k] = H_inv[k].inverse()
        grad_new[k] += lam * w_approx[:, k] * num_data
    return spec_norm, H_inv, grad_new

def unlearn_step2(w_approx,_X,_y, lam, Delta,batch_size,feat_dim,num_classes, device):
    num_data = _X.shape[0]
    delta_p = torch.zeros(num_data, Delta.size(1)).to(device)
    grad_old= torch.zeros(num_classes, feat_dim).to(device)
    idx = 0
    while idx < num_data:
        X = _X[idx : idx + batch_size].to(device)
        y= _y[idx : idx + batch_size].to(device)
        cur_batch_size = X.size(0)
        end_index = idx + cur_batch_size
        delta_p[idx:end_index] += X @ Delta
        for k in range(num_classes):
            Y = y[:, k]
            z = torch.sigmoid(Y * X.mv(w_approx[:, k]))
            grad_old[k] += X.t().mv((z - 1) * Y)
        idx += batch_size
    for k in range(num_classes):
        grad_old[k] += lam * w_approx[:, k] * num_data
    return delta_p,grad_old

def lr_loss(w, X, y, lam):
    """
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        averaged training loss with L2 regularization
    """
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


def lr_optimize(
    X,
    y,
    lam,
    b=None,
    num_steps=100,
    tol=1e-32,
    verbose=False,
    opt_choice="LBFGS",
    lr=0.01,
    wd=0,
    X_val=None,
    y_val=None,
):
    """
    b is the noise here. It is either pre-computed for worst-case, or pre-defined.
    """
    device = X.device
    w = torch.empty(X.size(1)).float()
    init.normal_(w, 0, 1)
    w = torch.autograd.Variable(w.to(device), requires_grad=True)

    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)

    if opt_choice == "LBFGS":
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == "Adam":
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()

        if verbose:
            print(
                "Iteration %d: loss = %.6f, grad_norm = %.6f"
                % (i + 1, loss.cpu(), w.grad.norm())
            )

        if opt_choice == "LBFGS":
            optimizer.step(closure)
        elif opt_choice == "Adam":
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            train_acc = lr_eval(w, X, y)
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print(
                    f"train accuracy = {train_acc:.4f}, Val accuracy = {val_acc:.4f}, Best Val acc = {best_val_acc:.4f}"
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


def ovr_lr_optimize_dataloader(
    train_loader,
    lam,
    init_method="zero",
    weight=None,
    b=None,
    num_steps=100,
    tol=1e-32,
    verbose=False,
    opt_choice="LBFGS",
    lr=0.01,
    wd=0,
    patience=20,
    X_val=None,
    y_val=None,
):
    """
    y: (n_train, c). one-hot
    y_val: (n_val,) NOT one-hot
    """
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    cur_patience = patience
    device = b.device
    w = torch.zeros(b.size()).float()
    if init_method == "kaiming":
        init.kaiming_uniform_(w, a=math.sqrt(5))
        # logger.info("use kaiming_uniform_ initialization")
    elif init_method == "xavier":
        init.xavier_uniform_(w)
        # logger.info("use xavier_uniform_ initialization")
    # else:
    #     logger.info("use zero initialization")
    w = torch.autograd.Variable(w.to(device), requires_grad=True)

    if opt_choice == "LBFGS":
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == "Adam":
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")
    num_train=train_loader.dataset.__len__()
    def closure():
        if b is None:
            return (
                ovr_lr_loss_dataloader(w, X, y, weight)
                + lam * w.pow(2).sum() / 2 * X.size(0) / num_train
            )
        else:
            return (
                ovr_lr_loss_dataloader(w, X, y, weight)
                + lam * w.pow(2).sum() / 2 * X.size(0) / num_train
                + (b * w).sum() / (num_train << 1) * X.size(0)
            )

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        for x, y in train_loader:
            X, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ovr_lr_loss_dataloader(w, X, y, weight)
            loss += lam * w.pow(2).sum() / 2 * X.size(0) / num_train
            if b is not None:
                if weight is None:
                    # no / X.size(0)
                    loss += (b * w).sum() / (num_train) / (num_train) * X.size(0)
                # print("b*w:", ((b * w).sum() / X.size(0)))
                else:
                    loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
            loss.backward()

            if verbose:
                logger.debug(
                    "Iteration %d: loss = %.6f,  grad_norm = %.6f, loss = %.6f"
                    % (
                        i + 1,
                        loss.cpu(),
                        w.grad.norm(),
                        lam * w.pow(2).sum() / 2 * X.size(0),
                    )
                )
                # print("w:", w[0, :10].cpu().detach().numpy())

            if opt_choice == "LBFGS":
                optimizer.step(closure)
            elif opt_choice == "Adam":
                optimizer.step()
            else:
                raise ("Error: Not supported optimizer.")

            if X_val is not None:
                train_acc = ovr_lr_eval(w, X, y)
                val_acc = ovr_lr_eval(w, X_val, y_val)
                if verbose:
                    logger.debug(
                        f"train accuracy = {train_acc:.4f}, Val accuracy = {val_acc:.4f}, Best Val acc = {best_val_acc:.4f}"
                    )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    w_best = w.clone().detach()
                    cur_patience = patience
                else:
                    cur_patience -= 1
                    if cur_patience <= 0:
                        return w_best
            else:
                w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


def ovr_lr_optimize_handloader(
    X_train,
    y_train,
    lam,
    batch_size=100000,
    init_method="zero",
    weight=None,
    b=None,
    num_steps=100,
    tol=1e-32,
    verbose=False,
    opt_choice="LBFGS",
    lr=0.01,
    wd=0,
    patience=20,
    X_val=None,
    y_val=None,
):
    """
    y: (n_train, c). one-hot
    y_val: (n_val,) NOT one-hot
    """
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    cur_patience = patience
    num_train = X_train.shape[0]
    device = b.device
    w = torch.zeros(b.size()).float()
    if init_method == "kaiming":
        init.kaiming_uniform_(w, a=math.sqrt(5))
        # logger.info("use kaiming_uniform_ initialization")
    elif init_method == "xavier":
        init.xavier_uniform_(w)
    #     logger.info("use xavier_uniform_ initialization")
    # else:
    #     logger.info("use zero initialization")
    w = torch.autograd.Variable(w.to(device), requires_grad=True)

    if opt_choice == "LBFGS":
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == "Adam":
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    elif opt_choice == "SGD":
        optimizer = optim.SGD([w], lr=lr, weight_decay=wd)
    elif opt_choice == "SGDM":
        optimizer = optim.SGD([w], lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ("Error: Not supported optimizer.")

    def closure():
        if b is None:
            return (
                ovr_lr_loss_dataloader(w, X, y, weight)
                + lam * w.pow(2).sum() / 2 * X.size(0) / num_train
            )
        else:
            return (
                ovr_lr_loss_dataloader(w, X, y, weight)
                + lam * w.pow(2).sum() / 2 * X.size(0) / num_train
                + (b * w).sum() / (num_train << 1) * X.size(0)
            )

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        idx = 0
        shuffled_indices = torch.randperm(num_train)
        X_train_tmp=X_train[shuffled_indices]
        y_train_tmp=y_train[shuffled_indices]
        while idx < num_train:
            X = X_train_tmp[idx : idx + batch_size].to(device)
            y = y_train_tmp[idx : idx + batch_size].to(device)
            idx += batch_size
            optimizer.zero_grad()
            loss = ovr_lr_loss_dataloader(w, X, y, weight)
            loss += lam * w.pow(2).sum() / 2 * X.size(0) / num_train
            if b is not None:
                if weight is None:
                    # no / X.size(0)
                    loss += (b * w).sum() / (num_train) / (num_train) * X.size(0)
                # print("b*w:", ((b * w).sum() / X.size(0)))
                else:
                    loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
            loss.backward()

            if verbose:
                logger.debug(
                    "Iteration %d: loss = %.6f,  grad_norm = %.6f, lambda loss = %.6f"
                    % (
                        i + 1,
                        loss.cpu(),
                        w.grad.norm(),
                        lam * w.pow(2).sum() / 2 * X.size(0) / num_train
                    )
                )
                # print("w:", w[0, :10].cpu().detach().numpy())

            if opt_choice == "LBFGS":
                optimizer.step(closure)
            else: 
                optimizer.step()

            if X_val is not None:
                train_acc = ovr_lr_eval(w, X, y)
                val_acc = ovr_lr_eval(w, X_val, y_val)
                if verbose:
                    logger.debug(
                        f"train accuracy = {train_acc:.4f}, Val accuracy = {val_acc:.4f}, Best Val acc = {best_val_acc:.4f}"
                    )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    w_best = w.clone().detach()
                    cur_patience = patience
                else:
                    cur_patience -= 1
                    if cur_patience <= 0:
                        return w_best
            else:
                w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


def ovr_lr_optimize(
    X,
    y,
    lam,
    init_method="zero",
    weight=None,
    b=None,
    num_steps=100,
    tol=1e-32,
    verbose=False,
    opt_choice="LBFGS",
    lr=0.01,
    wd=0,
    X_val=None,
    y_val=None,
):
    """
    y: (n_train, c). one-hot
    y_val: (n_val,) NOT one-hot
    """
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    w = torch.zeros(b.size()).float()
    if init_method == "kaiming":
        init.kaiming_uniform_(w, a=math.sqrt(5))
        # logger.info("use kaiming_uniform_ initialization")
    elif init_method == "xavier":
        init.xavier_uniform_(w)
    #     logger.info("use xavier_uniform_ initialization")
    # else:
    #     logger.info("use zero initialization")
    w = torch.autograd.Variable(w.to(X.device), requires_grad=True)

    # print("b:", b[0, :10])

    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)

    if opt_choice == "LBFGS":
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == "Adam":
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
                # print("b*w:", ((b * w).sum() / X.size(0)))
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()

        if verbose:
            logger.info(
                "Iteration %d: loss = %.6f, grad_norm = %.6f, regularizer loss = %.6f"
                % (i + 1, loss.cpu(), w.grad.norm(), lam * w.pow(2).sum() / 2)
            )
            # print("w:", w[0, :10].cpu().detach().numpy())

        if opt_choice == "LBFGS":
            optimizer.step(closure)
        elif opt_choice == "Adam":
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        if X_val is not None:
            val_acc = ovr_lr_eval(w, X_val, y_val)
            if verbose:
                logger.info(
                    f"Val accuracy = {val_acc}, Best Val acc = {best_val_acc}"
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


def ovr_lr_loss(w, X, y, lam, weight=None):
    """
     input:
        w: (d,c)
        X: (n,d)
        y: (n,c), one-hot
        lambda: scalar
        weight: (c,) / None
    return:
        loss: scalar
    """
    z = batch_multiply(X, w) * y
    # print("X*w:", batch_multiply(X, w)[:5])
    # print("y:", y[:5])
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


def ovr_lr_loss_dataloader(w, X, y, weight=None):
    """
         input:
            w: (d,c)
            X: (n,d)
            y: (n,c), one-hot
            lambda: scalar
            weight: (c,) / None
    return:
            loss: scalar
        no mean, since X is not full-batch
    """
    z = batch_multiply(X, w) * y
    # print("X*w:", batch_multiply(X, w)[:5])
    # print("y:", y[:5])
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum()
    else:
        return -F.logsigmoid(z).mul_(weight).sum()


# gradient of loss wrt w for binary classification
def lr_grad_dataloader(w, dataloader, lam, feat_dim, num_classes):
    """
    The gradient here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        gradient: (d,)
    """
    ans = torch.zeros(num_classes, feat_dim).to(w.device)
    total_num = 0
    for x, y in dataloader:
        X = x.to(w.device)
        for k in range(num_classes):
            Y = y[:, k].to(w.device)
            z = torch.sigmoid(Y * X.mv(w[:, k]))
            ans[k] += X.t().mv((z - 1) * Y)
        total_num += X.size(0)
    for k in range(num_classes):
        ans[k] += lam * w[:, k] * total_num
    return ans


def lr_grad_handloader(w, _X, _y, batch_size, lam, feat_dim, num_classes):
    """
    The gradient here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        gradient: (d,)
    """
    ans = torch.zeros(num_classes, feat_dim).to(w.device)
    idx = 0
    num_data = _X.shape[0]
    while idx < num_data:
        X = _X[idx : idx + batch_size].to(w.device)
        y = _y[idx : idx + batch_size].to(w.device)
        idx += batch_size
        for k in range(num_classes):
            Y = y[:, k]
            z = torch.sigmoid(Y * X.mv(w[:, k]))
            ans[k] += X.t().mv((z - 1) * Y)
    for k in range(num_classes):
        ans[k] += lam * w[:, k] * num_data
    return ans


def lr_grad(w, X, y, lam):
    """
    The gradient here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        gradient: (d,)
    """
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w


def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i + 1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(A.device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(A.device)


def compute_delta_p(dataloader, Delta, num_train):
    """
    input: x: (n,d)
           Delta:(k,d)
    """
    delta_p = torch.zeros(Delta.size(0), num_train).to(Delta[0].device)
    start_index = 0
    for _, (x, _) in enumerate(dataloader):
        X = x.to(Delta[0].device)
        batch_size = X.size(0)
        end_index = start_index + batch_size
        for k in range(Delta.size(0)):
            delta_p[k, start_index:end_index] += X.mv(Delta[k])
        start_index = end_index
    return delta_p


def compute_delta_p_handloader(_X, batch_size, Delta):
    """
    input: x: (n,d)
           Delta:(k,d)
    """
    device = Delta.device
    num_data = _X.shape[0]
    delta_p = torch.zeros(num_data, Delta.size(1)).to(device)
    idx = 0
    while idx < num_data:
        X = _X[idx : idx + batch_size].to(device)
        cur_batch_size = X.size(0)
        end_index = idx + cur_batch_size
        delta_p[idx:end_index] += X @ Delta
        idx += batch_size
    return delta_p


def ovr_lr_eval(w, X, y):
    """
    input:
        w: (d,c)
        X: (n,d)
        y: (n,), NOT one-hot
    return:
        loss: scalarf
    """
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    pred = X.mm(w).max(1)[1]
    return pred.eq(y).float().mean()


# evaluate function for binary classification
def lr_eval(w, X, y):
    """
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
    return:
        prediction accuracy
    """
    return X.mv(w).sign().eq(y).float().mean()
