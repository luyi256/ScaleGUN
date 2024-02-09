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
from torch.autograd import grad
logger = None


def setup_unlearn_logger(name):
    global logger
    logger = logging.getLogger(name)


def train_model(model, device, X_train, y_train, batch_size, optimizer, epochs,X_val,y_val,evaluator,checkpt_file,patience,verbose=False,noises=None):
    bad_counter = 0
    best = 0
    model.train()
    start_time=time.time()
    tot_ep_time=0
    num_train=X_train.shape[0]
    for epoch in range(epochs):
        one_ep_start_time=time.time()
        loss_list = []
        idx=0
        shuffled_indices = torch.randperm(num_train)
        X_train_tmp=X_train[shuffled_indices]
        y_train_tmp=y_train[shuffled_indices]
        while idx<num_train:
            x=X_train_tmp[idx:idx+batch_size].to(device)
            y=y_train_tmp[idx:idx+batch_size].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y)
            if noises is not None:
                for i,param in enumerate(model.parameters()):
                    if param.requires_grad:
                        loss+=(param.data*noises[i]).sum()/(num_train<<1)*x.size(0)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            idx+=batch_size
            # print(f"epoch: {epoch}, idx: {idx}, GPU memory: {torch.cuda.memory_allocated()/1024/1024}")
            del x,y,out
        # print(f"epoch: {epoch}, idx: {idx}, GPU memory: {torch.cuda.memory_allocated()/1024/1024}")
        train_ep = time.time()-one_ep_start_time
        tot_ep_time+=train_ep
        f1_val = test(model, device, X_val,y_val, batch_size,evaluator)
        if verbose:
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch:{epoch+1:02d}," f"Train_loss:{loss:.3f}",
                    f"Valid_acc:{100*f1_val:.2f}%",
                    f"Time_cost:{train_ep:.3f}/{tot_ep_time:.3f}",
                )
        if f1_val > best:
            best = f1_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter ==patience:
            logger.info(f"{epoch}, Early Stop!")
            break
        # print(f"epoch: {epoch}, GPU memory: {torch.cuda.memory_allocated()/1024/1024}")
        
    train_time = time.time() - start_time
    # logger.info(f"Train cost: {train_time:.2f}s")
    # logger.info(f"Train epochs cost: {tot_ep_time:.2f}s")
    # logger.info(f"Avg loss: {np.mean(loss_list):.4f}")
    return train_time


def cal_grad(model, device, loader,retain=False):
    model.eval()
    model.zero_grad()
    model_params = [p for p in model.parameters() if p.requires_grad]
    tot_loss=torch.zeros(1).cuda(device)
    for step, (x, y) in enumerate(loader):
        x, y = x.cuda(device), y.cuda(device)
        out = model(x)
        loss = F.nll_loss(out, y,reduction='sum')
        tot_loss=tot_loss+loss
    cur_grad = grad(tot_loss/loader.dataset.__len__(), model_params, create_graph=retain)
    return cur_grad

def cal_grad_handloader(model, device, _X,_y,batch_size,retain=False):
    model.eval()
    model.zero_grad()
    model_params = [p for p in model.parameters() if p.requires_grad]
    num_data=_X.shape[0]
    idx=0
    tot_loss=torch.zeros(1).cuda(device).requires_grad_()
    while idx<num_data:
        x=_X[idx:idx+batch_size].to(device)
        y=_y[idx:idx+batch_size].to(device)
        out = model(x)
        loss = F.nll_loss(out, y,reduction='sum')
        tot_loss=tot_loss+loss
        idx+=batch_size
    cur_grad = grad(tot_loss/num_data, model_params, create_graph=retain)
    return cur_grad


def cal_grad_data(model, device, X_train, y_train, retain):
    model.eval()
    model.zero_grad()
    model_params = [p for p in model.parameters() if p.requires_grad]
    x, y = X_train.cuda(device), y_train.cuda(device)
    out = model(x)
    loss = F.nll_loss(out, y, reduction='mean')
    grads = grad(loss, model_params, create_graph=retain)
    return grads

def _get_fmin_loss_fn(v, **kwargs):
    """
    1/2 x^T H x - v^T x <--> Hx=v
    """
    device = kwargs["device"]

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        # calculate hvp=Hx
        _hvp = hvp(kwargs["_grad"].view(-1), kwargs["p"], x).view(-1)  # grad,p,x
        _hvp += kwargs["damping"] * x
        obj = 0.5 * torch.dot(_hvp, x) - torch.dot(v.view(-1), x)
        return obj.detach().cpu().numpy()

    return get_fmin_loss


def _get_fmin_grad_fn(v, **kwargs):
    device = kwargs["device"]

    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        _hvp = hvp(kwargs["_grad"].view(-1), kwargs["p"], x).view(-1)
        _hvp += kwargs["damping"] * x
        return (_hvp - v.view(-1)).detach().cpu().numpy()

    return get_fmin_grad

def com_accuracy(y_pred, y):
    pred = y_pred.data.max(1)[1]
    pred = pred.reshape(pred.size(0),1)
    correct = pred.eq(y.data).cpu().sum()
    accuracy = correct.to(dtype=torch.long) * 100. / len(y)
    return accuracy

@torch.no_grad()
def test(model, device, X_val, y_val,batch_size, evaluator=None):
    model.eval()
    # y_pred, y_true = [], []
    acc_list=[]
    num_data=X_val.shape[0]
    idx=0
    while idx<num_data:
        x=X_val[idx:idx+batch_size].to(device)
        y=y_val[idx:idx+batch_size].to(device)
        out = model(x)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        # y_true.append(y.unsqueeze(1))
        idx+=batch_size
        acc=com_accuracy(out,y.unsqueeze(1))
        acc_list.append(acc.item())
    #     return evaluator.eval(
    #     {
    #         "y_true": torch.cat(y_true, dim=0),
    #         "y_pred": torch.cat(y_pred, dim=0),
    #     }
    # )["acc"]
        del x,y,out
    return np.mean(acc_list)/100

def hvps(grad_all, model_params, h_estimate):
    element_product = 0
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)

    return_grads = grad(element_product, model_params,create_graph=True)
    return return_grads


def hvp(_grad, model_param, h_estimate):
    element_product = torch.sum(_grad * h_estimate)

    return_grads = grad(element_product, model_param,create_graph=True)
    return return_grads[0]
