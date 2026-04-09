import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
import numpy as np
from torch.utils.data import Subset
import sys

class Logger(object):
    def __init__(self, fileN="record.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()                 # flush the file after each write
    def flush(self):
        self.log.flush()
        
sys.stdout = Logger("logs/sisa.txt")

def average_parameters(num_train_env, list_vars, list_alpha):
    sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
    for i in range(num_train_env):
        W_n = list_vars[i]
        alpha = list_alpha[i]
        sum_vars = [sum_ + alpha*update for sum_, update in zip(sum_vars, W_n)]
    return sum_vars
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def global_norm(tensors):
    # tensors: list[Tensor] same shapes as params
    # returns scalar tensor on same device
    s = None
    for t in tensors:
        if t is None:
            continue
        v = t.detach()
        val = (v * v).sum()
        s = val if s is None else (s + val)
    return torch.sqrt(s + 1e-12)


def generate_W_global(n_parties, W_b, P_b, sigma_b, alpha_b, l2_lambda):
    """
    W_b: list of length n_parties, each entry is a list of tensors (local params)
    P_b: list of length n_parties, each entry is a list of tensors (dual params)
    sigma_b: list of length n_parties, each entry is scalar float
    alpha_b: list of length n_parties, each entry is client weight
    """
    n_params = len(W_b[0])
    W_global = []

    denom = sum(alpha_b[i] * sigma_b[i] for i in range(n_parties)) + l2_lambda

    for p_idx in range(n_params):
        num = 0.0
        for i in range(n_parties):
            num = num + alpha_b[i] * (sigma_b[i] * W_b[i][p_idx] + P_b[i][p_idx])

        Wg = num / denom
        W_global.append(Wg.clone())

    return W_global

def zero_grad(params):
    """
    Zeroes out gradients for the given parameters.
    """
    for param in params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def split_and_aggregate_minibatches(client_subsets, num_splits=10):
    """
    Args:
        client_subsets: List of torch.utils.data.Subset, one per client.
        num_splits: Number of minibatches to split each client's data into.
    
    Returns:
        A list of length `num_splits`, where each element is a tuple:
            (list of Subsets from clients for this global batch,
             list of sample ratios for each minibatch in that global batch)
    """
    global_batches = []

    for split_id in range(num_splits):
        minibatch_list = []
        size_list = []

        for client_ds in client_subsets:
            indices = client_ds.indices if isinstance(client_ds, Subset) else list(range(len(client_ds)))
            total_len = len(indices)
            split_sizes = [total_len // num_splits] * num_splits
            for i in range(total_len % num_splits):
                split_sizes[i] += 1

            # Compute start and end index for this split
            start = sum(split_sizes[:split_id])
            end = start + split_sizes[split_id]
            mb_indices = indices[start:end]

            minibatch_list.append(Subset(client_ds.dataset, mb_indices))
            size_list.append(len(mb_indices))

        total_size = sum(size_list)
        ratio_list = [size / total_size for size in size_list]
        global_batches.append((minibatch_list, ratio_list))

    return global_batches

def heuristic_update_sigma(sigma_old, primal_res, dual_res, mu=10.0, tau=2.0):
    sigma_new = sigma_old
    if primal_res > mu * dual_res:
        sigma_new = sigma_old * tau
    elif dual_res > mu * primal_res:
        sigma_new = sigma_old / tau
    return sigma_new

def online_ogd_update_u(u, primal_res, dual_res, eta_u, G_clip, u_min, u_max, eps=1e-12):
    target_u = torch.log(primal_res + eps) - torch.log(dual_res + eps)
    loss = 0.5 * (u - target_u) ** 2
    (g_u,) = torch.autograd.grad(loss, u, retain_graph=False, create_graph=False)
    g_u = torch.clamp(g_u, -G_clip, G_clip)

    with torch.no_grad():
        u_new = torch.clamp(u - eta_u * g_u, min=u_min, max=u_max)

    return u_new.detach(), loss.detach()

def online_convex_bal_update_u(
    u,
    primal_res,
    delta_y,
    eta_u=0.05,
    G_clip=10.0,
    u_min=-20.0,
    u_max=20.0,
    eps=1e-12,
):
    """
    Per-client projected OGD on u = log(sigma), using the proof-style balance loss

        ell_k(u) = 0.5 * (u - a_k)^2
        a_k = log(primal_res) - log(delta_y)

    where delta_y is the pre-sigma dual-side quantity (dual_base).
    """
    r_clip = torch.clamp(primal_res, min=eps)
    dy_clip = torch.clamp(delta_y, min=eps)

    # target = torch.log(r_clip) - torch.log(dy_clip)
    target = -torch.log(r_clip) + torch.log(dy_clip)
    grad_u = u - target                  # exact gradient of 0.5 * (u - target)^2
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)

    with torch.no_grad():
        u_new = u - eta_u * grad_u
        u_new = torch.clamp(u_new, min=u_min, max=u_max)
        loss_val = 0.5 * (u - target).pow(2)

    return u_new.detach(), loss_val.detach(), target.detach(), grad_u.detach()

import math
import torch


def online_convex_hybrid_update_u(
    u_old,
    primal_res,
    delta_y,
    eta_u,
    alpha=1.0,
    lambda0=1.0,
    tau_mag=1.0,
    G_clip=1.0,
    u_min=-20.0,
    u_max=20.0,
    eps=1e-12,
    target_clip=True,
):
    """
    Convex hybrid loss in u = log sigma:

        ell(u) = (alpha/2) * (u - a)^2 + (lambda_k/2) * (u - u_old)^2

    where
        a = log(primal_res + eps) - log(delta_y + eps)
        lambda_k = lambda0 * tau_mag / (primal_res + delta_y + tau_mag)

    Args:
        u_old:      torch scalar tensor, current log-sigma
        primal_res: torch scalar tensor
        delta_y:    torch scalar tensor
        eta_u:      scalar stepsize
        alpha:      balance weight
        lambda0:    max stabilization weight
        tau_mag:    magnitude scale for gating
        G_clip:     gradient clipping threshold
        u_min/u_max: box constraints in log-sigma
        eps:        numerical epsilon
        target_clip: whether to clip a into [u_min, u_max]

    Returns:
        u_new, loss, a_target, grad_u, lambda_k
    """
    # Freeze the statistics
    p = primal_res.detach()
    d = delta_y.detach()

    a_target = torch.log(p + eps) - torch.log(d + eps)
    if target_clip:
        a_target = torch.clamp(a_target, min=u_min, max=u_max)

    lambda_k = lambda0 * tau_mag / (p + d + tau_mag)

    # Optimize the convex frozen loss at current round
    u = u_old.detach().clone().requires_grad_(True)
    loss = 0.5 * alpha * (u - a_target) ** 2 + 0.5 * lambda_k * (u - u_old.detach()) ** 2

    grad_u = torch.autograd.grad(loss, u)[0]
    grad_u = torch.clamp(grad_u, min=-G_clip, max=G_clip)

    with torch.no_grad():
        u_new = u_old - eta_u * grad_u
        u_new = torch.clamp(u_new, min=u_min, max=u_max)

    return u_new.detach(), loss.detach(), a_target.detach(), grad_u.detach(), lambda_k.detach()

def get_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--model', type=str, default='mlp',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo',
                        help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon/sisa')
    parser.add_argument('--use_projection_head', type=str2bool, default=False,
                        help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256,
                        help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive',
                        help='loss type for moon')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50,
                        help='number of maximum communication rounds')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout probability')
    parser.add_argument('--datadir', type=str, default='/data/yutong/datasets',
                        help='data directory')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='L2 regularization strength')
    parser.add_argument('--logdir', type=str, default='./logs/',
                        help='log directory path')
    parser.add_argument('--modeldir', type=str, default='./models/',
                        help='model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None,
                        help='log file name')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    parser.add_argument('--mu', type=float, default=0.001,
                        help='mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0,
                        help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0,
                        help='parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1,
                        help='sample ratio for each communication round')

    # sisa / ADMM-style params
    parser.add_argument('--sigma_lr', type=float, default=1.5e0,
                        help='initial sigma in sisa')
    parser.add_argument('--rho_lr', type=float, default=1e3,
                        help='initial rho-like scaling in sisa')
    parser.add_argument('--l2_lambda', type=float, default=1e-3,
                        help='hyperparameter l2_lambda in sisa')
    parser.add_argument('--mu_lr', type=float, default=0.997,
                        help='sigma decay factor in sisa')
    parser.add_argument('--decay_epoch', type=int, default=5,
                        help='epoch interval for sigma decay in sisa')
    parser.add_argument('--terminate_decay', type=int, default=50,
                        help='stop sigma decay after this epoch in sisa')

    # adaptive sigma mode
    parser.add_argument('--sigma_mode', type=str, default='fixed',
                        choices=['fixed', 'heuristic', 'online_balance', 'online_convex_bal', 'online_convex_bal_denug', 'online_hybrid'],
                        help='sigma update mode for sisa')

    parser.add_argument('--sigma_min', type=float, default=1e-6,
                        help='minimum allowed sigma')
    parser.add_argument('--sigma_max', type=float, default=1e6,
                        help='maximum allowed sigma')

    # heuristic sigma update
    parser.add_argument('--sigma_mu', type=float, default=10.0,
                        help='residual balancing threshold mu')
    parser.add_argument('--sigma_tau', type=float, default=2.0,
                        help='residual balancing multiplier tau')

    # online update on u = log(sigma)
    parser.add_argument('--eta_u', type=float, default=0.05,
                        help='step size for online update of u = log(sigma)')
    parser.add_argument('--G_clip', type=float, default=10.0,
                        help='gradient clipping threshold for u update')
    parser.add_argument('--sigma_update_freq', type=int, default=1,
                        help='update sigma every this many eligible rounds')

    # trust-region / stabilization for adaptive sigma
    parser.add_argument('--sigma_max_delta', type=float, default=0.2,
                        help='maximum change in log(sigma) per update before late-stage shrinking')
    parser.add_argument('--sigma_max_delta_min', type=float, default=0.01,
                        help='minimum trust-region size for log(sigma) updates')

    # EMA smoothing for online_convex_bal
    parser.add_argument('--sigma_ema_beta', type=float, default=0.9,
                        help='EMA coefficient for smoothing primal_res and delta_y in online_convex_bal')

    # blending schedule for online_convex_bal
    parser.add_argument('--sigma_blend', type=float, default=0.8,
                        help='initial blend weight for accepting sigma update candidates')
    parser.add_argument('--sigma_blend_min', type=float, default=0.05,
                        help='minimum blend weight for late-stage sigma updates')

    # late-stage stabilization for online_convex_bal
    parser.add_argument('--sigma_stabilize_start', type=int, default=80,
                        help='epoch after which deadband-based stabilization starts')
    parser.add_argument('--sigma_deadband', type=float, default=0.1,
                        help='deadband threshold on log primal/delta_y imbalance')

    # numerical stability
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='numerical stability epsilon')

    # wandb
    parser.add_argument('--use_wandb', type=str2bool, default=False,
                        help='whether to log metrics to wandb')
    parser.add_argument('--wandb_project', type=str, default='federated-learning',
                        help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity or team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb run name')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='wandb group name')
    parser.add_argument('--wandb_job_type', type=str, default='train',
                        help='wandb job type')
    
    # hybrid update for adaptive sigma params
    parser.add_argument('--hybrid_alpha', type=float, default=1.0,
                        help='alpha parameter for hybrid convex update of u')
    parser.add_argument('--hybrid_lambda0', type=float, default=1.0,
                        help='lambda0 parameter for hybrid convex update of u')
    parser.add_argument('--hybrid_tau_mag', type=float, default=1.0,
                        help='tau_mag parameter for hybrid convex update of u')


    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para

def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):

    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2-pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model = None, prev_model_pool = None, round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        prev_models=[]
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list



def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    
    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            # dir="/data/yutong/wandb",
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            config=vars(args)
        )

    seed = args.init_seed
    #logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    #logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    #print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)


    '''print("=== client dataset sizes ===")
    for client_id, idxs in net_dataidx_map.items():
         print(f"  client {client_id:2d}: {len(idxs):4d} samples")'''


    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    # 2) Pure sequential non-IID batch training with Adam
    if args.alg == 'sequential':
        # prepare the global full training dataset (already loaded earlier)
        # train_ds_global is from: get_dataloader(...)[2]
        # wrap each client's indices into a Subset

        # single model (same network for all), single optimizer
        '''client_subsets = [
            torch.utils.data.Subset(train_ds_global, net_dataidx_map[c])
            for c in range(args.n_parties)
        ]

        batches = split_and_aggregate_minibatches(client_subsets, num_splits=10)
        nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args)
        model = nets[0].to(device)

        num_gpu = 10
        epoches = 10
        W_n_0 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]    
        W_b_initial = [[param.clone() for param in W_n_0] for _ in range(num_gpu)]
        P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(num_gpu)]
        accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(num_gpu)]

        sigma_lr = args.sigma_lr
        rho_lr = args.rho_lr
        l2_lambda = args.l2_lambda



        #alpha_b = [1/3, 1/3, 1/3]
        alpha_b = [1/num_gpu for _ in range(num_gpu)]
        W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b, l2_lambda)
        epsilon = 1e-8
        updated_iteration = 1.0
        beta_rmsprop = 0.999 # 0.99 not sure which one is better
        criterion = nn.CrossEntropyLoss().to(device)
        test_loader = data.DataLoader(dataset=test_ds_global, batch_size=128, shuffle=False)
        test_record = []



        # Print ratio info for each aggregated global minibatch
        for epoch in range(args.epochs):
            print(f"\n>>> Epoch {epoch}/{args.epochs-1}")
            random.shuffle(batches) # shuffle
            for ii, (batch_list, alpha_b) in enumerate(batches):
                total_train_loss = 0
                #for j, r in enumerate(ratios):
                for sb in range(num_gpu):
                    loader = DataLoader(batch_list[sb], batch_size=len(batch_list[sb]), shuffle=False)
                    with torch.no_grad():
                        for param, w in zip(model.parameters(), W_global):
                                param.copy_(w)

                    W_n = W_b_initial[sb]
                    P_n = P_b_initial[sb]
                    accumulators = accumulators_initial[sb]
                    for x, y in loader:
                        x, y = x.to(device), y.to(device).long()
                        out = model(x)
                        loss = criterion(out, y)
                    total_train_loss += loss.item()

                    zero_grad(model.parameters())
                    #optimizer.zero_grad()
                    loss.backward()
                    gradients = [param.grad for param in model.parameters()]

                    with torch.no_grad():

                        for i, (param_wn, param_pn, gradient, param_wg, accumulator) in enumerate(zip(W_n, P_n, gradients, W_global, accumulators)):
                            #velocity.mul_(args.beta1).add_((1 - args.beta1) * (gradient + param_pn))
                            accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * (gradient + param_pn).pow(2))
                            #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                            #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
                            
                            
                            bias_correction2 = 1 - beta_rmsprop** updated_iteration                        
                            corrected_accumulator = accumulator / (bias_correction2)
                            #bias_correction1 = 1 - args.beta1** updated_iteration                        
                            #corrected_velocity= velocity / (bias_correction1)
                            
                            #delta = param_wg -  (gradient+ param_pn)/(sigma_lr_current + rho_lr_current*(torch.sqrt(corrected_accumulator) + args.eps))
                            delta = param_wg -  (gradient+ param_pn)/(sigma_lr+ rho_lr * (torch.sqrt(corrected_accumulator) + epsilon))
                            
                            param_wn.copy_(delta.detach())
                            param_pn.add_(sigma_lr * (param_wn - param_wg))

                    del loss
                    del out
                    
                updated_iteration += 1
                        
                    
                with torch.no_grad():
                    #W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b_n[(update_count-num_gpu):update_count])
                    W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b, l2_lambda)
                    for param, w in zip(model.parameters(), W_global):
                        param.copy_(w)

                print(f"client {ii} epoch Average training loss: {total_train_loss/num_gpu}")

            test_acc = compute_accuracy(model, test_loader, get_confusion_matrix=False, device=device)
            test_record.append(test_acc)
            print(f"\n>>> test accuracy: {test_acc:.2%}")

        print(test_record)
                    

        exit(0)

        train_all_in_list = []
        test_all_in_list = []
        if args.noise > 0:
            for party_id in range(args.n_parties):
                dataidxs = net_dataidx_map[party_id]

                noise_level = args.noise
                if party_id == args.n_parties - 1:
                    noise_level = 0

                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
                else:
                    noise_level = args.noise / (args.n_parties - 1) * party_id
                    train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
                train_all_in_list.append(train_ds_local)
                test_all_in_list.append(test_ds_local)
            train_all_in_ds = data.ConcatDataset(train_all_in_list)
            train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
            test_all_in_ds = data.ConcatDataset(test_all_in_list)
            test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)'''


    elif args.alg == 'sisa':
        nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args)
        model = nets[0].to(device)

        W_n_0 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]
        W_b_initial = [[param.clone() for param in W_n_0] for _ in range(args.n_parties)]
        P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.n_parties)]
        accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.n_parties)]

        sigma_init = args.sigma_lr
        sigma_b = [float(sigma_init) for _ in range(args.n_parties)]
        u_sigma_b = [
            torch.tensor(math.log(max(sigma_init, 1e-12)), device=device)
            for _ in range(args.n_parties)
        ]

        rho_lr = args.rho_lr
        l2_lambda = args.l2_lambda

        sigma_mode = getattr(args, "sigma_mode", "fixed")
        sigma_min = getattr(args, "sigma_min", 1e-6)
        sigma_max = getattr(args, "sigma_max", 1e6)
        eta_u = getattr(args, "eta_u", 1e-3)
        G_clip = getattr(args, "G_clip", 1.0)
        sigma_update_freq = getattr(args, "sigma_update_freq", 1)

        # base trust-region size in log-space, will shrink over time
        sigma_max_delta = getattr(args, "sigma_max_delta", 0.2)
        sigma_max_delta_min = getattr(args, "sigma_max_delta_min", 0.01)

        # EMA smoothing for convex_bal
        sigma_ema_beta = getattr(args, "sigma_ema_beta", 0.9)

        # late-stage stabilization
        sigma_stabilize_start = getattr(args, "sigma_stabilize_start", 80)
        sigma_deadband = getattr(args, "sigma_deadband", 0.1)

        # blending schedule: aggressive early, damped later
        sigma_blend = getattr(args, "sigma_blend", 0.8)
        sigma_blend_min = getattr(args, "sigma_blend_min", 0.05)

        total_data_points = sum(len(net_dataidx_map[r]) for r in range(args.n_parties))
        alpha_b = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]

        W_global = generate_W_global(
            args.n_parties, W_b_initial, P_b_initial, sigma_b, alpha_b, l2_lambda
        )
        W_global_prev = [w.clone().detach() for w in W_global]

        epsilon = 1e-8
        updated_iteration = 1.0
        beta_rmsprop = 0.999
        criterion = nn.CrossEntropyLoss().to(device)
        test_record = []

        # EMA trackers for online_convex_bal
        primal_res_ema = [None for _ in range(args.n_parties)]
        delta_y_ema = [None for _ in range(args.n_parties)]

        for epoch in range(args.comm_round):
            epoch_train_correct = 0
            epoch_train_total = 0
            epoch_train_loss = 0.0

            sigma_before_epoch = sigma_b.copy()

            primal_res_hist = [0.0 for _ in range(args.n_parties)]
            dual_res_hist = [0.0 for _ in range(args.n_parties)]
            delta_y_hist = [0.0 for _ in range(args.n_parties)]

            epoch_primal_res = []
            epoch_dual_res = []
            epoch_delta_y = []

            for sb in range(args.n_parties):
                dataidxs = net_dataidx_map[sb]
                noise_level = args.noise
                if sb == args.n_parties - 1:
                    noise_level = 0

                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_dataloader(
                        args.dataset, args.datadir, args.batch_size, 32,
                        dataidxs, noise_level, sb, args.n_parties - 1
                    )
                else:
                    noise_level = args.noise / (args.n_parties - 1) * sb
                    train_dl_local, test_dl_local, _, _ = get_dataloader(
                        args.dataset, args.datadir, args.batch_size, 32,
                        dataidxs, noise_level
                    )

                if isinstance(train_dl_local, list):
                    train_dataloader = train_dl_local
                else:
                    train_dataloader = [train_dl_local]

                with torch.no_grad():
                    for param, w in zip(model.parameters(), W_global):
                        param.copy_(w)

                W_n = W_b_initial[sb]
                P_n = P_b_initial[sb]
                accumulators = accumulators_initial[sb]

                W_n_prev = [w.clone().detach() for w in W_n]

                zero_grad(model.parameters())

                for tmp in train_dataloader:
                    for batch_idx, (x, target) in enumerate(tmp):
                        x, target = x.to(device), target.to(device)
                        x.requires_grad = True
                        target = target.long()

                        out = model(x)
                        loss = criterion(out, target)

                        with torch.no_grad():
                            pred = torch.argmax(out, dim=1)
                            epoch_train_correct += (pred == target).sum().item()
                            epoch_train_total += target.size(0)
                            epoch_train_loss += loss.item() * target.size(0)

                        loss.backward()

                gradients = [param.grad for param in model.parameters()]

                with torch.no_grad():
                    current_sigma = sigma_b[sb]
                    current_rho = rho_lr

                    for i, (param_wn, param_pn, gradient, param_wg, accumulator) in enumerate(
                        zip(W_n, P_n, gradients, W_global, accumulators)
                    ):
                        accumulator.mul_(beta_rmsprop).add_(
                            (1 - beta_rmsprop) * (gradient + param_pn).pow(2)
                        )

                        bias_correction2 = 1 - beta_rmsprop ** updated_iteration
                        corrected_accumulator = accumulator / bias_correction2

                        delta = param_wg - (gradient + param_pn) / (
                            current_sigma + current_rho * (torch.sqrt(corrected_accumulator) + epsilon)
                        )

                        param_wn.copy_(delta.detach())
                        param_pn.add_(current_sigma * (param_wn - param_wg))

                    # primal_res = global_norm([a - b for a, b in zip(W_n, W_global)])
                    # dual_res = current_sigma * global_norm([a - b for a, b in zip(W_global, W_global_prev)])
                    # delta_y = global_norm([a - b for a, b in zip(W_n, W_n_prev)])

                    # primal_res_hist[sb] = primal_res.item()
                    # dual_res_hist[sb] = dual_res.item()
                    # delta_y_hist[sb] = delta_y.item()
                
                    # epoch_primal_res.append(primal_res.item())
                    # epoch_dual_res.append(dual_res.item())
                    # epoch_delta_y.append(delta_y.item())
                    
                    primal_res = global_norm([a - b for a, b in zip(W_n, W_global)])
                    # dual_base = global_norm([a - b for a, b in zip(W_global, W_global_prev)])
                    dual_base = global_norm([a - b for a, b in zip(W_n, W_n_prev)])
                    dual_res = current_sigma * dual_base

                    primal_res_hist[sb] = primal_res.item()
                    dual_res_hist[sb] = dual_res.item()
                    delta_y_hist[sb] = dual_base.item()   # keep the same container name if you want

                    epoch_primal_res.append(primal_res.item())
                    epoch_dual_res.append(dual_res.item())
                    epoch_delta_y.append(dual_base.item())

                # updated_iteration += 1
                zero_grad(model.parameters())
                del loss
                del out

            with torch.no_grad():
                W_global_prev = [w.clone().detach() for w in W_global]
                W_global = generate_W_global(
                    args.n_parties, W_b_initial, P_b_initial, sigma_b, alpha_b, l2_lambda
                )
                for param, w in zip(model.parameters(), W_global):
                    param.copy_(w)

            avg_primal_res = sum(alpha_b[i] * epoch_primal_res[i] for i in range(args.n_parties))
            avg_dual_res = sum(alpha_b[i] * epoch_dual_res[i] for i in range(args.n_parties))
            avg_delta_y = sum(alpha_b[i] * epoch_delta_y[i] for i in range(args.n_parties))

            sigma_loss_list = []
            sigma_target_list = []
            sigma_grad_u_list = []

            if epoch % 10 == 0 and 0 < epoch:
                if sigma_mode == "heuristic" and ((epoch + 1) % sigma_update_freq == 0):
                    for sb in range(args.n_parties):
                        sigma_new = heuristic_update_sigma(
                            sigma_b[sb],
                            epoch_primal_res[sb],
                            epoch_dual_res[sb],
                            mu=getattr(args, "sigma_mu", 10.0),
                            tau=getattr(args, "sigma_tau", 2.0),
                        )
                        sigma_b[sb] = float(max(sigma_min, min(sigma_max, sigma_new)))

                elif sigma_mode == "online_balance" and ((epoch + 1) % sigma_update_freq == 0):
                    for sb in range(args.n_parties):
                        old_u = u_sigma_b[sb].detach()
                        u = old_u.clone().requires_grad_(True)

                        u_new, sigma_loss = online_ogd_update_u(
                            u,
                            torch.tensor(epoch_primal_res[sb], device=device),
                            torch.tensor(epoch_dual_res[sb], device=device),
                            eta_u=eta_u / (epoch + 1),
                            G_clip=G_clip,
                            u_min=math.log(sigma_min),
                            u_max=math.log(sigma_max),
                        )

                        with torch.no_grad():
                            u_new = torch.clamp(
                                u_new,
                                min=old_u - sigma_max_delta,
                                max=old_u + sigma_max_delta,
                            )

                        u_sigma_b[sb] = u_new.detach()
                        sigma_b[sb] = float(torch.exp(u_sigma_b[sb]).item())
                        sigma_loss_list.append(sigma_loss.item())

                elif sigma_mode == "online_convex_bal" and ((epoch + 1) % sigma_update_freq == 0):
                    for sb in range(args.n_parties):
                        old_u = u_sigma_b[sb].detach()

                        # EMA smoothing of signals
                        cur_primal = float(epoch_primal_res[sb])
                        cur_delta_y = float(epoch_delta_y[sb])

                        if primal_res_ema[sb] is None:
                            primal_res_ema[sb] = cur_primal
                            delta_y_ema[sb] = cur_delta_y
                        else:
                            primal_res_ema[sb] = (
                                sigma_ema_beta * primal_res_ema[sb]
                                + (1.0 - sigma_ema_beta) * cur_primal
                            )
                            delta_y_ema[sb] = (
                                sigma_ema_beta * delta_y_ema[sb]
                                + (1.0 - sigma_ema_beta) * cur_delta_y
                            )

                        primal_smooth = primal_res_ema[sb]
                        delta_y_smooth = delta_y_ema[sb]
                        eps_val = getattr(args, "eps", 1e-12)

                        imbalance = math.log(
                            (primal_smooth + eps_val) / (delta_y_smooth + eps_val)
                        )

                        # late-stage deadband: once nearly balanced, stop twitching
                        if epoch >= sigma_stabilize_start and abs(imbalance) < sigma_deadband:
                            u_new = old_u
                            sigma_loss = torch.tensor(0.0, device=device)
                            sigma_target = torch.tensor(imbalance, device=device)
                            grad_u = torch.tensor(0.0, device=device)
                        else:
                            u_candidate, sigma_loss, sigma_target, grad_u = online_convex_bal_update_u(
                                u=old_u,
                                primal_res=torch.tensor(primal_smooth, device=device),
                                delta_y=torch.tensor(delta_y_smooth, device=device),
                                eta_u=eta_u / math.sqrt(epoch + 1),  # slower decay than 1/(epoch+1)
                                G_clip=G_clip,
                                u_min=math.log(sigma_min),
                                u_max=math.log(sigma_max),
                                eps=eps_val,
                            )

                            with torch.no_grad():
                                # shrinking trust region in log-sigma space
                                max_delta_k = max(
                                    sigma_max_delta_min,
                                    sigma_max_delta / math.sqrt(epoch + 1)
                                )
                                u_candidate = torch.clamp(
                                    u_candidate,
                                    min=old_u - max_delta_k,
                                    max=old_u + max_delta_k,
                                )

                                # aggressive early, damped later
                                blend_k = max(
                                    sigma_blend_min,
                                    sigma_blend / math.sqrt(epoch + 1)
                                )
                                u_new = (1.0 - blend_k) * old_u + blend_k * u_candidate

                                # final projection
                                u_new = torch.clamp(
                                    u_new,
                                    min=math.log(sigma_min),
                                    max=math.log(sigma_max),
                                )

                        u_sigma_b[sb] = u_new.detach()
                        sigma_b[sb] = float(torch.exp(u_sigma_b[sb]).item())

                        sigma_loss_list.append(float(sigma_loss.item()))
                        sigma_target_list.append(float(sigma_target.item()))
                        sigma_grad_u_list.append(float(grad_u.item()))
                elif sigma_mode == "online_convex_bal_debug" and ((epoch + 1) % sigma_update_freq == 0):
                    for sb in range(args.n_parties):    
                        old_u = u_sigma_b[sb].detach()
                        eps_val = getattr(args, "eps", 1e-12)

                        cur_primal = float(epoch_primal_res[sb])
                        cur_delta_y = float(epoch_delta_y[sb])   # now this is dual_base

                        u_new, sigma_loss, sigma_target, grad_u = online_convex_bal_update_u(
                            u=old_u,
                            primal_res=torch.tensor(cur_primal, device=device),
                            delta_y=torch.tensor(cur_delta_y, device=device),
                            eta_u=eta_u / math.sqrt(epoch + 1),
                            G_clip=G_clip,
                            u_min=math.log(sigma_min),
                            u_max=math.log(sigma_max),
                            eps=eps_val,
                        )

                        u_sigma_b[sb] = u_new.detach()
                        sigma_b[sb] = float(torch.exp(u_sigma_b[sb]).item())

                        # print(
                        #     f"client {sb} "
                        #     f"u={old_u.item():.4f} "
                        #     f"sigma={math.exp(old_u.item()):.4e} "
                        #     f"primal={cur_primal:.4e} "
                        #     f"dual_base={cur_delta_y:.4e} "
                        #     f"target={sigma_target.item():.4f} "
                        #     f"grad={grad_u.item():.4f}"
                        # )
                elif sigma_mode == "online_hybrid" and ((epoch + 1) % sigma_update_freq == 0):
                    for sb in range(args.n_parties):
                        old_u = u_sigma_b[sb].detach()

                        cur_primal = float(epoch_primal_res[sb])
                        cur_delta_y = float(epoch_delta_y[sb])

                        # EMA smoothing, same as before
                        if primal_res_ema[sb] is None:
                            primal_res_ema[sb] = cur_primal
                            delta_y_ema[sb] = cur_delta_y
                        else:
                            primal_res_ema[sb] = (
                                sigma_ema_beta * primal_res_ema[sb]
                                + (1.0 - sigma_ema_beta) * cur_primal
                            )
                            delta_y_ema[sb] = (
                                sigma_ema_beta * delta_y_ema[sb]
                                + (1.0 - sigma_ema_beta) * cur_delta_y
                            )

                        primal_smooth = primal_res_ema[sb]
                        delta_y_smooth = delta_y_ema[sb]
                        eps_val = getattr(args, "eps", 1e-12)

                        u_candidate, sigma_loss, sigma_target, grad_u, lambda_k = online_convex_hybrid_update_u(
                            u_old=old_u,
                            primal_res=torch.tensor(primal_smooth, device=device),
                            delta_y=torch.tensor(delta_y_smooth, device=device),
                            eta_u=eta_u / math.sqrt(epoch + 1),
                            alpha=getattr(args, "hybrid_alpha", 1.0),
                            lambda0=getattr(args, "hybrid_lambda0", 1.0),
                            tau_mag=getattr(args, "hybrid_tau_mag", 0.1),
                            G_clip=G_clip,
                            u_min=math.log(sigma_min),
                            u_max=math.log(sigma_max),
                            eps=eps_val,
                            target_clip=True,
                        )

                        with torch.no_grad():
                            # trust region
                            # max_delta_k = max(
                            #     sigma_max_delta_min,
                            #     sigma_max_delta / math.sqrt(epoch + 1)
                            # )
                            # u_candidate = torch.clamp(
                            #     u_candidate,
                            #     min=old_u - max_delta_k,
                            #     max=old_u + max_delta_k,
                            # )

                            # damped blending
                            blend_k = max(
                                sigma_blend_min,
                                sigma_blend / math.sqrt(epoch + 1)
                            )
                            u_new = (1.0 - blend_k) * old_u + blend_k * u_candidate

                            # optional deadband on target mismatch
                            if epoch >= sigma_stabilize_start and abs((old_u - sigma_target).item()) < sigma_deadband:
                                u_new = old_u

                            u_new = torch.clamp(
                                u_new,
                                min=math.log(sigma_min),
                                max=math.log(sigma_max),
                            )

                        u_sigma_b[sb] = u_new.detach()
                        sigma_b[sb] = float(torch.exp(u_sigma_b[sb]).item())

                        sigma_loss_list.append(float(sigma_loss.item()))
                        sigma_target_list.append(float(sigma_target.item()))
                        sigma_grad_u_list.append(float(grad_u.item()))

                elif sigma_mode == "fixed":
                    pass

            train_acc = epoch_train_correct / max(epoch_train_total, 1)
            train_loss = epoch_train_loss / max(epoch_train_total, 1)

            test_acc = compute_accuracy(model, test_dl_global, get_confusion_matrix=False, device=device)
            test_record.append(test_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            if getattr(args, "use_wandb", False):
                log_dict = {
                    "train/acc": train_acc,
                    "train/loss": train_loss,
                    "test/acc": test_acc,
                    "primal_res/avg": avg_primal_res,
                    "dual_res/avg": avg_dual_res,
                    "delta_y/avg": avg_delta_y,
                    "rho_fixed": rho_lr,
                    "sigma/avg": sum(alpha_b[i] * sigma_b[i] for i in range(args.n_parties)),
                    "sigma/min": min(sigma_b),
                    "sigma/max": max(sigma_b),
                    
                }
                
                log_dict[f"ratio_primal_over_dualbase/client_{i}"] = primal_res_hist[i] / max(delta_y_hist[i], 1e-12)
                log_dict[f"log_ratio_primal_over_dualbase/client_{i}"] = math.log(max(primal_res_hist[i], 1e-12)) - math.log(max(delta_y_hist[i], 1e-12))
                log_dict[f"target_exp/client_{i}"] = math.exp(sigma_target_list[i]) if i < len(sigma_target_list) else None

                for i in range(args.n_parties):
                    log_dict[f"primal_res/client_{i}"] = primal_res_hist[i]
                    log_dict[f"dual_res/client_{i}"] = dual_res_hist[i]
                    log_dict[f"delta_y/client_{i}"] = delta_y_hist[i]
                    log_dict[f"sigma/client_{i}"] = sigma_b[i]
                    log_dict[f"log_sigma/client_{i}"] = math.log(max(sigma_b[i], 1e-12))
                    log_dict[f"sigma_change/client_{i}"] = sigma_b[i] - sigma_before_epoch[i]
                    log_dict[f"primal_over_dualbase/client_{i}"] = primal_res_hist[i] / max(delta_y_hist[i], 1e-12)
                    log_dict[f"log_primal_over_dualbase/client_{i}"] = math.log(max(primal_res_hist[i], 1e-12)) - math.log(max(delta_y_hist[i], 1e-12))
                    log_dict[f"sigma_times_dualbase/client_{i}"] = sigma_b[i] * delta_y_hist[i]
                    log_dict[f"primal_minus_sigma_dualbase/client_{i}"] = primal_res_hist[i] - sigma_b[i] * delta_y_hist[i]

                    if primal_res_ema[i] is not None:
                        log_dict[f"primal_res_ema/client_{i}"] = primal_res_ema[i]
                    if delta_y_ema[i] is not None:
                        log_dict[f"delta_y_ema/client_{i}"] = delta_y_ema[i]

                if len(sigma_loss_list) > 0:
                    log_dict["sigma_loss/avg"] = sum(sigma_loss_list) / len(sigma_loss_list)

                if len(sigma_target_list) > 0:
                    log_dict["sigma_target/avg"] = sum(sigma_target_list) / len(sigma_target_list)

                if len(sigma_grad_u_list) > 0:
                    log_dict["sigma_grad_u/avg"] = sum(sigma_grad_u_list) / len(sigma_grad_u_list)

                wandb.log(log_dict, step=epoch)

        print('######################################################')
        print('The highest test accuracy is:', max(test_record))
        print('######################################################')


    elif args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)


    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))


            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            total_n = sum(n_list)
            #print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    #if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    #else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n


            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, moon_model=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)

   
