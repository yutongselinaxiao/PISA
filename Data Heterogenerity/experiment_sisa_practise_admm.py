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
import wandb

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


def generate_W_global(num_batches, W_n_list, P_n_list, tau_lr, alpha, l2_lambda):
    # --- Alpha on whole augmented Lagrangian term ---
    # Derived from: min_w lambda/2 ||w||^2 + sum_i alpha_i [<pi_i, w_i - w> + sigma/2 ||w_i - w||^2]
    # Global update: w = (sigma * sum_i(alpha_i * w_i) + sum_i(alpha_i * pi_i)) / (lambda + sigma * sum_i(alpha_i))
    # Since sum_i(alpha_i) = 1, denominator = lambda + sigma
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = (tau_lr*W_n_avg[i] + P_n_avg[i]) / (tau_lr + l2_lambda)
        W_n_avg[i].detach()

    # --- [COMMENTED OUT] Alpha only on F_i (not on dual/quad terms) ---
    # Derived from: min_w lambda/2 ||w||^2 + sum_i [<pi_i, w_i - w> + sigma/2 ||w_i - w||^2]
    # Global update: w = (sigma * sum_i(w_i) + sum_i(pi_i)) / (lambda + m * sigma)
    # Difference: here alpha does NOT weight the consensus penalty, so the global
    # update uses plain sums instead of alpha-weighted averages.
    # W_n_sum = [torch.zeros_like(var) for var in W_n_list[0]]
    # P_n_sum = [torch.zeros_like(var) for var in P_n_list[0]]
    # for i in range(num_batches):
    #     for j in range(len(W_n_sum)):
    #         W_n_sum[j] = W_n_sum[j] + W_n_list[i][j]
    #         P_n_sum[j] = P_n_sum[j] + P_n_list[i][j]
    # for j in range(len(W_n_sum)):
    #     W_n_sum[j] = (tau_lr * W_n_sum[j] + P_n_sum[j]) / (l2_lambda + num_batches * tau_lr)
    #     W_n_sum[j].detach()

    return W_n_avg

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="the optimizer for the local solve: 'sgd', 'adam', "
                             "'amsgrad', 'adam_warmstart' (Adam with m, v, t "
                             "persisted across ADMM rounds), 'adamw_admm_explicit' "
                             "(Adam on task gradient only; ADMM regularizer applied "
                             "as a decoupled per-batch/epoch/round step a la "
                             "AdamW; cold m, v, t each round), or "
                             "'adamw_admm_explicit_warmstart' (same decoupled "
                             "regularizer but persists Adam's m, v, t across "
                             "rounds).")
    parser.add_argument('--local_init', type=str, default='reset',
                        choices=['reset', 'warm'],
                        help="How to initialize the local solve at each ADMM round: "
                             "'reset' resets w_i to w_global^k each round (default; "
                             "matches classical ADMM x-subproblem); 'warm' warm-starts "
                             "from w_i^{k-1}. Empirically warm regressed 36/70 cells in "
                             "Apr 2026 -- kept as opt-in for future work.")
    # NOTE (2026-04-27, REMOVED): --admm_reg_lr was used for the AdamW-style
    # decoupled-regularizer optimizers (adamw_admm_explicit, adamw_admm_implicit).
    # Both variants were reverted -- see local_admm_train docstring CHANGE LOG.
    #
    # NOTE (2026-05-02, RE-INTRODUCED without the knob): adamw_admm_explicit
    # and adamw_admm_explicit_warmstart now hard-code eta_r = args.lr / max(sigma, 1)
    # applied per batch. The sigma-invariant scaling (inv_sigma) is the
    # fundamental fix for the prior collapse pathology -- it is not a tunable
    # because eta_r * alpha * sigma without inv_sigma diverges at our sigma_0
    # range. No new flags. See local_admm_train docstring CHANGE LOG entry.
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--sigma_lr', type=float, default=1.5e0, help='hyperparameter sigma in sisa')
    parser.add_argument('--rho_lr', type=float, default=1e3, help='hyperparameter rho in sisa')
    parser.add_argument('--l2_lambda', type=float, default=1e-3, help='hyperparameter l2_lambda in sisa')
    parser.add_argument('--mu_lr', type=float, default=0.997, help='hyperparameter mu in sisa')
    parser.add_argument('--decay_epoch', type=float, default=5, help='hyperparameter decay_epoch in sisa')
    parser.add_argument('--terminate_decay', type=float, default=50, help='hyperparameter stop sigma decay in sisa')
    
    parser.add_argument('--sigma_mode', type=str, default='fixed',
                        help='fixed / online_convex_bal / online_convex_bal_lipschitz / '
                             'heuristic / online_task_aware')
    parser.add_argument('--sigma_min', type=float, default=1e-6,
                        help='minimum sigma')
    parser.add_argument('--sigma_max', type=float, default=1e4,
                        help='maximum sigma')
    parser.add_argument('--eta_u', type=float, default=0.1,
                        help='stepsize for online update of log(sigma)')
    parser.add_argument('--G_clip', type=float, default=10.0,
                        help='gradient clip for online sigma update')
    parser.add_argument('--sigma_update_freq', type=int, default=1,
                        help='update sigma every this many communication rounds')
    parser.add_argument('--sigma_ema_beta', type=float, default=0.9,
                        help='EMA smoothing for primal/dual signals')
    parser.add_argument('--sigma_mu', type=float, default=10.0,
                        help='threshold ratio for heuristic sigma update (He et al. 2000)')
    parser.add_argument('--sigma_tau', type=float, default=2.0,
                        help='multiplicative factor for heuristic sigma update')
    parser.add_argument('--sigma_kmax', type=int, default=50,
                        help='stop heuristic adjustment after this many rounds (sum tau_k < inf)')
    parser.add_argument('--eps', type=float, default=1e-12,
                        help='numerical epsilon')
    parser.add_argument('--task_lambda', type=float, default=1.0,
                        help='weight of task-awareness term in online_task_aware sigma mode')

    # ---- OGD step-size schedule (consumed by online_convex_bal_lipschitz) ----
    parser.add_argument('--eta_u_decay', type=str, default='none',
                        choices=['none', 'inverse', 'inv_sqrt', 'textbook_sc'],
                        help='OGD step schedule on u=log(sigma). textbook_sc = '
                             '1/(2*k) (parameter-free, ignores --eta_u). k counts '
                             'sigma-update events, not rounds.')

    # ---- BB Lipschitz floor (consumed by online_convex_bal_lipschitz) ----
    parser.add_argument('--lipschitz_estimator', type=str, default='ema',
                        choices=['ema', 'running_min', 'running_median'],
                        help='smoothing rule for BB-type Lipschitz estimate.')
    parser.add_argument('--lipschitz_window_size', type=int, default=20,
                        help='ring buffer size for running_min / running_median')
    parser.add_argument('--lipschitz_ema_beta', type=float, default=0.9,
                        help='EMA beta for L_hat smoothing')
    parser.add_argument('--lipschitz_min_dz', type=float, default=1e-6,
                        help='minimum ||z^k - z^{k-1}|| for a usable BB sample')
    parser.add_argument('--lipschitz_max', type=float, default=1e8,
                        help='upper clip on raw BB L_hat per round')
    parser.add_argument('--lipschitz_floor_alpha', type=float, default=1.0,
                        help='hard projection floor: sigma >= alpha * L_hat. '
                             'alpha=1 reproduces the canonical hard projection.')

    # Polyak-style EMA on the local iterate w during local training. Returned
    # to the global aggregation in place of the raw post-training w. Designed
    # to smooth out per-batch class bias on extreme non-iid splits (label1
    # cells), where each batch sees a single class and the raw w bounces
    # widely between class-specific local minima. β = 0 (default) disables;
    # β ∈ (0, 1) maintains w_ema ← β·w_ema + (1−β)·w after each batch step
    # (post optimizer.step and post-explicit-shrinkage), and returns w_ema as
    # the client's local iterate for the round.
    parser.add_argument('--local_weight_ema_beta', type=float, default=0.0,
                        help='Polyak EMA on local iterate w during local training. '
                             '0 = disabled (default); 0.99 = aggressive smoothing. '
                             'Affects only local_admm_train; the σ-rule and '
                             'global aggregation are unchanged.')

    # σ-coupled-but-bounded consensus step for adamw_admm_explicit{_warmstart}.
    # When cap = 0 (default), uses the σ-invariant rate eta_r = lr/σ — the
    # 2026-05-02 fix that prevented collapse at large σ but also unhooks σ
    # from consensus strength entirely. When cap > 0, uses
    #     eta_r = min(lr, cap / (alpha_i * sigma))
    # so the per-batch (w − w_g) coefficient becomes
    #     eta_r * alpha_i * sigma = min(lr * alpha_i * sigma, cap)
    # i.e. it scales linearly with σ until it saturates at `cap`. At σ above
    # the crossover, the cap (cap / batch) is the effective per-batch
    # consensus pull — strong enough to enforce real consensus over a local
    # round (~30 batches), but bounded away from the prior 100%/batch
    # collapse pathology. The π coefficient becomes eta_r * alpha_i =
    # min(lr*alpha, cap/sigma), so π's contribution is also damped at large σ
    # in the same way the σ-invariant fix used to do for the entire step.
    # Intended for label1 cells where AdamW-explicit's σ-decoupled shrinkage
    # currently fails. cap = 0.1 (10%/batch) is a reasonable starting value.
    parser.add_argument('--adamw_consensus_cap', type=float, default=0.0,
                        help='Bound on per-batch consensus rate for '
                             'adamw_admm_explicit{_warmstart}. 0 = disabled '
                             '(keep σ-invariant lr/σ scheme); 0.1 = cap at '
                             '10%%/batch. See docstring of local_admm_train.')

    parser.add_argument('--use_wandb', type=bool, default=False, help='whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='sisa-adaptive-sigma', help='wandb project name')
    parser.add_argument('--wandb_group', type=str, default='default-group', help='wandb group name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--local_log_dir', type=str, default=None,
                        help='Directory to save per-round metrics as CSV for offline plotting. '
                             'Disabled when None.')
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

import math

def clone_params(param_list):
    return [p.detach().clone() for p in param_list]


def params_l2_sq(param_list):
    total = None
    for p in param_list:
        val = torch.sum(p.detach() * p.detach())
        total = val if total is None else total + val
    return total


def diff_global_norm(list_a, list_b):
    total = None
    for a, b in zip(list_a, list_b):
        diff = (a.detach() - b.detach())
        val = torch.sum(diff * diff)
        total = val if total is None else total + val
    return torch.sqrt(total)


def global_norm(tensors):
    """L2 norm of the concatenated tensor list (skipping Nones)."""
    total = None
    for t in tensors:
        if t is None:
            continue
        v = t.detach()
        val = (v * v).sum()
        total = val if total is None else total + val
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total + 1e-12)


def compute_task_grad_at_z(model, train_dl_local, w_global, criterion, device):
    """Compute the raw task-gradient sum (no 1/n_batches scaling) at z^k = w_global.

    Used by the BB Lipschitz floor in online_convex_bal_lipschitz mode.
    Mirrors the implicit accumulation in _online.py line 1631-1633: the
    BB ratio ||dg||/||dz|| is scale-invariant, and consistency across
    rounds requires the same per-client dataloader pass each round.

    Side effect: leaves model parameters set to w_global (the local
    solver will reset again on entry, so this is safe). Caller is
    responsible for switching the model back to train mode after the
    Adam-based local solve.
    """
    with torch.no_grad():
        for p, wg in zip(model.parameters(), w_global):
            p.copy_(wg)
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for x, target in train_dl_local:
        x, target = x.to(device), target.to(device).long()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
    return [p.grad.detach().clone() if p.grad is not None else None
            for p in model.parameters()]


def local_admm_train(model, train_dl_local, w_global, pi_local, sigma_lr, args, device="cpu", alpha_i=1.0,
                     optimizer_state=None):
    """
    Approximately solve:

        min_wi alpha_i * F_i(wi) + <pi_i, wi - w_global> + (sigma/2)||wi - w_global||^2

    Initialization for the local solve is controlled by args.local_init:
      - 'reset' (default): reset model params to w_global^k at the start of
        each round. The caller's warm-start (W_b_initial[sb] copied into
        model before this call) is overwritten. This is the original ADMM
        x-subproblem behavior.
      - 'warm': leave the model at whatever the caller set it to, i.e.\
        w_i^{k-1} (the previous round's local solution). Tested but reverted
        as default after empirical regression -- see CHANGE LOG below.

    args.optimizer chooses the inner solver:
      - 'sgd'             : plain SGD with momentum on the augmented Lagrangian.
      - 'adam'            : Adam (fresh m, v each round) on the aug Lagrangian.
      - 'amsgrad'         : AMSGrad variant.
      - 'adam_warmstart'  : Adam with m, v, t persisted across rounds via
        optimizer_state. Adam sees the FULL aug Lagrangian gradient (task +
        dual + quadratic), so sigma's contribution to v cancels sigma out of
        the effective step magnitude (see notes/adam_warmstart_pseudocode.tex).

    CHANGE LOG (2026-04-24, REVERTED same day): briefly used local_init='warm'
    as default, removing the inner reset-to-w_global. Empirical verdict from
    sisa-exact-admm-warmstart vs sisa-exact-admm-sgd-epochs-4-22 (Section 9
    of the dashboard): 36/70 paired cells regressed (mnist_label1 ep=1
    dropped 33-44pp; fmnist_label1 ep=10 dropped 27-29pp). Only 8 cells
    improved (narrow win on ep=10 mnist_label2/3 and fmnist_label3). Reverted
    default to local_init='reset'. Warm-start preserved as opt-in for future
    investigation -- the 8 improvements suggest it might help in a specific
    regime (long local epochs on milder heterogeneity) that we have not yet
    characterized.

    CHANGE LOG (2026-04-27, REVERTED): added two AdamW-style decoupled
    optimizer variants -- 'adamw_admm_explicit' and 'adamw_admm_implicit' --
    intended to fix the sigma-cancellation pathology of 'adam_warmstart' by
    splitting the augmented Lagrangian gradient: Adam sees only the task
    gradient, while the ADMM regularizer alpha*(pi + sigma*(w-w_g)) is
    applied as a decoupled per-batch update. Pseudocode lived in
    notes/adam_warmstart_pseudocode.tex Algorithms 3 and 4.

    Failure mode: applying the regularizer per batch with stepsize
    admm_reg_lr=1e-3 led to a per-batch shrinkage factor toward w_g of
    eta_r * alpha * sigma / (1 + eta_r * alpha * sigma). With alpha~0.1
    and sigma=1e4, this is ~50% per batch, so over ~100 batches per round
    w is yanked to w_g and Adam's task step accumulates nothing. The model
    sat at random-guess accuracy on cifar10 across all sigma. Same issue at
    sigma=1e3 over a longer horizon. The sigma-cancellation in coupled
    Adam was actually PROTECTIVE -- it normalized the regularizer contribution
    down to ~lr scale; decoupling exposed its full strength which is too
    strong when applied at every batch. SISA's closed-form update avoids
    this because it computes w from scratch as wg - (g+pi)/(sigma + ...),
    so the step is bounded by 1/sigma rather than scaling linearly with
    sigma. A correct generalization (SISA-with-Adam-preconditioner: w_new
    = wg - (g + pi) / (sigma + rho*sqrt(v_hat) + eps)) was discussed but
    not implemented because the sigma-robustness story is driven by the
    Lipschitz floor + textbook_sc decay (online.py), not the local solver.
    Reverted both adamw_admm_* variants. See pilot file
    generate_and_run_sisa_jobs_adamw_pilot.py for the failed launch.

    CHANGE LOG (2026-05-02, RE-INTRODUCED with sigma-invariant fix): the
    explicit decoupled-regularizer variant is back as 'adamw_admm_explicit'
    (cold m, v, t each round) and 'adamw_admm_explicit_warmstart' (Adam
    state persisted across rounds via optimizer_states, mirroring
    'adam_warmstart' plumbing).

    The fix: the reg-step size is hard-coded to
        eta_r = args.lr / max(sigma, 1)
    so per-batch shrinkage rate
        eta_r * alpha * sigma = args.lr * alpha
    is sigma-invariant (~1e-4 at args.lr=1e-3, alpha~0.1). Cumulative
    shrinkage over ep=10 (~1000 batches) is bounded at ~10%, well below
    the prior ~100% collapse at sigma=1e4. AdamW-faithful: applied per
    batch alongside the Adam step, never per-epoch / per-round (those
    were considered as workarounds but reduce nothing -- per-application
    rate stays the same and the implementation is no longer AdamW). No
    new flags; eta_r is tied to args.lr and the inv_sigma scaling is
    fundamental, not optional.

    Returns:
        (params, avg_loss) for sgd/adam/amsgrad/adamw_admm_explicit
        (params, avg_loss, optimizer_state_dict) for adam_warmstart and
        adamw_admm_explicit_warmstart

    LOCAL POLYAK EMA (2026-05-18): if `args.local_weight_ema_beta > 0`, the
    returned `params` is a per-param EMA `w_ema ← β·w_ema + (1−β)·w` updated
    AFTER each batch's optimizer.step() (and the explicit-shrinkage step for
    AdamW-explicit). Smooths out per-batch class bias on label1 cells where
    each batch sees a single class; the σ-rule and global aggregation see
    the smoothed iterate without modification. β=0 (default) preserves the
    original behavior exactly. Applies to all optimizers; tested primarily
    with adamw_admm_explicit{_warmstart}.

    σ-COUPLED BOUNDED SHRINKAGE (2026-05-18): if `args.adamw_consensus_cap > 0`,
    the AdamW-explicit per-batch reg step switches from the σ-invariant rate
    eta_r = lr / σ to a σ-coupled-but-bounded rate
        eta_r = min(lr, cap / (alpha_i · σ))
    so the per-batch (w − w_g) coefficient becomes
        eta_r · alpha_i · σ  =  min(lr · alpha_i · σ, cap)
    — linear in σ up to the crossover σ* = cap / (lr · alpha_i), then capped
    at `cap` per batch above. Re-establishes σ as a real consensus knob in
    AdamW-explicit, which the σ-invariant fix had removed. Intended for
    label1 cells where σ-invariant shrinkage cannot enforce strong enough
    consensus. cap = 0 (default) preserves the σ-invariant formula exactly.
    """
    # ----- Initialization: reset (default) or warm-start (opt-in) -----
    local_init = getattr(args, "local_init", "reset")
    if local_init == "reset":
        with torch.no_grad():
            for p, wg in zip(model.parameters(), w_global):
                p.copy_(wg)
    elif local_init == "warm":
        # Caller is expected to have copied W_b_initial[sb] into model.
        # Leave parameters as-is.
        pass
    else:
        raise ValueError(f"Unknown local_init: {local_init!r}; expected 'reset' or 'warm'.")

    # ----- Optimizer dispatch -----
    if args.optimizer == 'adam_warmstart':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.rho, weight_decay=args.reg)
    elif args.optimizer == 'adamw_admm_explicit':
        # Cold m, v, t each round. Adam sees task gradient only; the
        # ADMM regularizer is applied as a decoupled per-batch step
        # below (see training loop).
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw_admm_explicit_warmstart':
        # Same decoupled regularizer as adamw_admm_explicit, but Adam's
        # m, v, t persist across ADMM rounds via optimizer_states[sb].
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    is_adamw_decoupled = args.optimizer in (
        'adamw_admm_explicit', 'adamw_admm_explicit_warmstart',
    )

    criterion = nn.CrossEntropyLoss().to(device)
    model.train()

    epoch_loss = 0.0
    n_batches = 0

    # AdamW-decoupled reg-step coefficient.
    #
    # Default (consensus_cap = 0): eta_r = args.lr / max(sigma, 1) gives a
    # sigma-invariant per-batch shrinkage rate eta_r * alpha * sigma =
    # args.lr * alpha (~1e-4). This was the 2026-05-02 fix for the prior
    # collapse pathology at large sigma — but it also unhooks sigma from the
    # consensus pull entirely, which is what fails on label1 cells.
    #
    # Option 1 (consensus_cap > 0): eta_r = min(lr, cap / (alpha * sigma)).
    # Per-batch consensus rate on (w − w_g) becomes
    #     eta_r * alpha * sigma = min(lr * alpha * sigma, cap)
    # i.e. it scales linearly with sigma until it saturates at `cap`. Above
    # the crossover sigma* = cap / (lr * alpha), the cap binds and the
    # per-batch consensus rate is exactly `cap` regardless of how large sigma
    # grows — strong (~cap per batch, cumulative ~95% over ~30 batches at
    # cap=0.1) but bounded away from runaway collapse.
    consensus_cap = float(getattr(args, "adamw_consensus_cap", 0.0))
    if is_adamw_decoupled:
        sigma_safe = max(sigma_lr, 1.0)
        if consensus_cap > 0.0:
            adamw_eta_r = min(
                args.lr,
                consensus_cap / (max(alpha_i, 1e-12) * sigma_safe),
            )
        else:
            adamw_eta_r = args.lr / sigma_safe
    else:
        adamw_eta_r = 0.0

    # Polyak-style EMA on w during local training. β=0 disables; the buffer
    # is initialized to the starting parameters (post reset/warm init) and
    # updated AFTER each batch's optimizer + explicit-shrinkage step. The
    # ema is what's returned to the global aggregation, so the σ-rule sees
    # a smoother per-client iterate. Designed for label1 where per-batch
    # gradients are dominated by a single class.
    local_w_ema_beta = float(getattr(args, "local_weight_ema_beta", 0.0))
    use_local_w_ema = 0.0 < local_w_ema_beta < 1.0
    w_ema = None
    if use_local_w_ema:
        with torch.no_grad():
            w_ema = [p.detach().clone() for p in model.parameters()]

    for _ in range(args.epochs):
        for x, target in train_dl_local:
            x, target = x.to(device), target.to(device).long()

            optimizer.zero_grad()
            out = model(x)
            task_loss = criterion(out, target)

            dual_term = 0.0
            quad_term = 0.0
            for p, wg, pi in zip(model.parameters(), w_global, pi_local):
                diff = p - wg
                dual_term = dual_term + torch.sum(pi * diff)
                quad_term = quad_term + 0.5 * sigma_lr * torch.sum(diff * diff)

            if is_adamw_decoupled:
                # AdamW-style: Adam sees ONLY the task gradient.
                backprop_loss = alpha_i * task_loss
            else:
                # Coupled: Adam sees the full augmented Lagrangian.
                backprop_loss = alpha_i * (task_loss + dual_term + quad_term)

            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if is_adamw_decoupled:
                # Decoupled per-batch ADMM regularizer step (post-Adam):
                #   w <- w - eta_r * alpha_i * (pi + sigma * (w - w_g))
                with torch.no_grad():
                    for p, wg, pi in zip(model.parameters(), w_global, pi_local):
                        p.sub_(adamw_eta_r * alpha_i * (pi + sigma_lr * (p - wg)))

            # Polyak EMA on w after optimizer.step + explicit shrinkage:
            #   w_ema <- β·w_ema + (1−β)·w
            if use_local_w_ema:
                with torch.no_grad():
                    for buf, p in zip(w_ema, model.parameters()):
                        buf.mul_(local_w_ema_beta).add_(
                            (1.0 - local_w_ema_beta) * p.detach()
                        )

            # Logging always reflects the full augmented-Lagrangian loss
            # so train_local_admm_loss_avg is comparable across optimizers.
            full_loss = alpha_i * (task_loss + dual_term + quad_term)
            epoch_loss += float(full_loss.item())
            n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    if use_local_w_ema:
        # Return the smoothed local iterate. The global aggregation and the
        # σ-rule see this; per-batch class bias is averaged out before it
        # hits the residuals.
        params = [buf.detach().clone() for buf in w_ema]
    else:
        params = [p.detach().clone() for p in model.parameters()]

    if args.optimizer in ('adam_warmstart', 'adamw_admm_explicit_warmstart'):
        return params, avg_loss, optimizer.state_dict()
    return params, avg_loss


def heuristic_update_sigma(sigma_old, primal_res, dual_res, mu=10.0, tau=2.0,
                           k=0, k_max=50):
    """
    Strategy S3 from He, Yang & Wang (2000):
    Self-adaptive penalty parameter adjustment for ADMM.

    Compares primal residual (||w_i - w||) vs dual residual (||w^{k+1} - w^k||).
    - If primal >> dual: increase sigma  (need tighter consensus)
    - If dual >> primal: decrease sigma  (penalty too aggressive)

    The adjustment factor is (1 + tau_k) where tau_k = 1 for k <= k_max
    and 0 otherwise, satisfying sum(tau_k) < inf for convergence.

    Args:
        sigma_old: current penalty parameter
        primal_res: primal residual norm
        dual_res: dual residual norm (||w^{k+1} - w^k||)
        mu: threshold ratio (default 10.0)
        tau: multiplicative/divisive factor when k > k_max (uses fixed sigma)
        k: current iteration index
        k_max: stop adjusting after this many rounds (ensures sum tau_k < inf)
    Returns:
        sigma_new: updated penalty parameter
    """
    if k > k_max:
        # tau_k = 0 for k > k_max => no adjustment (convergence guarantee)
        return sigma_old

    # tau_k = 1 for k <= k_max, so factor = 1 + tau_k = tau (default 2.0)
    sigma_new = sigma_old
    if primal_res > mu * dual_res:
        sigma_new = sigma_old * tau
    elif dual_res > mu * primal_res:
        sigma_new = sigma_old / tau
    return sigma_new


def online_convex_bal_update_u(
    u,
    primal_res,
    dual_base,
    eta_u=0.1,
    u_min=math.log(1e-6),
    u_max=math.log(1e4),
    eps=1e-12,
    G_clip=10.0,
):
    """
    Online update for u = log(sigma) with loss
        0.5 * (u - (log(primal_res) - log(dual_base)))^2
    """
    primal_clip = torch.clamp(primal_res.detach(), min=eps)
    dual_clip = torch.clamp(dual_base.detach(), min=eps)

    target = torch.log(primal_clip) - torch.log(dual_clip)
    grad_u = u - target
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)

    with torch.no_grad():
        u_new = u - eta_u * grad_u
        u_new = torch.clamp(u_new, min=u_min, max=u_max)
        loss_val = 0.5 * (u - target).pow(2)

    return u_new.detach(), loss_val.detach(), target.detach(), grad_u.detach()


def online_convex_bal_lipschitz_update_u(
    u,
    primal_res,
    dual_base,
    L_hat,
    eta_u=0.05,
    G_clip=10.0,
    u_min=math.log(1e-6),
    u_max=math.log(1e4),
    eps=1e-12,
    lipschitz_floor_alpha=1.0,
):
    """OGD on u=log(sigma) with hard Lipschitz projection (port of the
    canonical _online.py implementation). The projection enforces
        sigma >= alpha * L_hat
    by taking a max in log-space after the OGD step. alpha=1 reproduces
    the original hard projection used in adaptive_sigma_lipschitz_proof.tex.

    Returns: (u_new, residual_loss, target, log_L, floor_active, grad_u).
    """
    primal_clip = torch.clamp(primal_res.detach(), min=eps)
    dual_clip = torch.clamp(dual_base.detach(), min=eps)
    L_clip = torch.clamp(L_hat.detach(), min=eps)

    target = torch.log(primal_clip) - torch.log(dual_clip)
    log_L = torch.log(L_clip)
    log_floor = log_L + math.log(max(lipschitz_floor_alpha, eps))

    diff = u - target
    res_loss = diff.pow(2)
    grad_u = 2.0 * diff
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)

    with torch.no_grad():
        u_raw = u - eta_u * grad_u
        floor_active = (u_raw < log_floor).to(log_floor.dtype)
        u_new = torch.maximum(u_raw, log_floor)
        u_new = torch.clamp(u_new, min=u_min, max=u_max)

    return (u_new.detach(), res_loss.detach(), target.detach(),
            log_L.detach(), floor_active.detach(), grad_u.detach())


def online_task_aware_update_u(
    u,
    primal_res,
    dual_base,
    train_loss_curr,
    train_loss_prev,
    task_lambda=1.0,
    eta_u=0.1,
    u_min=math.log(1e-6),
    u_max=math.log(1e4),
    eps=1e-12,
    G_clip=10.0,
):
    """
    Task-aware online update for u = log(sigma).

    L(u) = (u - target)^2 + task_lambda * relu(loss_curr - loss_prev) * |u - target|

    When training loss is decreasing, the task term vanishes and the update
    reduces to pure residual balance.
    """
    primal_clip = torch.clamp(primal_res.detach(), min=eps)
    dual_clip = torch.clamp(dual_base.detach(), min=eps)

    target = torch.log(primal_clip) - torch.log(dual_clip)
    diff = u - target

    # Residual balance gradient
    grad_residual = 2.0 * diff

    # Task-awareness gradient: activated only when loss increases
    loss_increase = max(0.0, train_loss_curr - train_loss_prev)
    grad_task = loss_increase * torch.sign(diff)

    # Combined
    grad_u = grad_residual + task_lambda * grad_task
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)

    with torch.no_grad():
        u_new = u - eta_u * grad_u
        u_new = torch.clamp(u_new, min=u_min, max=u_max)
        residual_loss = diff.pow(2)
        task_loss = loss_increase * torch.abs(diff)
        total_loss = residual_loss + task_lambda * task_loss

    return (
        u_new.detach(),
        total_loss.detach(),
        residual_loss.detach(),
        task_loss.detach(),
        target.detach(),
        grad_u.detach(),
    )


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

    seed = args.init_seed
    #logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    #logger.info("Partitioning data")
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --use_wandb=True was passed.")

        if args.wandb_run_name is None:
            args.wandb_run_name = f"{args.alg}-{args.dataset}-seed{args.init_seed}"

        wandb_run = wandb.init(
            project=args.wandb_project,
            # dir="/data/yutong/wandb",
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=vars(args),
            reinit=True,
        )
        # auto-aggregate best/last for the main table + sensitivity plot
        wandb.define_metric("test/acc", summary="max")
        wandb.define_metric("test/acc", summary="last")
        wandb.define_metric("train/acc", summary="last")
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

        # Initialize global model w^0
        W_n_0 = [param.detach().clone() for param in model.parameters()]

        # Local primal variables w_i
        W_b_initial = [[param.clone() for param in W_n_0] for _ in range(args.n_parties)]

        # Unscaled dual variables pi_i
        P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.n_parties)]

        # Per-client optimizer states for adam_warmstart
        optimizer_states = [None for _ in range(args.n_parties)]

        sigma_lr = args.sigma_lr
        l2_lambda = args.l2_lambda
        sigma_mode = getattr(args, "sigma_mode", "fixed")
        sigma_min = getattr(args, "sigma_min", 1e-6)
        sigma_max = getattr(args, "sigma_max", 1e4)
        eta_u = getattr(args, "eta_u", 0.1)
        G_clip = getattr(args, "G_clip", 10.0)
        sigma_update_freq = getattr(args, "sigma_update_freq", 1)
        sigma_ema_beta = getattr(args, "sigma_ema_beta", 0.9)
        eps_val = getattr(args, "eps", 1e-12)
        task_lambda = getattr(args, "task_lambda", 1.0)
        prev_train_loss = None

        # ---- BB Lipschitz floor state (consumed by online_convex_bal_lipschitz) ----
        eta_u_decay = getattr(args, "eta_u_decay", "none")
        lipschitz_estimator = getattr(args, "lipschitz_estimator", "ema")
        lipschitz_window_size = getattr(args, "lipschitz_window_size", 20)
        lipschitz_ema_beta = getattr(args, "lipschitz_ema_beta", 0.9)
        lipschitz_min_dz = getattr(args, "lipschitz_min_dz", 1e-6)
        lipschitz_max = getattr(args, "lipschitz_max", 1e8)
        lipschitz_floor_alpha = getattr(args, "lipschitz_floor_alpha", 1.0)
        grad_global_prev = None
        z_prev_bb = None
        L_hat_ema = torch.tensor(0.0, device=device)
        L_hat_buffer = []
        # k counter for textbook_sc 1/(2k) — increments each sigma-update fire
        # so the schedule is invariant to sigma_update_freq (matches _online.py).
        sigma_update_step = 0

        # u = log sigma for online update
        u_sigma = torch.tensor(
            math.log(max(sigma_lr, eps_val)),
            device=device
        )

        # EMA buffers
        primal_res_ema = None
        dual_base_ema = None

        total_data_points = sum(len(net_dataidx_map[r]) for r in range(args.n_parties))
        alpha_b = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]

        # Initial global model from ADMM formula
        W_global = generate_W_global(
            args.n_parties, W_b_initial, P_b_initial, sigma_lr, alpha_b, l2_lambda
        )

        with torch.no_grad():
            for param, w in zip(model.parameters(), W_global):
                param.copy_(w)

        test_record = []

        # Local CSV logging for offline plotting (independent of wandb).
        local_csv_path = None
        local_csv_file = None
        local_csv_writer = None
        local_csv_fields = [
            "round", "dataset", "partition", "method", "sigma_mode",
            "sigma_lr_init", "epochs", "lr", "seed",
            "test_acc", "train_local_admm_loss_avg",
            "primal_res_avg", "delta_w_global_avg",
            "sigma_value", "log_sigma_value",
            "sigma_loss", "sigma_target", "sigma_grad",
        ]
        if args.local_log_dir:
            import csv as _csv
            os.makedirs(args.local_log_dir, exist_ok=True)
            run_tag = args.wandb_run_name or f"{args.dataset}_{args.alg}_seed{args.init_seed}"
            safe_tag = run_tag.replace("/", "_")
            local_csv_path = os.path.join(args.local_log_dir, f"{safe_tag}.csv")
            local_csv_file = open(local_csv_path, "w", newline="")
            local_csv_writer = _csv.DictWriter(local_csv_file, fieldnames=local_csv_fields)
            local_csv_writer.writeheader()

            meta_path = os.path.join(args.local_log_dir, f"{safe_tag}.meta.json")
            with open(meta_path, "w") as _mf:
                json.dump(vars(args), _mf, indent=2, default=str)

        bb_criterion = nn.CrossEntropyLoss().to(device)

        for round_idx in range(args.comm_round):
            logger.info(f"ADMM round {round_idx}")
            W_global_prev = [w.detach().clone() for w in W_global]

            # Snapshot z^k for BB Lipschitz estimate (only used by lipschitz mode)
            z_curr_bb = None
            grad_global_curr = None
            if sigma_mode == "online_convex_bal_lipschitz":
                z_curr_bb = [w.detach().clone() for w in W_global]

            # -----------------------------------------
            # 1) Local primal step: update each w_i^{k+1}
            # -----------------------------------------
            new_W_b = []
            local_losses = []

            for sb in range(args.n_parties):
                # Warm-start model from this client's previous local solution
                # added warm start differs here
                with torch.no_grad():
                    for param, w_prev in zip(model.parameters(), W_b_initial[sb]):
                        param.copy_(w_prev)

                dataidxs = net_dataidx_map[sb]

                noise_level = args.noise
                if sb == args.n_parties - 1:
                    noise_level = 0

                if args.noise_type == 'space':
                    train_dl_local, _, _, _ = get_dataloader(
                        args.dataset, args.datadir, args.batch_size, 32,
                        dataidxs, noise_level, sb, args.n_parties - 1
                    )
                else:
                    noise_level = args.noise / (args.n_parties - 1) * sb
                    train_dl_local, _, _, _ = get_dataloader(
                        args.dataset, args.datadir, args.batch_size, 32,
                        dataidxs, noise_level
                    )

                # BB Lipschitz: per-client task-grad pass at z^k = W_global,
                # accumulated alpha-weighted into grad_global_curr. Done BEFORE
                # the local solve so model state at z^k is preserved for the
                # gradient. local_admm_train's `reset` init will overwrite the
                # model again, so this leaves no residue.
                if sigma_mode == "online_convex_bal_lipschitz":
                    grad_i = compute_task_grad_at_z(
                        model, train_dl_local, W_global, bb_criterion, device,
                    )
                    if grad_global_curr is None:
                        grad_global_curr = [
                            torch.zeros_like(g) if g is not None else None
                            for g in grad_i
                        ]
                    with torch.no_grad():
                        for j, g in enumerate(grad_i):
                            if g is not None and grad_global_curr[j] is not None:
                                grad_global_curr[j].add_(alpha_b[sb] * g)

                result = local_admm_train(
                    model=model,
                    train_dl_local=train_dl_local,
                    w_global=W_global,
                    pi_local=P_b_initial[sb],
                    sigma_lr=sigma_lr,
                    args=args,
                    device=device,
                    alpha_i=alpha_b[sb],
                    optimizer_state=optimizer_states[sb]
                )

                if args.optimizer in ('adam_warmstart', 'adamw_admm_explicit_warmstart'):
                    W_i_new, avg_local_loss, optimizer_states[sb] = result
                else:
                    W_i_new, avg_local_loss = result

                new_W_b.append(W_i_new)
                local_losses.append(avg_local_loss)

            W_b_initial = new_W_b

            # -----------------------------------------
            # 2) Global primal step: update w^{k+1}
            # -----------------------------------------
            W_global = generate_W_global(
                args.n_parties,
                W_b_initial,
                P_b_initial,
                sigma_lr,
                alpha_b,
                l2_lambda
            )

            with torch.no_grad():
                for param, w in zip(model.parameters(), W_global):
                    param.copy_(w)

            # -----------------------------------------
            # 3) Dual step: update pi_i^{k+1}
            # -----------------------------------------
            with torch.no_grad():
                for sb in range(args.n_parties):
                    for pi, wi, wg in zip(P_b_initial[sb], W_b_initial[sb], W_global):
                        pi.add_(sigma_lr * (wi - wg))

            # -----------------------------------------
            # 4) Residual diagnostics
            # -----------------------------------------
            # primal residual: sqrt(sum_i alpha_i ||w_i - w||^2)
            primal_sq = None
            for sb in range(args.n_parties):
                client_sq = None
                for wi, wg in zip(W_b_initial[sb], W_global):
                    diff = wi - wg
                    val = torch.sum(diff * diff)
                    client_sq = val if client_sq is None else client_sq + val
                weighted = alpha_b[sb] * client_sq
                primal_sq = weighted if primal_sq is None else primal_sq + weighted
            avg_primal_res = torch.sqrt(primal_sq)

            # dual-like signal: ||w^{k+1} - w^k||
            avg_dual_base = diff_global_norm(W_global, W_global_prev)

            # optional EMA smoothing
            cur_primal = float(avg_primal_res.item())
            cur_dual_base = float(avg_dual_base.item())

            if primal_res_ema is None:
                primal_res_ema = cur_primal
                dual_base_ema = cur_dual_base
            else:
                primal_res_ema = sigma_ema_beta * primal_res_ema + (1.0 - sigma_ema_beta) * cur_primal
                dual_base_ema = sigma_ema_beta * dual_base_ema + (1.0 - sigma_ema_beta) * cur_dual_base

            primal_smooth = torch.tensor(primal_res_ema, device=device)
            dual_smooth = torch.tensor(dual_base_ema, device=device)

            sigma_loss = None
            sigma_target = None
            sigma_grad = None
            sigma_res_loss = None
            sigma_task_loss = None
            ta_loss_increase = None
            lf_log_L = None
            lf_floor_active = None
            L_hat_tensor = None
            eta_u_eff = None

            # BB Lipschitz update: needs previous round's grad and z, so the
            # estimate first becomes available from round 1. Mirrors
            # _online.py lines 1762-1851.
            if (sigma_mode == "online_convex_bal_lipschitz"
                    and grad_global_curr is not None):
                with torch.no_grad():
                    if grad_global_prev is not None and z_prev_bb is not None:
                        dz_tensors = [a - b for a, b in zip(z_curr_bb, z_prev_bb)]
                        dz_norm = global_norm(dz_tensors)
                        if dz_norm.item() >= lipschitz_min_dz:
                            dg_tensors = []
                            for a, b in zip(grad_global_curr, grad_global_prev):
                                if a is None and b is None:
                                    continue
                                if a is None:
                                    dg_tensors.append(-b.detach())
                                elif b is None:
                                    dg_tensors.append(a.detach())
                                else:
                                    dg_tensors.append(a.detach() - b.detach())
                            dg_norm = global_norm(dg_tensors)
                            L_hat_raw = torch.clamp(dg_norm / dz_norm, max=lipschitz_max)
                            L_hat_ema = (lipschitz_ema_beta * L_hat_ema
                                         + (1.0 - lipschitz_ema_beta) * L_hat_raw)
                            L_hat_buffer.append(float(L_hat_raw.item()))
                            if len(L_hat_buffer) > lipschitz_window_size:
                                L_hat_buffer.pop(0)

                    if lipschitz_estimator == "ema":
                        L_hat_tensor = L_hat_ema.clone()
                    elif lipschitz_estimator == "running_min" and L_hat_buffer:
                        L_hat_tensor = torch.tensor(min(L_hat_buffer), device=device)
                    elif lipschitz_estimator == "running_median" and L_hat_buffer:
                        sb_buf = sorted(L_hat_buffer)
                        n_buf = len(sb_buf)
                        med = (sb_buf[n_buf // 2] if n_buf % 2 == 1
                               else 0.5 * (sb_buf[n_buf // 2 - 1] + sb_buf[n_buf // 2]))
                        L_hat_tensor = torch.tensor(med, device=device)
                    else:
                        L_hat_tensor = torch.tensor(0.0, device=device)

            # -----------------------------------------
            # 5) Adaptive sigma update for NEXT round
            # -----------------------------------------
            if ((round_idx + 1) % sigma_update_freq == 0):
                if sigma_mode == "heuristic":
                    # He et al. S3 compares ||r|| vs mu*||s|| where s = sigma * ||Δw||
                    # dual_smooth is the unscaled ||w^{k+1} - w^k||, so scale by sigma_lr
                    scaled_dual = sigma_lr * float(dual_smooth.item())
                    sigma_new = heuristic_update_sigma(
                        sigma_lr,
                        float(primal_smooth.item()),
                        scaled_dual,
                        mu=args.sigma_mu,
                        tau=args.sigma_tau,
                        k=round_idx,
                        k_max=args.sigma_kmax,
                    )
                    sigma_lr = float(max(sigma_min, min(sigma_max, sigma_new)))
                    u_sigma = torch.tensor(math.log(max(sigma_lr, eps_val)), device=device)

                elif sigma_mode == "online_convex_bal":
                    # Diminishing step size: eta_k = eta_u / sqrt(k+1)
                    # Required for O(sqrt(K)) regret in online convex optimization
                    # Satisfies: sum eta_k = inf (exploration), sum eta_k^2 < inf (convergence)
                    # eta_k = eta_u / math.sqrt(round_idx + 1.0)
                    eta_k = eta_u
                    u_new, sigma_loss, sigma_target, sigma_grad = online_convex_bal_update_u(
                        u_sigma,
                        primal_smooth,
                        dual_smooth,
                        eta_u=eta_k,
                        u_min=math.log(sigma_min),
                        u_max=math.log(sigma_max),
                        eps=eps_val,
                        G_clip=G_clip,
                    )
                    u_sigma = u_new
                    sigma_lr = float(torch.exp(u_new).item())

                elif sigma_mode == "online_convex_bal_lipschitz":
                    sigma_update_step += 1
                    L_hat_arg = (L_hat_tensor if L_hat_tensor is not None
                                 else torch.tensor(0.0, device=device))
                    if eta_u_decay == "inverse":
                        eta_u_eff = eta_u / sigma_update_step
                    elif eta_u_decay == "inv_sqrt":
                        eta_u_eff = eta_u / math.sqrt(sigma_update_step)
                    elif eta_u_decay == "textbook_sc":
                        eta_u_eff = 1.0 / (2.0 * sigma_update_step)
                    else:
                        eta_u_eff = eta_u
                    (u_new, sigma_loss, sigma_target, lf_log_L,
                     lf_floor_active, sigma_grad) = online_convex_bal_lipschitz_update_u(
                        u=u_sigma,
                        primal_res=primal_smooth,
                        dual_base=dual_smooth,
                        L_hat=L_hat_arg,
                        eta_u=eta_u_eff,
                        G_clip=G_clip,
                        u_min=math.log(sigma_min),
                        u_max=math.log(sigma_max),
                        eps=eps_val,
                        lipschitz_floor_alpha=lipschitz_floor_alpha,
                    )
                    u_sigma = u_new
                    sigma_lr = float(torch.exp(u_new).item())

                elif sigma_mode == "online_task_aware":
                    cur_train_loss = sum(local_losses) / max(len(local_losses), 1)
                    prev_loss = prev_train_loss if prev_train_loss is not None else cur_train_loss

                    eta_k = eta_u
                    (u_new, sigma_loss, sigma_res_loss, sigma_task_loss,
                     sigma_target, sigma_grad) = online_task_aware_update_u(
                        u_sigma,
                        primal_smooth,
                        dual_smooth,
                        train_loss_curr=cur_train_loss,
                        train_loss_prev=prev_loss,
                        task_lambda=task_lambda,
                        eta_u=eta_k,
                        u_min=math.log(sigma_min),
                        u_max=math.log(sigma_max),
                        eps=eps_val,
                        G_clip=G_clip,
                    )
                    u_sigma = u_new
                    sigma_lr = float(torch.exp(u_new).item())
                    ta_loss_increase = max(0.0, cur_train_loss - prev_loss)
                    prev_train_loss = cur_train_loss

                elif sigma_mode == "fixed":
                    sigma_lr /= args.mu_lr

                else:
                    raise ValueError(f"Unsupported sigma_mode: {sigma_mode}")

            # -----------------------------------------
            # 6) Evaluate and log
            # -----------------------------------------
            test_acc = compute_accuracy(model, test_dl_global, get_confusion_matrix=False, device=device)
            test_record.append(test_acc)

            logger.info('>> avg local ADMM loss: %f' % (sum(local_losses) / max(len(local_losses), 1)))
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> primal_res/avg: %f' % avg_primal_res.item())
            logger.info('>> delta_w_global: %f' % avg_dual_base.item())
            logger.info('>> sigma_lr: %f' % sigma_lr)

            if sigma_loss is not None:
                logger.info('>> sigma_loss: %f' % sigma_loss.item())
            if sigma_target is not None:
                logger.info('>> sigma_target: %f' % sigma_target.item())
            if sigma_grad is not None:
                logger.info('>> sigma_grad: %f' % sigma_grad.item())

            if args.use_wandb:
                log_dict = {
                    "round": round_idx,
                    "test/acc": test_acc,
                    "train/local_admm_loss_avg": sum(local_losses) / max(len(local_losses), 1),
                    "primal_res/avg": avg_primal_res.item(),
                    "delta_w_global/avg": avg_dual_base.item(),
                    "sigma/value": sigma_lr,
                    "log_sigma/value": math.log(max(sigma_lr, eps_val)),
                }

                if sigma_loss is not None:
                    log_dict["sigma/loss"] = sigma_loss.item()
                if sigma_target is not None:
                    log_dict["sigma/target"] = sigma_target.item()
                if sigma_grad is not None:
                    log_dict["sigma/grad"] = sigma_grad.item()
                if sigma_res_loss is not None:
                    log_dict["sigma/residual_loss"] = sigma_res_loss.item()
                    log_dict["sigma/task_loss"] = sigma_task_loss.item()
                    log_dict["sigma/loss_increase"] = ta_loss_increase

                if sigma_mode == "online_convex_bal_lipschitz":
                    if lf_log_L is not None:
                        log_dict["sigma/log_L_hat"] = lf_log_L.item()
                        log_dict["sigma/L_hat"] = float(torch.exp(lf_log_L).item())
                    if lf_floor_active is not None:
                        log_dict["sigma/floor_active"] = lf_floor_active.item()
                    if L_hat_tensor is not None:
                        log_dict["sigma/L_hat_ema"] = float(L_hat_tensor.item())
                    log_dict["sigma/L_hat_buffer_size"] = len(L_hat_buffer)
                    if eta_u_eff is not None:
                        log_dict["sigma/eta_u_eff"] = float(eta_u_eff)
                    log_dict["sigma/update_step"] = sigma_update_step

                for sb in range(args.n_parties):
                    client_sq = None
                    for wi, wg in zip(W_b_initial[sb], W_global):
                        diff = wi - wg
                        val = torch.sum(diff * diff)
                        client_sq = val if client_sq is None else client_sq + val
                    log_dict[f"primal_res/client_{sb}"] = torch.sqrt(client_sq).item()

                wandb.log(log_dict, step=round_idx)

            if local_csv_writer is not None:
                csv_row = {
                    "round": round_idx,
                    "dataset": args.dataset,
                    "partition": args.partition,
                    "method": sigma_mode,
                    "sigma_mode": sigma_mode,
                    "sigma_lr_init": args.sigma_lr,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "seed": args.init_seed,
                    "test_acc": test_acc,
                    "train_local_admm_loss_avg": sum(local_losses) / max(len(local_losses), 1),
                    "primal_res_avg": avg_primal_res.item(),
                    "delta_w_global_avg": avg_dual_base.item(),
                    "sigma_value": sigma_lr,
                    "log_sigma_value": math.log(max(sigma_lr, eps_val)),
                    "sigma_loss": sigma_loss.item() if sigma_loss is not None else "",
                    "sigma_target": sigma_target.item() if sigma_target is not None else "",
                    "sigma_grad": sigma_grad.item() if sigma_grad is not None else "",
                }
                local_csv_writer.writerow(csv_row)
                local_csv_file.flush()

            # Rotate BB Lipschitz state for next round.
            if sigma_mode == "online_convex_bal_lipschitz":
                grad_global_prev = grad_global_curr
                z_prev_bb = z_curr_bb

        if local_csv_file is not None:
            local_csv_file.close()
            logger.info(f"Saved local metrics to {local_csv_path}")

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

            if args.use_wandb:
                wandb.log({
                    "round": round,
                    "test/acc": test_acc,
                    "train/acc": train_acc,
                }, step=round)


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

            if args.use_wandb:
                wandb.log({
                    "round": round,
                    "test/acc": test_acc,
                    "train/acc": train_acc,
                }, step=round)

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

            if args.use_wandb:
                wandb.log({
                    "round": round,
                    "test/acc": test_acc,
                    "train/acc": train_acc,
                }, step=round)

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

            if args.use_wandb:
                wandb.log({
                    "round": round,
                    "test/acc": test_acc,
                    "train/acc": train_acc,
                }, step=round)

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

            if args.use_wandb:
                wandb.log({
                    "round": round,
                    "test/acc": test_acc,
                    "train/acc": train_acc,
                }, step=round)

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

    if args.use_wandb and wandb_run is not None:
        wandb.finish()

