"""Train CIFAR10 with online adaptive σ + Lipschitz floor — l2_lower_loss variant
(matches the paper's Table 3 VGG-11 entry, which uses explicit L2 on the global
w plus per-sub-batch gradient-norm normalization on top of the lower-loss procedure).

Modified from l2_lower_loss_mPiAM_varying_rho_sigma.py. Replaces the hardcoded
σ schedule
    sigma_lr_current = ((1/(σ+ρ))/lr_curr) * σ
with an OGD update on u=log(σ) projected onto σ ≥ α·exp(L̂), reusing the
standalone module at ../../Data Heterogenerity/lipschitz_ogd.py.

Preserves all VGG-specific paper details intact:
  - --l2_lambda flag (explicit L2 on global w via P_n_avg/(tau_lr + l2_lambda))
  - normalized_factor[sb] per-sub-batch rescaling in the local solver
  - generate_W_global_normalized aggregation

Switch with --sigma_mode:
  fixed                          (default): original lr-coupled σ schedule
  online_convex_bal              : OGD on u=log(σ), bounded only by [σ_min, σ_max]
  online_convex_bal_lipschitz    : OGD plus σ ≥ α·exp(L̂) hard floor

σ-update cadence: per mini-batch when --sigma_update_freq=1 (default), or once
every K mini-batches if larger.

Adds wandb logging gated by --use_wandb.
"""

from __future__ import print_function

import math
import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *
from adabound import AdaBound
from torch.optim import Adam, SGD
from optimizers import *

# Reach across to Data Heterogenerity/ for the standalone OGD module so we
# don't carry a stale copy here.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_DH_DIR = os.path.join(_REPO_ROOT, 'Data Heterogenerity')
if _DH_DIR not in sys.path:
    sys.path.insert(0, _DH_DIR)
from lipschitz_ogd import LipschitzFloorOGD, global_norm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (l2 lipschitz)')
    parser.add_argument('--total_epoch', default=200, type=int)
    parser.add_argument('--decay_epoch', default=150, type=int)
    parser.add_argument('--model', default='resnet', type=str,
                        choices=['resnet', 'densenet', 'vgg'])
    parser.add_argument('--optim', default='sgd', type=str,
                        choices=['sgd', 'adam', 'adamw', 'adabelief', 'yogi',
                                 'msvag', 'radam', 'fromage', 'adabound'])
    parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--final_lr', default=0.1, type=float)
    parser.add_argument('--gamma', default=1e-3, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_gpu', default=4, type=int)
    parser.add_argument('--sigma_lr', default=0.08, type=float)
    parser.add_argument('--rho_lr', default=10000, type=float)
    parser.add_argument('--beta_rmsprop', default=0.9, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--baseline_acc', default=0.9, type=float)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--l2_lambda', default=0.0, type=float,
                        help='explicit L2 on global w at aggregation: '
                             'W_n_avg + P_n_avg / (tau_lr + l2_lambda)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--datadir', default='./data', type=str,
                        help='CIFAR-10 root directory')

    # Adaptive σ (Lipschitz-floored OGD)
    parser.add_argument('--sigma_mode', type=str, default='fixed',
                        choices=['fixed', 'heuristic', 'online_convex_bal',
                                 'online_convex_bal_lipschitz'],
                        help='fixed = original lr-coupled schedule; '
                             'heuristic = Boyd residual-balance multiplicative rule; '
                             'online_convex_bal = OGD on u=log(σ), no floor; '
                             'online_convex_bal_lipschitz = OGD + BB Lipschitz floor')
    parser.add_argument('--heuristic_mu', type=float, default=10.0)
    parser.add_argument('--heuristic_tau', type=float, default=2.0)
    parser.add_argument('--sigma_min', type=float, default=1e-6)
    parser.add_argument('--sigma_max', type=float, default=1e6)
    parser.add_argument('--eta_u', type=float, default=0.05,
                        help='OGD step size on u=log(σ); ignored under textbook_sc decay')
    parser.add_argument('--eta_u_decay', type=str, default='textbook_sc',
                        choices=['none', 'inverse', 'inv_sqrt', 'textbook_sc'])
    parser.add_argument('--G_clip', type=float, default=10.0)
    parser.add_argument('--sigma_update_freq', type=int, default=1,
                        help='fire one σ-OGD step every this many mini-batches')

    # BB Lipschitz estimator
    parser.add_argument('--lipschitz_estimator', type=str, default='ema',
                        choices=['ema', 'running_min', 'running_median',
                                 'ema_per_layer_median'])
    parser.add_argument('--lipschitz_window_size', type=int, default=20)
    parser.add_argument('--lipschitz_ema_beta', type=float, default=0.9)
    parser.add_argument('--lipschitz_min_dz', type=float, default=1e-6)
    parser.add_argument('--lipschitz_max', type=float, default=1e8)
    parser.add_argument('--lipschitz_floor_alpha', type=float, default=1.0)

    # wandb
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default='paper-lipschitz-vision-cifar10-vgg')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_job_type', type=str, default='train')
    parser.add_argument('--wandb_log_per_step', type=str2bool, default=False)

    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
                                               shuffle=False, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
                                              shuffle=False, num_workers=2)
    return train_loader, test_loader


def get_ckpt_name(model, optimizer, lr, momentum, beta1, beta2, eps, weight_decay,
                  reset, run, sigma_mode):
    return '{}-{}-lr{}-betas{}-{}-eps{}-wd{}-mom{}-reset{}-run{}-sigmode{}'.format(
        model, optimizer, lr, beta1, beta2, eps, weight_decay, momentum,
        str(reset), run, sigma_mode)


def average_parameters_with_normalized(num_train_env, list_vars, list_alpha, list_normalized):
    sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
    for i in range(num_train_env):
        W_n = list_vars[i]
        alpha = list_alpha[i]
        normalized_factor = list_normalized[i]
        sum_vars = [sum_ + alpha * update / normalized_factor
                    for sum_, update in zip(sum_vars, W_n)]
    return sum_vars


def average_parameters(num_train_env, list_vars, list_alpha):
    sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
    for i in range(num_train_env):
        W_n = list_vars[i]
        alpha = list_alpha[i]
        sum_vars = [sum_ + alpha * update for sum_, update in zip(sum_vars, W_n)]
    return sum_vars


def generate_W_global(num_batches, W_n_list, P_n_list, tau_lr, alpha, l2_lambda):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / (tau_lr + l2_lambda)
        W_n_avg[i].detach()
    return W_n_avg


def generate_W_global_normalized(num_batches, W_n_list, P_n_list, tau_lr, alpha,
                                  list_normalized, l2_lambda):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters_with_normalized(num_batches, P_n_list, alpha,
                                                  list_normalized)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / (tau_lr + l2_lambda)
        W_n_avg[i].detach()
    return W_n_avg


def build_model(args, device):
    print('==> Building model..')
    net = {'resnet': ResNet34, 'densenet': DenseNet121, 'vgg': vgg11}[args.model]()
    net = net.to(device)
    if 'cuda' in str(device):
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    if args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay, eps=args.eps)
    if args.optim == 'fromage':
        return Fromage(model_params, args.lr)
    if args.optim == 'radam':
        return RAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay, eps=args.eps)
    if args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay, eps=args.eps)
    if args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                         weight_decay=args.weight_decay, eps=args.eps)
    if args.optim == 'yogi':
        return Yogi(model_params, args.lr, betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay)
    if args.optim == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay)
    if args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    print('Optimizer not found')


def zero_grad(params):
    for param in params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    print('###############################################')
    print(' test acc %.3f' % accuracy)
    print('###############################################')
    return accuracy


def adjust_learning_rate(learning_rate, epoch, step_size=25, gamma=0.5):
    """Original VGG schedule preserved verbatim — only used in fixed sigma_mode."""
    if epoch % 5 == 0 and 21 < epoch < 61:
        learning_rate *= gamma
    if epoch % step_size == 0 and 0 < epoch < 21:
        learning_rate *= gamma
    if epoch % step_size == 0 and 61 < epoch < 150:
        learning_rate *= gamma
    if epoch % 25 == 0 and epoch > 100:
        learning_rate *= 0.35
    return learning_rate


def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_inner_solver_diagnostics(
    P_b, accumulators, num_sb, beta_rmsprop, updated_iter, eps,
    sigma_curr, rho_curr,
):
    """Diagnostics for inner-solver instability:
      - ||π||: dual-variable L2 norm per sub-batch (mean and max). π runaway
        is one suspected failure mode (gradient swamped by accumulated dual).
      - mean(ρ·√v): preconditioner term in the local solver denominator.
      - σ / mean(ρ·√v): leverage ratio (<<1 ρ-dominance, ~1 balanced, >>1 σ-dominated).
    Note: VGG entry's normalized_factor[sb] cancels in the ratio, so the
    diagnostic is identical to the resnet entry's (deliberately).
    """
    with torch.no_grad():
        pi_norms = []
        for sb in range(num_sb):
            sq_sum = sum(p.detach().pow(2).sum() for p in P_b[sb])
            pi_norms.append(float(torch.sqrt(sq_sum + 1e-12).item()))
        pi_norm_mean = sum(pi_norms) / len(pi_norms) if pi_norms else 0.0
        pi_norm_max = max(pi_norms) if pi_norms else 0.0

        bc2 = max(1 - beta_rmsprop ** updated_iter, 1e-12)
        rho_sv_sum, n_terms = 0.0, 0
        for sb in range(num_sb):
            for acc in accumulators[sb]:
                corrected = acc.detach() / bc2
                term = float(
                    (rho_curr * (torch.sqrt(corrected) + eps)).mean().item()
                )
                rho_sv_sum += term
                n_terms += 1
        rho_sv_mean = rho_sv_sum / max(n_terms, 1)
        ratio = sigma_curr / max(rho_sv_mean, 1e-12)

    return {
        'diagnostic/pi_norm_mean': pi_norm_mean,
        'diagnostic/pi_norm_max': pi_norm_max,
        'diagnostic/rho_sqrt_v_mean': rho_sv_mean,
        'diagnostic/sigma_over_rho_sqrt_v': ratio,
    }


def main():
    parser = get_parser()
    args = parser.parse_args()

    _seed_everything(args.seed)

    train_loader, test_loader = build_dataset(args)
    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.use_wandb:
        if not _WANDB_AVAILABLE:
            print('WARNING: --use_wandb=true but `wandb` is not installed; disabling.')
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                config=vars(args),
            )

    ckpt_name = get_ckpt_name(
        model=args.model, optimizer=args.optim, lr=args.lr,
        momentum=args.momentum, beta1=args.beta1, beta2=args.beta2,
        eps=args.eps, weight_decay=args.weight_decay,
        reset=args.reset, run=args.run, sigma_mode=args.sigma_mode,
    )
    print('ckpt_name', ckpt_name)

    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss()

    W_n_0 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]
    W_b_initial = [[param.clone() for param in W_n_0] for _ in range(args.num_gpu)]
    P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
    accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]

    sigma_lr = args.sigma_lr
    alpha_b = [1.0 / args.num_gpu for _ in range(args.num_gpu)]
    print('alpha_b is: ', alpha_b)
    W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial,
                                  sigma_lr, alpha_b, args.l2_lambda)
    zero_grad(model.parameters())

    learning_rate_current = 1.0 / (args.sigma_lr + args.rho_lr)
    sigma_lr_current = args.sigma_lr
    rho_lr_current = args.rho_lr
    updated_iteration = 1.0
    best_acc = args.baseline_acc
    train_accuracies, test_accuracies = [], []
    normalized_factor = []  # populated at epoch=0, iter_idx=0; one entry per sub-batch

    # ---- Adaptive-σ state (consumed in any non-fixed mode) ----
    floor = None
    last_floor_metrics = {}
    if args.sigma_mode in (
        'online_convex_bal_lipschitz', 'online_convex_bal', 'heuristic'
    ):
        ogd_mode = {
            'online_convex_bal_lipschitz': 'lipschitz',
            'online_convex_bal': 'no_floor',
            'heuristic': 'heuristic',
        }[args.sigma_mode]
        param_names = (
            [n for n, _ in model.named_parameters()]
            if ogd_mode == 'lipschitz' else None
        )
        floor = LipschitzFloorOGD(
            sigma_init=args.sigma_lr,
            device=device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            eta_u=args.eta_u,
            eta_u_decay=args.eta_u_decay,
            G_clip=args.G_clip,
            estimator=args.lipschitz_estimator,
            window_size=args.lipschitz_window_size,
            ema_beta=args.lipschitz_ema_beta,
            min_dz=args.lipschitz_min_dz,
            max_L=args.lipschitz_max,
            lipschitz_floor_alpha=args.lipschitz_floor_alpha,
            param_names=param_names,
            eps=args.eps,
            mode=ogd_mode,
            heuristic_mu=args.heuristic_mu,
            heuristic_tau=args.heuristic_tau,
        )
        sigma_lr_current = args.sigma_lr
        rho_lr_current = args.rho_lr

    for epoch in range(args.total_epoch):
        start = time.time()
        total_train_loss = 0.0
        epoch_correct, epoch_total = 0, 0

        if args.sigma_mode == 'fixed':
            learning_rate_current = adjust_learning_rate(
                learning_rate_current, epoch,
                step_size=args.decay_epoch, gamma=args.lr_gamma,
            )
            sigma_lr_current = ((1.0 / (args.sigma_lr + args.rho_lr)) / learning_rate_current) * args.sigma_lr
            rho_lr_current = 1.0 / learning_rate_current - sigma_lr_current
            print('Epoch %d  LR %.6g  sigma %.6g  rho %.6g'
                  % (epoch, learning_rate_current, sigma_lr_current, rho_lr_current))
        else:
            print('Epoch %d  sigma %.6g  rho %.6g  step %d'
                  % (epoch, sigma_lr_current, rho_lr_current,
                     floor.update_step if floor else 0))

        for iter_idx, (images, target) in enumerate(train_loader):
            sub_batch_size = images.size(0) // args.num_gpu
            alpha_b = []

            # Snapshot z^k = W_global before any modification this round
            # (used by BB estimator and dual residual).
            z_curr_bb = [w.clone().detach() for w in W_global]

            # Snapshot W_n^k for delta_y = ||W_n^{k+1} - W_n^k||.
            W_n_prev_list = [
                [w.clone().detach() for w in W_b_initial[sb]]
                for sb in range(args.num_gpu)
            ]

            grad_global_curr = None  # alpha-weighted accumulated grad at z^k

            for sb in range(args.num_gpu):
                with torch.no_grad():
                    for param, w in zip(model.parameters(), W_global):
                        param.copy_(w)

                if sb == args.num_gpu - 1:
                    images_sub = images[sub_batch_size * sb:].to(device)
                    target_sub = target[sub_batch_size * sb:].to(device)
                else:
                    images_sub = images[sub_batch_size * sb:sub_batch_size * (sb + 1)].to(device)
                    target_sub = target[sub_batch_size * sb:sub_batch_size * (sb + 1)].to(device)

                W_n = W_b_initial[sb]
                P_n = P_b_initial[sb]
                accumulators = accumulators_initial[sb]
                alpha_b.append(images_sub.size(0) / images.size(0))

                output = model(images_sub)
                loss = criterion(output, target_sub)
                total_train_loss += loss.item()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    epoch_correct += (pred == target_sub).sum().item()
                    epoch_total += target_sub.size(0)

                zero_grad(model.parameters())
                loss.backward()
                gradients = [
                    param.grad + args.weight_decay * param for param in model.parameters()
                ]

                # Compute per-sub-batch gradient-norm normalizer at first iter.
                if epoch == 0 and iter_idx == 0:
                    factor = torch.sqrt(
                        sum(torch.sum(g ** 2) for g in gradients if g is not None)
                    )
                    print('the l2-norm of gradient is:', factor)
                    normalized_factor.append(factor)

                # Accumulate alpha-weighted grad at z^k for the BB estimator.
                if floor is not None and floor.use_lipschitz_floor:
                    if grad_global_curr is None:
                        grad_global_curr = [
                            torch.zeros_like(g) if g is not None else None for g in gradients
                        ]
                    with torch.no_grad():
                        for j, g in enumerate(gradients):
                            if g is not None and grad_global_curr[j] is not None:
                                grad_global_curr[j].add_(alpha_b[-1] * g.detach())

                with torch.no_grad():
                    for param_wn, param_pn, gradient, param_wg, accumulator in zip(
                        W_n, P_n, gradients, W_global, accumulators
                    ):
                        accumulator.mul_(args.beta_rmsprop).add_(
                            (1 - args.beta_rmsprop) * (gradient + param_pn).pow(2)
                        )
                        bias_correction2 = 1 - args.beta_rmsprop ** updated_iteration
                        corrected_accumulator = accumulator / bias_correction2

                        # paper-faithful local solver: σ and ρ both rescaled by
                        # per-sub-batch normalized_factor.
                        delta = param_wg - (gradient + param_pn) / (
                            sigma_lr_current * normalized_factor[sb]
                            + rho_lr_current * normalized_factor[sb]
                            * (torch.sqrt(corrected_accumulator) + args.eps)
                        )
                        param_wn.copy_(delta.detach())
                        param_pn.add_(sigma_lr_current * (param_wn - param_wg))

                del loss, output

            updated_iteration += 1

            # Save W_global^k before re-aggregation (for residual computation).
            W_global_prev = [w.clone().detach() for w in W_global]
            with torch.no_grad():
                W_global = generate_W_global_normalized(
                    args.num_gpu, W_b_initial, P_b_initial,
                    sigma_lr_current, alpha_b, normalized_factor, args.l2_lambda
                )
                for param, w in zip(model.parameters(), W_global):
                    param.copy_(w)

            # ---- σ update event ----
            if floor is not None and (iter_idx + 1) % args.sigma_update_freq == 0:
                with torch.no_grad():
                    primal_res_per_sb = [
                        global_norm([a - b for a, b in zip(W_b_initial[sb], z_curr_bb)])
                        for sb in range(args.num_gpu)
                    ]
                    primal_res = sum(
                        alpha_b[sb] * primal_res_per_sb[sb] for sb in range(args.num_gpu)
                    )
                    delta_y_per_sb = [
                        global_norm([a - b for a, b in zip(W_b_initial[sb], W_n_prev_list[sb])])
                        for sb in range(args.num_gpu)
                    ]
                    delta_y = sum(
                        alpha_b[sb] * delta_y_per_sb[sb] for sb in range(args.num_gpu)
                    )
                    dual_res = sigma_lr_current * global_norm(
                        [a - b for a, b in zip(W_global, W_global_prev)]
                    )

                sigma_new, sigma_metrics = floor.step(
                    z_curr=z_curr_bb,
                    grad_curr=grad_global_curr,
                    primal_res=primal_res,
                    delta_y=delta_y,
                    dual_res=dual_res,
                )
                sigma_lr_current = sigma_new
                rho_lr_current = args.rho_lr  # ρ stays fixed in adaptive mode
                last_floor_metrics = sigma_metrics

                if args.use_wandb and args.wandb_log_per_step:
                    wandb.log(
                        {**sigma_metrics, 'iter': iter_idx, 'epoch': epoch},
                        step=int(updated_iteration),
                    )

        train_loss_avg = total_train_loss / max(len(train_loader) * args.num_gpu, 1)
        train_acc = 100. * epoch_correct / max(epoch_total, 1)
        test_acc = test(model, device, test_loader, criterion)
        end = time.time()
        print('Time: {:.2f}s  train_loss {:.4f}  train_acc {:.2f}  test_acc {:.2f}'.format(
            end - start, train_loss_avg, train_acc, test_acc))
        model.train()

        if test_acc > best_acc:
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if args.use_wandb:
            log_dict = {
                'train/loss': train_loss_avg,
                'train/acc': train_acc,
                'test/acc': test_acc,
                'sigma_lr_current': sigma_lr_current,
                'rho_lr_current': rho_lr_current,
                'epoch': epoch,
                'time/epoch_sec': end - start,
            }
            if args.sigma_mode == 'fixed':
                log_dict['lr_current'] = learning_rate_current
            if floor is not None:
                log_dict.update(last_floor_metrics)
            log_dict.update(_compute_inner_solver_diagnostics(
                P_b_initial, accumulators_initial, args.num_gpu,
                args.beta_rmsprop, updated_iteration, args.eps,
                sigma_lr_current, rho_lr_current,
            ))
            wandb.log(log_dict, step=epoch)

        if not os.path.isdir('curve_lower_loss'):
            os.makedirs('curve_lower_loss', exist_ok=True)
        torch.save(
            {'train_acc': train_accuracies, 'test_acc': test_accuracies},
            os.path.join('curve_lower_loss', ckpt_name),
        )

    print('The best accuracy is:', max(test_accuracies))
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
