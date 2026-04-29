"""Standalone Lipschitz-floored OGD module for adaptive σ in SISA-family
training.

Extracted from experiment_sisa_practise_online.py so the same machinery can be
reused in both the FL Data-Heterogeneity codebase (`Data Heterogenerity/`) and
the centralized vision codebase (`Vison Model/cifar10/`). The two consumer
scripts compute residuals slightly differently, so this module keeps the API
narrow: callers supply (z_curr, grad_curr, primal_res, delta_y) and receive
(sigma_new, metrics).

What's exported:
- `online_convex_bal_lipschitz_update_u`: pure-functional OGD step on
  u=log(σ) with hard floor σ ≥ α·exp(L̂).
- `online_convex_bal_update_u`: same OGD step WITHOUT the Lipschitz floor;
  σ is bounded only by [u_min, u_max]. The function takes a `delta_y`
  argument used as the "dual proxy" in the residual-balance target
  τ = log(primal_res / delta_y). The same function serves two modes:
    * `online_convex_bal` (legacy): caller passes Δy = ‖w_i^{k+1}-w_i^k‖
      (local change). σ-coupling is lost when ρ·√v dominates the local
      step, so OGD has no fixed point in that regime — see VGG failure
      analysis in benchmark_ogd.tex §"VGG: σ-rule failure modes".
    * `online_true_bal`: caller passes the canonical Boyd dual residual
      s = σ·‖w_g^{k+1}-w_g^k‖. The σ-factor in s gives a self-balancing
      fixed point — when σ rises, s rises, equilibrium r ≈ s achievable.
      Recommended for ill-conditioned problems where the inner solver's
      preconditioner term ρ·√v swamps σ.
- `heuristic_update_sigma`: classic Boyd residual-balance multiplicative rule
  (σ *= τ when primal ≫ dual, σ /= τ when dual ≫ primal). Standard ADMM
  baseline for adaptive σ; predates OGD-style updates.
- `compute_eta_u_eff`: diminishing step-size schedule
  (none | inverse | inv_sqrt | textbook_sc).
- `BBLipschitzEstimator`: Barzilai-Borwein L̂ with
  EMA / running_min / running_median / ema_per_layer_median smoothing.
- `LipschitzFloorOGD`: thin orchestrator wiring the above together; tracks
  u_sigma, sigma_update_step, BB previous state, and emits a metrics dict
  ready for wandb.log. Pass `use_lipschitz_floor=False` to disable the BB
  step + floor projection (no-floor variant).

See notes/lipschitz_floor_theorems.tex for the proofs and
project_lipschitz_floor_hard_projection.md for the design rationale.
"""

import math
import torch


def global_norm(tensors, eps=1e-12):
    """L2 norm over a list of tensors, skipping None entries."""
    s = None
    for t in tensors:
        if t is None:
            continue
        v = t.detach()
        val = (v * v).sum()
        s = val if s is None else (s + val)
    if s is None:
        return torch.tensor(0.0)
    return torch.sqrt(s + eps)


def online_convex_bal_lipschitz_update_u(
    u,
    primal_res,
    delta_y,
    L_hat,
    eta_u=0.05,
    G_clip=10.0,
    u_min=-20.0,
    u_max=20.0,
    eps=1e-12,
    lipschitz_floor_alpha=1.0,
):
    """OGD step on residual-balance loss with hard Lipschitz floor.

    Loss:           L(u) = (u - τ)^2,  τ = log(primal_res / delta_y)
    Step:           u_raw = u - η · clip(2(u-τ), ±G_clip)
    Hard floor:     u_new = max(u_raw, log(α) + log(L̂))
    Global bounds:  u_new ∈ [u_min, u_max]

    Returns (u_new, res_loss, target, log_L, floor_active, grad_u) — all
    detached scalar tensors.
    """
    r_clip = torch.clamp(primal_res, min=eps)
    dy_clip = torch.clamp(delta_y, min=eps)
    L_clip = torch.clamp(L_hat, min=eps)

    target = torch.log(r_clip) - torch.log(dy_clip)
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

    return (
        u_new.detach(),
        res_loss.detach(),
        target.detach(),
        log_L.detach(),
        floor_active.detach(),
        grad_u.detach(),
    )


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
    """OGD step on residual-balance loss WITHOUT a Lipschitz floor.

    Loss:           L(u) = (u - τ)^2,  τ = log(primal_res / delta_y)
    Step:           u_new = u - η · clip(2(u-τ), ±G_clip)
    Global bounds:  u_new ∈ [u_min, u_max]

    σ-floor here is whatever the caller passes via u_min = log(sigma_min).
    Matches the original `online_convex_bal` mode at
    experiment_sisa_practise_online.py:81.

    Returns (u_new, res_loss, target, grad_u) — all detached scalar tensors.
    """
    r_clip = torch.clamp(primal_res, min=eps)
    dy_clip = torch.clamp(delta_y, min=eps)

    target = torch.log(r_clip) - torch.log(dy_clip)
    diff = u - target
    res_loss = diff.pow(2)
    grad_u = 2.0 * diff
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)

    with torch.no_grad():
        u_new = u - eta_u * grad_u
        u_new = torch.clamp(u_new, min=u_min, max=u_max)

    return (
        u_new.detach(),
        res_loss.detach(),
        target.detach(),
        grad_u.detach(),
    )


def heuristic_update_sigma(sigma_old, primal_res, dual_res, mu=10.0, tau=2.0):
    """Boyd residual-balance heuristic for ADMM σ.

        σ_new = τ · σ_old   if  primal_res > μ · dual_res
        σ_new = σ_old / τ   if  dual_res   > μ · primal_res
        σ_new = σ_old       otherwise

    Mirrors heuristic_update_sigma at experiment_sisa_practise_online.py:62.
    Caller is responsible for clamping to [σ_min, σ_max] after this returns.
    """
    sigma_new = sigma_old
    p = float(primal_res)
    d = float(dual_res)
    if p > mu * d:
        sigma_new = sigma_old * tau
    elif d > mu * p:
        sigma_new = sigma_old / tau
    return sigma_new


def compute_eta_u_eff(eta_u, k_sigma, eta_u_decay='none'):
    """Diminishing step-size schedule for OGD on u=log(σ).

    k_sigma is the σ-update event counter (starts at 1 on first call).
    `textbook_sc` = 1/(μ·k) with μ=2 (the strong-convexity constant of the
    residual-balance loss (u-τ)^2). Parameter-free; ignores eta_u.
    """
    if k_sigma <= 0 or eta_u_decay == 'none':
        return eta_u
    if eta_u_decay == 'inverse':
        return eta_u / k_sigma
    if eta_u_decay == 'inv_sqrt':
        return eta_u / math.sqrt(k_sigma)
    if eta_u_decay == 'textbook_sc':
        return 1.0 / (2.0 * k_sigma)
    return eta_u


class BBLipschitzEstimator:
    """Barzilai-Borwein Lipschitz estimate with smoothing.

        L̂_k = ||g_k - g_{k-1}|| / ||z_k - z_{k-1}||

    Caller passes (z_curr, grad_curr) at each update event. The first call
    seeds previous state and returns L̂ = 0 (no estimate yet). Subsequent
    calls compute the BB ratio, update internal smoothers, and return the
    selected estimator.
    """

    def __init__(
        self,
        device,
        estimator='ema',
        window_size=20,
        ema_beta=0.9,
        min_dz=1e-6,
        max_L=1e8,
        num_param_groups=None,
    ):
        self.device = device
        self.estimator = estimator
        self.window_size = window_size
        self.ema_beta = ema_beta
        self.min_dz = min_dz
        self.max_L = max_L

        self.L_hat_ema = torch.tensor(0.0, device=device)
        self.L_hat_buffer = []
        self.L_hat_ema_per_layer = (
            [0.0] * num_param_groups if num_param_groups else None
        )
        self.z_prev = None
        self.grad_prev = None
        self.last_raw = None
        self.last_dz_norm = None
        self.last_dg_norm = None

    def update(self, z_curr, grad_curr):
        if self.z_prev is None or self.grad_prev is None:
            self._snapshot(z_curr, grad_curr)
            return self._dispatch()

        with torch.no_grad():
            dz = [a - b for a, b in zip(z_curr, self.z_prev)]
            dz_norm = global_norm(dz)
            if dz_norm.item() >= self.min_dz:
                dg = []
                for a, b in zip(grad_curr, self.grad_prev):
                    if a is None and b is None:
                        continue
                    if a is None:
                        dg.append(-b.detach())
                    elif b is None:
                        dg.append(a.detach())
                    else:
                        dg.append(a.detach() - b.detach())
                dg_norm = global_norm(dg)
                L_hat_raw = torch.clamp(dg_norm / dz_norm, max=self.max_L)
                self.last_raw = float(L_hat_raw.item())
                self.last_dz_norm = float(dz_norm.item())
                self.last_dg_norm = float(dg_norm.item())

                self.L_hat_ema = (
                    self.ema_beta * self.L_hat_ema
                    + (1.0 - self.ema_beta) * L_hat_raw
                )
                self.L_hat_buffer.append(self.last_raw)
                if len(self.L_hat_buffer) > self.window_size:
                    self.L_hat_buffer.pop(0)

                if self.L_hat_ema_per_layer is not None:
                    for j in range(len(z_curr)):
                        dz_j = z_curr[j] - self.z_prev[j]
                        dz_j_norm = float(dz_j.norm().item())
                        if dz_j_norm < self.min_dz:
                            continue
                        a = grad_curr[j]
                        b = self.grad_prev[j]
                        if a is None and b is None:
                            continue
                        if a is None:
                            dg_j_norm = float(b.detach().norm().item())
                        elif b is None:
                            dg_j_norm = float(a.detach().norm().item())
                        else:
                            dg_j_norm = float((a.detach() - b.detach()).norm().item())
                        ratio_j = min(dg_j_norm / dz_j_norm, self.max_L)
                        self.L_hat_ema_per_layer[j] = (
                            self.ema_beta * self.L_hat_ema_per_layer[j]
                            + (1.0 - self.ema_beta) * ratio_j
                        )

            self._snapshot(z_curr, grad_curr)

        return self._dispatch()

    def _snapshot(self, z_curr, grad_curr):
        self.z_prev = [z.clone().detach() for z in z_curr]
        self.grad_prev = [
            g.clone().detach() if g is not None else None for g in grad_curr
        ]

    def _dispatch(self):
        if self.estimator == 'ema':
            return self.L_hat_ema.clone()
        if self.estimator == 'running_min' and self.L_hat_buffer:
            return torch.tensor(min(self.L_hat_buffer), device=self.device)
        if self.estimator == 'running_median' and self.L_hat_buffer:
            s = sorted(self.L_hat_buffer)
            n = len(s)
            med = 0.5 * (s[n // 2 - 1] + s[n // 2]) if n % 2 == 0 else s[n // 2]
            return torch.tensor(med, device=self.device)
        if self.estimator == 'ema_per_layer_median' and self.L_hat_ema_per_layer:
            nz = [v for v in self.L_hat_ema_per_layer if v > 0]
            if nz:
                s = sorted(nz)
                n = len(s)
                med = 0.5 * (s[n // 2 - 1] + s[n // 2]) if n % 2 == 0 else s[n // 2]
                return torch.tensor(med, device=self.device)
        return torch.tensor(0.0, device=self.device)

    def per_layer_summary(self):
        """Return {max, min, median, max_over_median, max_over_min} of nonzero
        per-layer L̂ EMAs, or None if not tracking per-layer."""
        if not self.L_hat_ema_per_layer:
            return None
        nz = [v for v in self.L_hat_ema_per_layer if v > 0]
        if not nz:
            return None
        s = sorted(nz)
        n = len(s)
        median = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        max_v = s[-1]
        min_v = s[0]
        return {
            'max': max_v,
            'min': min_v,
            'median': median,
            'max_over_median': max_v / max(median, 1e-12),
            'max_over_min': max_v / max(min_v, 1e-12),
        }


class LipschitzFloorOGD:
    """Orchestrator: tracks u = log(σ), runs BB + OGD per call, exposes σ.

    Typical usage (per σ-update event):
        sigma, metrics = floor.step(
            z_curr=W_global,
            grad_curr=grad_global_alpha_weighted,
            primal_res=primal_res_tensor,
            delta_y=delta_y_tensor,
        )
        # log `metrics` to wandb; use `sigma` (float) in subsequent local solves
    """

    def __init__(
        self,
        sigma_init,
        device,
        sigma_min=1e-6,
        sigma_max=1e6,
        eta_u=0.05,
        eta_u_decay='textbook_sc',
        G_clip=10.0,
        estimator='ema',
        window_size=20,
        ema_beta=0.9,
        min_dz=1e-6,
        max_L=1e8,
        lipschitz_floor_alpha=1.0,
        param_names=None,
        eps=1e-8,
        use_lipschitz_floor=None,
        mode='lipschitz',
        heuristic_mu=10.0,
        heuristic_tau=2.0,
        dead_band_mu=1.0,
    ):
        # dead_band_mu: trust-region threshold on |target|. When
        # |log(r/s)| < log(dead_band_mu), the OGD update is skipped
        # ("residuals balanced enough to be in the noise floor"). Mirrors the
        # heuristic's μ-threshold dead-band. dead_band_mu=1.0 disables the
        # dead-band (always update). Suggested: 10.0 to match Boyd's heuristic.
        # Only applied to OGD-based modes (lipschitz / no_floor / true_bal*),
        # not to 'heuristic' (which has the dead-band built in).
        # Backward-compat: legacy `use_lipschitz_floor` boolean still works.
        if use_lipschitz_floor is True:
            mode = 'lipschitz'
        elif use_lipschitz_floor is False:
            mode = 'no_floor'
        # Modes:
        #   'lipschitz'           : OGD on (u-log(r/Δy))² + hard floor σ ≥ exp(L̂)
        #   'no_floor'            : OGD on (u-log(r/Δy))², no floor (legacy convex_bal)
        #   'heuristic'           : Boyd multiplicative on r vs σ·‖Δw_g‖
        #   'true_bal'            : OGD on (u-log(r/s))² with s=σ·‖Δw_g‖, no floor
        #   'true_bal_lipschitz'  : same OGD with the Lipschitz floor on top
        assert mode in ('lipschitz', 'no_floor', 'heuristic',
                        'true_bal', 'true_bal_lipschitz'), \
            f'unknown mode: {mode!r}'

        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.u_min = math.log(sigma_min)
        self.u_max = math.log(sigma_max)
        self.eta_u = eta_u
        self.eta_u_decay = eta_u_decay
        self.G_clip = G_clip
        self.lipschitz_floor_alpha = lipschitz_floor_alpha
        self.eps = eps
        self.mode = mode
        # for back-compat reads + grad-accumulation gate in callers
        self.use_lipschitz_floor = mode in ('lipschitz', 'true_bal_lipschitz')
        self.heuristic_mu = heuristic_mu
        self.heuristic_tau = heuristic_tau
        self.dead_band_mu = float(dead_band_mu)
        self._dead_band_log = (math.log(self.dead_band_mu)
                               if self.dead_band_mu > 1.0 else 0.0)

        self.u_sigma = torch.tensor(
            math.log(max(sigma_init, 1e-12)), device=device
        )
        self.sigma = float(sigma_init)
        self.update_step = 0

        if mode in ('lipschitz', 'true_bal_lipschitz'):
            self.estimator = BBLipschitzEstimator(
                device=device,
                estimator=estimator,
                window_size=window_size,
                ema_beta=ema_beta,
                min_dz=min_dz,
                max_L=max_L,
                num_param_groups=len(param_names) if param_names else None,
            )
        else:
            self.estimator = None
        self.param_names = param_names

    def step(self, z_curr=None, grad_curr=None, primal_res=None,
             delta_y=None, dual_res=None):
        """Run one σ-update event.

        Required inputs depend on `self.mode`:
          - 'lipschitz':           z_curr, grad_curr, primal_res, delta_y
          - 'no_floor':                                primal_res, delta_y
          - 'heuristic':                               primal_res,            dual_res
          - 'true_bal':                                primal_res,            dual_res
          - 'true_bal_lipschitz':  z_curr, grad_curr, primal_res,            dual_res

        Returns (sigma_float, metrics_dict).
        """
        self.update_step += 1

        if not torch.is_tensor(primal_res):
            primal_res = torch.tensor(float(primal_res), device=self.device)

        if self.mode == 'heuristic':
            if dual_res is None:
                raise ValueError("heuristic mode requires `dual_res`")
            if not torch.is_tensor(dual_res):
                dual_res = torch.tensor(float(dual_res), device=self.device)
            sigma_new = heuristic_update_sigma(
                self.sigma,
                primal_res,
                dual_res,
                mu=self.heuristic_mu,
                tau=self.heuristic_tau,
            )
            sigma_new = max(self.sigma_min, min(self.sigma_max, sigma_new))
            self.sigma = float(sigma_new)
            self.u_sigma = torch.tensor(
                math.log(max(self.sigma, 1e-12)), device=self.device
            )
            metrics = {
                'sigma': self.sigma,
                'log_sigma': float(self.u_sigma.item()),
                'sigma/update_step': self.update_step,
                'sigma/heuristic_mu': self.heuristic_mu,
                'sigma/heuristic_tau': self.heuristic_tau,
                'primal_res': float(primal_res.item()),
                'dual_res': float(dual_res.item()),
            }
            return self.sigma, metrics

        # OGD-based modes: lipschitz / no_floor / true_bal / true_bal_lipschitz
        # Pick the "dual proxy" for the residual-balance target:
        #   convex_bal flavors (lipschitz, no_floor): use Δy = ‖w_i^{k+1}-w_i^k‖
        #   true_bal   flavors (true_bal*):           use s  = σ·‖w_g^{k+1}-w_g^k‖
        if self.mode in ('true_bal', 'true_bal_lipschitz'):
            if dual_res is None:
                raise ValueError(f"mode {self.mode!r} requires `dual_res`")
            dual_proxy = dual_res
        else:
            if delta_y is None:
                raise ValueError(f"mode {self.mode!r} requires `delta_y`")
            dual_proxy = delta_y
        if not torch.is_tensor(dual_proxy):
            dual_proxy = torch.tensor(float(dual_proxy), device=self.device)

        eta_u_eff = compute_eta_u_eff(
            self.eta_u, self.update_step, self.eta_u_decay
        )

        # Dead-band: if |log(r / dual_proxy)| < log(dead_band_mu), residuals
        # are within the trust region and OGD doesn't update u. Floor (in
        # lipschitz modes) still applies. Mirrors heuristic's μ-threshold.
        in_dead_band = False
        if self._dead_band_log > 0:
            with torch.no_grad():
                target_check = (
                    torch.log(torch.clamp(primal_res, min=self.eps))
                    - torch.log(torch.clamp(dual_proxy, min=self.eps))
                )
                if abs(float(target_check.item())) < self._dead_band_log:
                    in_dead_band = True
                    eta_u_eff = 0.0  # zero-step: u unchanged by OGD

        u = self.u_sigma.detach()

        if self.mode in ('lipschitz', 'true_bal_lipschitz'):
            L_hat_tensor = self.estimator.update(z_curr, grad_curr)
            (u_new, res_loss, target, log_L,
             floor_active, grad_u) = online_convex_bal_lipschitz_update_u(
                u=u,
                primal_res=primal_res,
                delta_y=dual_proxy,
                L_hat=L_hat_tensor,
                eta_u=eta_u_eff,
                G_clip=self.G_clip,
                u_min=self.u_min,
                u_max=self.u_max,
                eps=self.eps,
                lipschitz_floor_alpha=self.lipschitz_floor_alpha,
            )
        else:  # no_floor or true_bal
            (u_new, res_loss, target, grad_u) = online_convex_bal_update_u(
                u=u,
                primal_res=primal_res,
                delta_y=dual_proxy,
                eta_u=eta_u_eff,
                G_clip=self.G_clip,
                u_min=self.u_min,
                u_max=self.u_max,
                eps=self.eps,
            )
            log_L = None
            floor_active = None
            L_hat_tensor = None

        self.u_sigma = u_new
        self.sigma = float(torch.exp(u_new).item())

        metrics = {
            'sigma': self.sigma,
            'log_sigma': float(u_new.item()),
            'sigma/loss': float(res_loss.item()),
            'sigma/target': float(target.item()),
            'sigma/grad_u': float(grad_u.item()),
            'sigma/eta_u_eff': float(eta_u_eff),
            'sigma/update_step': self.update_step,
            'sigma/in_dead_band': float(in_dead_band),
            'primal_res': float(primal_res.item()),
        }
        if self.mode in ('true_bal', 'true_bal_lipschitz'):
            metrics['dual_res'] = float(dual_proxy.item())
        else:
            metrics['delta_y'] = float(dual_proxy.item())

        if self.mode == 'lipschitz':
            metrics['sigma/log_L_hat'] = float(log_L.item())
            metrics['sigma/L_hat'] = float(torch.exp(log_L).item())
            metrics['sigma/floor_active'] = float(floor_active.item())
            metrics['sigma/L_hat_ema'] = float(L_hat_tensor.item())
            metrics['sigma/L_hat_buffer_size'] = len(self.estimator.L_hat_buffer)

            if self.estimator.last_raw is not None:
                metrics['sigma/L_hat_raw'] = self.estimator.last_raw
                metrics['sigma/dz_norm'] = self.estimator.last_dz_norm
                metrics['sigma/dg_norm'] = self.estimator.last_dg_norm

            pl = self.estimator.per_layer_summary()
            if pl is not None:
                for k, v in pl.items():
                    metrics[f'sigma/L_hat_per_layer/{k}'] = v
                if self.param_names is not None:
                    for j, name in enumerate(self.param_names):
                        if self.estimator.L_hat_ema_per_layer[j] > 0:
                            metrics[f'sigma/L_hat_per_layer/{name}'] = (
                                self.estimator.L_hat_ema_per_layer[j]
                            )

        return self.sigma, metrics
