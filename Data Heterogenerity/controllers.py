"""Penalty-parameter (σ / ρ) controllers for SISA-family ADMM.

Pure-Python class-based controllers, ported from
`online_admm_applications/online_admm_experiments/controllers.py` (NumPy/JAX)
into a PyTorch-friendly form. No NumPy dependency. Controllers consume
scalar residuals (primal_norm, dual_base_norm) and return updated σ.

Available controllers:
  - FixedRho                       — paper baseline (no adaptation)
  - ResidualBalancing              — Boyd 2011 §3.4.1 (the "heuristic"),
                                     with optional EMA smoothing and k_max
                                     cutoff (He-Yang-Wang 2000 convergence
                                     guarantee)
  - NormalizedResidualBalancing    — Wohlberg 2017, residual-balance after
                                     dividing by ADMM stopping thresholds
  - SpectralAADMM                  — Xu-Figueiredo-Goldstein, AISTATS 2017
                                     (the canonical SOTA adaptive-ρ baseline)
  - OnlineOGD                      — OGD on u = log(σ) against residual
                                     -balance loss, with optional decay /
                                     gradient clipping / dead-band /
                                     Lipschitz floor

Integration: import this module from experiment_sisa_practise_admm.py or
experiment_sisa_practise_online.py. Each controller exposes:

    state = ctrl.init_state(rho0)
    decision = ctrl.update(k, state, primal_norm, dual_base_norm,
                            context={...})
    new_rho = decision.rho
    new_state = decision.state

`context` is a free-form dict for controller-specific extras:
  - SpectralAADMM needs `lambda_hat`, `lambda`, `h_value`, `g_value`
    (intermediate dual, final dual, A·x, B·z; for SISA consensus
    A=I per client, B=-I per client, so h = w_i and g = -w_g).
  - NormalizedResidualBalancing optionally consumes `primal_threshold`
    and `dual_threshold` (the ADMM stopping thresholds; default 1).
  - OnlineOGD with Lipschitz floor optionally consumes `L_hat`.

References:
  Boyd et al. (2011), "Distributed Optimization and Statistical Learning
    via the Alternating Direction Method of Multipliers", §3.4.1.
  He, Yang, Wang (2000), "Self-adaptive penalty parameter strategy for
    ADMM" (k_max cutoff for σ-adjustment convergence guarantee).
  Wohlberg (2017), "ADMM Penalty Parameter Selection by Residual
    Balancing".
  Xu, Figueiredo, Goldstein (2017), "Adaptive ADMM with Spectral Penalty
    Parameter Selection", AISTATS.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp, log, sqrt
from typing import Optional, Sequence

EPS = 1e-12


@dataclass
class PenaltyState:
    """Per-controller persistent state. Initialize via ctrl.init_state(rho0)."""
    rho: float
    log_rho: float
    # EMA-smoothed residuals (set by controllers that smooth)
    primal_ema: Optional[float] = None
    dual_base_ema: Optional[float] = None
    # OGD state
    u: Optional[float] = None  # log σ
    # SpectralAADMM state (per-iteration cached vectors)
    spectral_lambda_hat: Optional[Sequence[float]] = None
    spectral_lambda: Optional[Sequence[float]] = None
    spectral_h: Optional[Sequence[float]] = None
    spectral_g: Optional[Sequence[float]] = None


@dataclass
class ControllerDecision:
    """Return value of a controller update: new ρ, plus diagnostics."""
    rho: float
    changed: bool
    grad: float = 0.0
    loss: float = 0.0
    reason: str = "ok"
    state: Optional[PenaltyState] = None


class PenaltyController:
    """Base class. Subclasses override update()."""

    name: str = "base"

    def init_state(self, rho0: float) -> PenaltyState:
        return PenaltyState(rho=rho0, log_rho=log(max(rho0, EPS)))

    def update(self, k: int, state: PenaltyState,
               primal_norm: float, dual_base_norm: float,
               context: Optional[dict] = None) -> ControllerDecision:
        raise NotImplementedError


# ============================================================================
# 1. FixedRho — paper baseline (no adaptation)
# ============================================================================

@dataclass
class FixedRho(PenaltyController):
    name: str = "fixed"

    def update(self, k, state, primal_norm, dual_base_norm, context=None):
        return ControllerDecision(state.rho, False, 0.0, 0.0, "fixed", state)


# ============================================================================
# 2. ResidualBalancing — Boyd 2011 §3.4.1 (with optional improvements)
# ============================================================================

@dataclass
class ResidualBalancing(PenaltyController):
    """Boyd's residual-balance heuristic.

    Update rule:
        s = ρ · ‖dual_base‖           (canonical scaled dual residual)
        ρ ← τ·ρ      if  primal > μ·s
        ρ ← ρ/τ      if  s > μ·primal
        ρ unchanged  otherwise

    Optional improvements (vanilla Boyd has neither; both are recommended
    for stochastic problems by the standard ADMM literature):
      - EMA smoothing of residuals (β=0.9 default; set ema_beta=0 to disable)
      - k_max cutoff after which σ stops adjusting (He-Yang-Wang 2000;
        provides Σ τ_k < ∞ convergence guarantee). Default k_max=50.

    To run the BARE Boyd rule (no smoothing, no cutoff), set ema_beta=0
    and k_max=10**9.
    """
    mu: float = 10.0
    tau: float = 2.0
    rho_min: float = 1e-6
    rho_max: float = 1e8
    ema_beta: float = 0.9            # 0 disables smoothing
    k_max: int = 50                  # set to 10**9 for no cutoff
    update_period: int = 1
    name: str = "residual_balance"

    def update(self, k, state, primal_norm, dual_base_norm, context=None):
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "period", state)
        if k > self.k_max:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "kmax_cutoff", state)

        # EMA smoothing
        if self.ema_beta > 0:
            if state.primal_ema is None:
                primal_ema = primal_norm
                dual_base_ema = dual_base_norm
            else:
                primal_ema = self.ema_beta * state.primal_ema + (1 - self.ema_beta) * primal_norm
                dual_base_ema = (self.ema_beta * state.dual_base_ema
                                 + (1 - self.ema_beta) * dual_base_norm)
        else:
            primal_ema = primal_norm
            dual_base_ema = dual_base_norm

        primal = primal_ema
        dual = state.rho * dual_base_ema  # canonical dual: σ · ‖Δw_g‖

        rho = state.rho
        reason = "balanced"
        if primal > self.mu * max(dual, EPS):
            rho = min(self.rho_max, state.rho * self.tau)
            reason = "primal_gt_dual"
        elif dual > self.mu * max(primal, EPS):
            rho = max(self.rho_min, state.rho / self.tau)
            reason = "dual_gt_primal"

        new_state = replace(state, rho=rho, log_rho=log(max(rho, EPS)),
                             primal_ema=primal_ema, dual_base_ema=dual_base_ema)
        loss = 0.5 * (log(max(dual, EPS)) - log(max(primal, EPS))) ** 2
        return ControllerDecision(rho, rho != state.rho, 0.0, loss, reason, new_state)


# ============================================================================
# 3. NormalizedResidualBalancing — Wohlberg 2017
# ============================================================================

@dataclass
class NormalizedResidualBalancing(PenaltyController):
    """Wohlberg-style residual balance after dividing primal and dual by
    ADMM stopping thresholds (`primal_threshold`, `dual_threshold` in
    `context`). Often more robust than vanilla Boyd because the comparison
    is scale-aware.

    For SISA, sensible thresholds are the absolute + relative ADMM stopping
    criteria; pass them in via `context`. If absent, both default to 1
    (degenerates to plain ResidualBalancing without the EMA / k_max).
    """
    mu: float = 10.0
    tau: float = 2.0
    rho_min: float = 1e-6
    rho_max: float = 1e8
    update_period: int = 1
    name: str = "residual_balance_normalized"

    def update(self, k, state, primal_norm, dual_base_norm, context=None):
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "period", state)

        context = context or {}
        primal_threshold = max(float(context.get("primal_threshold", 1.0)), EPS)
        dual_threshold = max(float(context.get("dual_threshold", 1.0)), EPS)

        dual_norm = state.rho * dual_base_norm
        primal_scaled = primal_norm / primal_threshold
        dual_scaled = dual_norm / dual_threshold

        rho = state.rho
        reason = "balanced"
        if primal_scaled > self.mu * max(dual_scaled, EPS):
            rho = min(self.rho_max, state.rho * self.tau)
            reason = "primal_gt_dual_normalized"
        elif dual_scaled > self.mu * max(primal_scaled, EPS):
            rho = max(self.rho_min, state.rho / self.tau)
            reason = "dual_gt_primal_normalized"

        new_state = replace(state, rho=rho, log_rho=log(max(rho, EPS)))
        loss = 0.5 * (log(max(dual_scaled, EPS)) - log(max(primal_scaled, EPS))) ** 2
        return ControllerDecision(rho, rho != state.rho, 0.0, loss, reason, new_state)


# ============================================================================
# 4. SpectralAADMM — Xu, Figueiredo, Goldstein (AISTATS 2017)
# ============================================================================

@dataclass
class SpectralAADMM(PenaltyController):
    """Xu-Figueiredo-Goldstein spectral adaptive ρ.

    Estimates two spectral stepsizes from cached values of (intermediate
    dual λ_hat, final dual λ, A·x = h, B·z = g) at the current and
    previous iteration. Applies a correlation safeguard (only updates ρ
    when the angle between residual changes and dual changes is large
    enough).

    Required `context` entries (lists/sequences of floats; flatten across
    parameters):
        lambda_hat — intermediate unscaled dual after the x-update
        lambda     — final unscaled dual after the z-update
        h_value    — A·x = w_i (per-client local iterate, flattened)
        g_value    — B·z = -w_g (negated global, flattened)

    For SISA's consensus ADMM, A = I, B = -I per client, so:
        h_value = w_i^{k+1} (just-updated local)
        g_value = -w_g^{k+1} (just-updated global, negated)
        lambda  = π_i^{k+1} (just-updated dual)
        lambda_hat = π_i^{intermediate}, the dual after the x-update but
                     before the z-update. SISA's standard order doesn't
                     expose this; you'd need to split the dual update into
                     two stages: π_hat = π_old + ρ·(w_i^{k+1} - w_g^k),
                     then π_final = π_hat - ρ·(w_g^{k+1} - w_g^k).
                     See Xu et al. for details.

    If the context is missing or the flag day hasn't fired yet, returns a
    no-op decision (rho unchanged).
    """
    rho_min: float = 1e-6
    rho_max: float = 1e8
    update_period: int = 2
    correlation_threshold: float = 0.2
    name: str = "spectral_aadmm_xu2017"

    def update(self, k, state, primal_norm, dual_base_norm, context=None):
        if context is None:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "missing_context", state)
        try:
            lambda_hat = list(map(float, context["lambda_hat"]))
            lam = list(map(float, context["lambda"]))
            h_value = list(map(float, context["h_value"]))
            g_value = list(map(float, context["g_value"]))
        except (KeyError, TypeError):
            return ControllerDecision(state.rho, False, 0.0, 0.0, "missing_context", state)

        carried = replace(
            state,
            spectral_lambda_hat=list(lambda_hat),
            spectral_lambda=list(lam),
            spectral_h=list(h_value),
            spectral_g=list(g_value),
        )

        if state.spectral_lambda_hat is None or (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "spectral_wait", carried)

        d_lambda_hat = [a - b for a, b in zip(lambda_hat, state.spectral_lambda_hat)]
        d_lambda = [a - b for a, b in zip(lam, state.spectral_lambda)]
        d_h = [a - b for a, b in zip(h_value, state.spectral_h)]
        d_g = [a - b for a, b in zip(g_value, state.spectral_g)]

        alpha_step, alpha_cor = self._spectral_step(d_h, d_lambda_hat)
        beta_step, beta_cor = self._spectral_step(d_g, d_lambda)

        rho = state.rho
        reason = "spectral_reject"
        if alpha_cor > self.correlation_threshold and beta_cor > self.correlation_threshold:
            rho = float((alpha_step * beta_step) ** 0.5)
            reason = "spectral_geometric"
        elif alpha_cor > self.correlation_threshold:
            rho = alpha_step
            reason = "spectral_alpha"
        elif beta_cor > self.correlation_threshold:
            rho = beta_step
            reason = "spectral_beta"

        rho = float(min(max(rho, self.rho_min), self.rho_max))
        new_state = replace(carried, rho=rho, log_rho=log(max(rho, EPS)))
        changed = abs(rho / state.rho - 1.0) >= 1e-12
        loss = max(alpha_cor, 0.0) * max(beta_cor, 0.0)
        return ControllerDecision(rho, changed, 0.0, loss, reason, new_state)

    @staticmethod
    def _spectral_step(delta_grad, delta_dual):
        gn = sqrt(sum(v * v for v in delta_grad))
        dn = sqrt(sum(v * v for v in delta_dual))
        if gn <= EPS or dn <= EPS:
            return 1.0, -1.0
        dot = sum(a * b for a, b in zip(delta_grad, delta_dual))
        cor = dot / max(gn * dn, EPS)
        if dot <= EPS:
            return 1.0, cor

        gg = sum(v * v for v in delta_grad)
        dd = sum(v * v for v in delta_dual)
        step_sd = dd / max(dot, EPS)
        step_mg = dot / max(gg, EPS)
        if step_sd <= EPS or step_mg <= EPS:
            return 1.0, cor
        if 2.0 * step_mg > step_sd:
            step = step_mg
        else:
            step = step_sd - 0.5 * step_mg
        return max(step, EPS), cor


# ============================================================================
# 5. OnlineOGD — projected OGD on u = log(σ) with residual-balance loss
# ============================================================================

@dataclass
class OnlineOGD(PenaltyController):
    """OGD on u = log(σ) against the convex-bal target
        L(u) = 0.5 · (u − log(primal/dual))^2

    Diminishing-step variant uses η_k = 1/(2k) (textbook strongly-convex
    OGD step; parameter-free). Set decay='none' for constant η.

    Optional dead-band: skip update when |log(primal/dual)| < log(μ_dead)
    (analogous to heuristic's μ-trust region; default disabled with μ=1).

    Optional Lipschitz floor (set lipschitz_floor_alpha > 0 and pass
    context['L_hat']): projects u onto u ≥ log(α · L̂).
    """
    eta_u: float = 0.05
    decay: str = "textbook_sc"   # 'none' | 'textbook_sc' | 'inv_sqrt' | 'inverse'
    G_clip: float = 5.0
    rho_min: float = 1e-6
    rho_max: float = 1e8
    update_period: int = 1
    dead_band_mu: float = 1.0    # > 1 to enable dead-band
    lipschitz_floor_alpha: float = 0.0  # 0 disables floor
    name: str = "online_ogd"

    def init_state(self, rho0):
        return PenaltyState(rho=rho0, log_rho=log(max(rho0, EPS)),
                            u=log(max(rho0, EPS)))

    def _eta_eff(self, k):
        if self.decay == "none":
            return self.eta_u
        if self.decay == "textbook_sc":
            return 1.0 / (2.0 * max(k, 1))
        if self.decay == "inv_sqrt":
            return self.eta_u / sqrt(max(k, 1))
        if self.decay == "inverse":
            return self.eta_u / max(k, 1)
        return self.eta_u

    def update(self, k, state, primal_norm, dual_base_norm, context=None):
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "period", state)

        primal = max(primal_norm, EPS)
        # Use σ-scaled dual residual (Boyd canonical) by default.
        dual = max(state.rho * dual_base_norm, EPS)

        target = log(primal) - log(dual)
        u = state.u if state.u is not None else log(max(state.rho, EPS))

        if self.dead_band_mu > 1.0 and abs(target) < log(self.dead_band_mu):
            return ControllerDecision(state.rho, False, 0.0, 0.0, "dead_band", state)

        eta = self._eta_eff(k + 1)
        diff = u - target
        grad_u = max(-self.G_clip, min(self.G_clip, 2.0 * diff))
        u_new = u - eta * grad_u

        # Lipschitz floor
        if self.lipschitz_floor_alpha > 0 and context is not None:
            L_hat = context.get("L_hat")
            if L_hat is not None and L_hat > 0:
                log_floor = log(max(L_hat, EPS)) + log(self.lipschitz_floor_alpha)
                u_new = max(u_new, log_floor)

        u_new = max(log(self.rho_min), min(log(self.rho_max), u_new))
        rho_new = exp(u_new)
        new_state = replace(state, rho=rho_new, log_rho=u_new, u=u_new)
        loss = 0.5 * diff * diff
        return ControllerDecision(rho_new, rho_new != state.rho,
                                   grad_u, loss, "ogd_step", new_state)


# Convenience registry for generators / config dispatch.
CONTROLLERS = {
    "fixed": FixedRho,
    "residual_balance": ResidualBalancing,
    "residual_balance_normalized": NormalizedResidualBalancing,
    "spectral_aadmm_xu2017": SpectralAADMM,
    "online_ogd": OnlineOGD,
}


def make_controller(name: str, **kwargs) -> PenaltyController:
    """Factory: get controller by name with optional kwargs."""
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; "
                         f"available: {list(CONTROLLERS.keys())}")
    return CONTROLLERS[name](**kwargs)
