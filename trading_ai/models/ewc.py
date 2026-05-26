"""
Elastic Weight Consolidation (EWC)

Prevents catastrophic forgetting when the AI learns from new market regimes.
Computes Fisher Information Matrix from past task data and adds a quadratic
penalty to the loss when training on new tasks.

Usage with PPOAgent:
    ewc = EWC(ppo_agent.network, dataset_size=200)
    ewc.register_task(obs_batch)         # after task A
    # then during task B training:
    penalty = ewc.penalty(ppo_agent.network)
    loss = original_loss + penalty
"""
import logging
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    Parameters
    ----------
    model        : the nn.Module to protect (e.g. PPOAgent.network)
    dataset_size : number of samples used to estimate Fisher (default 200)
    """

    ewc_loss_coeff: float = 1000.0

    def __init__(self, model: nn.Module, dataset_size: int = 200):
        self.model        = model
        self.dataset_size = dataset_size

        # Stored after register_task()
        self._params_old: Optional[dict]  = None   # {name → tensor}
        self._fisher:     Optional[dict]  = None   # {name → tensor}
        self._task_count: int             = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_task(self, obs_batch: torch.Tensor, n_samples: int = 100):
        """
        Compute the diagonal Fisher Information Matrix for the current task.
        Call this at the END of training on a task (before starting a new task).

        obs_batch : (N, obs_size) float tensor  — recent observations
        n_samples : how many samples to use for Fisher estimation
        """
        device = next(self.model.parameters()).device
        obs_batch = obs_batch.to(device)

        # Use at most n_samples
        if obs_batch.size(0) > n_samples:
            idx = torch.randperm(obs_batch.size(0))[:n_samples]
            obs_batch = obs_batch[idx]

        # Save current parameter values as "old task" anchors
        self._params_old = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Compute diagonal Fisher ≈ E[( ∂ log π / ∂ θ )^2]
        self.model.eval()
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.model.named_parameters()
                  if param.requires_grad}

        for obs in obs_batch:
            self.model.zero_grad()
            obs_in = obs.unsqueeze(0)

            # Forward pass — get log-probabilities from actor head
            try:
                logits, _ = self.model(obs_in)
            except TypeError:
                # Some models may not return a tuple
                logits = self.model(obs_in)

            log_probs = torch.log_softmax(logits, dim=-1)
            # Pick the most likely action's log-prob for Fisher
            chosen    = log_probs.argmax(dim=-1)
            loss      = -log_probs[0, chosen]
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)

        n = max(obs_batch.size(0), 1)
        self._fisher = {name: f / n for name, f in fisher.items()}

        self._task_count += 1
        self.model.train()
        logger.info(
            "EWC: registered task #%d | params=%d | samples=%d",
            self._task_count,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            obs_batch.size(0),
        )

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularisation penalty.

        penalty = (ewc_loss_coeff / 2) * Σ F_i * (θ_i - θ*_i)^2

        Returns a scalar tensor (0.0 if no task has been registered yet).
        """
        if self._fisher is None or self._params_old is None:
            # No previous task registered → no penalty
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            fisher_val = self._fisher.get(name)
            old_val    = self._params_old.get(name)
            if fisher_val is None or old_val is None:
                continue
            # Move to same device as current param
            fisher_val = fisher_val.to(param.device)
            old_val    = old_val.to(param.device)
            penalty   += (fisher_val * (param - old_val).pow(2)).sum()

        return (self.ewc_loss_coeff / 2.0) * penalty

    @property
    def task_count(self) -> int:
        return self._task_count
