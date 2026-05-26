"""
TFT-based encoder components for Semi-Pro ActorCriticV2.

Components:
  GatedResidualNetwork       — LayerNorm(x + GLU(FC(ELU(FC(x))))) with skip
  VariableSelectionNetwork   — soft-selects important indicator groups
  TemporalTransformerEncoder — 4-layer, 8-head transformer with rolling context
  MoEActionHead              — 3 regime experts + soft gating network
  ActorCriticV2              — full Semi-Pro actor-critic using the above
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Observation layout (must match trading_env.py)
N_IND      = 40    # technical indicator features
N_PRICE    = 256   # PriceSequenceEncoder features
N_IND_VARS = 8     # indicator groups for VSN (40 / 8 = 5 per group)


# ── Gated Residual Network ─────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Stable non-linear transform: LayerNorm(skip(x) + gate(hidden))
    GLU gate = h1 * sigmoid(h2) where [h1, h2] = FC(ELU(FC(x)))
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim * 2)   # *2 for GLU split
        self.norm = nn.LayerNorm(output_dim)
        self.skip = (nn.Linear(input_dim, output_dim)
                     if input_dim != output_dim else nn.Identity())
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        h1, h2 = h.chunk(2, dim=-1)
        return self.norm(self.skip(x) + h1 * torch.sigmoid(h2))


# ── Variable Selection Network ─────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """
    Learns which indicator groups matter most for the current market state.
    Splits total_input_dim into n_vars equal-sized groups, processes each
    through its own GRN, then applies a softmax-weighted sum.
    """

    def __init__(self, total_input_dim: int, n_vars: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.n_vars  = n_vars
        self.var_dim = total_input_dim // n_vars

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(self.var_dim, output_dim, output_dim, dropout)
            for _ in range(n_vars)
        ])
        self.selection = GatedResidualNetwork(total_input_dim, output_dim, n_vars, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks    = x.split(self.var_dim, dim=-1)              # list of (B, var_dim)
        var_feats = torch.stack(
            [grn(chunk) for grn, chunk in zip(self.var_grns, chunks)], dim=1
        )                                                      # (B, n_vars, output_dim)
        weights = F.softmax(self.selection(x), dim=-1).unsqueeze(-1)  # (B, n_vars, 1)
        return (var_feats * weights).sum(dim=1)                # (B, output_dim)


# ── Temporal Transformer Encoder ───────────────────────────────────────────────

class TemporalTransformerEncoder(nn.Module):
    """
    4-layer, 8-head transformer with a rolling inference context buffer.

    Training mode  — each step processed independently (safe for PPO batches).
    Inference mode — accumulates a context of the last `context_len` steps so
                     the model can attend to recent market history.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.1, context_len: int = 32):
        super().__init__()
        # Guarantee d_model divisible by n_heads
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.d_model     = d_model
        self.context_len = context_len
        self._context: Optional[torch.Tensor] = None   # (B, T, d_model)

    def reset_context(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self._context = torch.zeros(batch_size, 0, self.d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model)
        x_seq = x.unsqueeze(1)   # (B, 1, d_model)

        if not self.training and self._context is not None:
            B = x.size(0)
            if self._context.size(0) == B and self._context.size(1) > 0:
                x_seq = torch.cat([self._context, x_seq], dim=1)
            if x_seq.size(1) > self.context_len:
                x_seq = x_seq[:, -self.context_len:]
            out = self.transformer(x_seq)
            self._context = x_seq[:, 1:].detach()   # slide window
        else:
            out = self.transformer(x_seq)

        return out[:, -1]   # (B, d_model) — last position


# ── Mixture-of-Experts Action Head ────────────────────────────────────────────

class MoEActionHead(nn.Module):
    """
    3 specialist expert networks (trending / ranging / volatile)
    blended by a soft gating network conditioned on the context.
    """

    def __init__(self, input_dim: int, n_actions: int, n_experts: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        hidden = input_dim // 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_actions),
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, A)
        weights    = F.softmax(self.gate(x), dim=-1).unsqueeze(-1)      # (B, E, 1)
        return (expert_out * weights).sum(dim=1)                        # (B, A)


# ── ActorCriticV2 — Semi-Pro Architecture ─────────────────────────────────────

class ActorCriticV2(nn.Module):
    """
    Semi-Pro ActorCritic:

      Indicators (40)  → VariableSelectionNetwork (8 groups × 5 features)
      Price seq (256)  → Linear projection
                      ↓ cat
             TemporalTransformerEncoder (4L, 8H)
                      ↓
      Actor: MoEActionHead (3 experts)  |  Critic: GatedResidualNetwork → 1
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        self.obs_size    = obs_size
        self.n_actions   = n_actions
        self.hidden_size = hidden_size
        half = hidden_size // 2

        # Indicator branch
        self.vsn = VariableSelectionNetwork(N_IND, N_IND_VARS, half)

        # Price branch
        self.price_proj = nn.Sequential(
            nn.Linear(N_PRICE, half),
            nn.LayerNorm(half),
            nn.GELU(),
        )

        # Temporal transformer (hidden_size = half + half)
        self.temporal = TemporalTransformerEncoder(
            d_model=hidden_size, n_heads=8, n_layers=4
        )

        # Actor (MoE) + Critic (GRN)
        self.actor       = MoEActionHead(hidden_size, n_actions)
        self.critic_head = GatedResidualNetwork(hidden_size, half, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_context(self, batch_size: int = 1):
        self.temporal.reset_context(batch_size)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        ind   = obs[:, :N_IND]
        price = obs[:, N_IND: N_IND + N_PRICE]
        feat  = torch.cat([self.vsn(ind), self.price_proj(price)], dim=-1)
        return self.temporal(feat)   # (B, hidden_size)

    def forward(self, obs: torch.Tensor):
        ctx    = self._encode(obs)
        logits = self.actor(ctx)
        logits = logits - logits.mean(dim=-1, keepdim=True)   # dueling norm
        value  = self.critic_head(ctx).squeeze(-1)
        return logits, value

    def get_action(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self(obs)
        return torch.softmax(logits, dim=-1)
