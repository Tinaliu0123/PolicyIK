"""Latent-alignment critic: a state-action tower and a goal tower whose squared difference feeds FlashSAC's categorical head."""
import torch
import torch.nn as nn

from flash_rl.agents.flashSAC.layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
)


def lan_cols_from_layout(lay: dict) -> dict:
    """Observation columns of the two towers: the arm's own channels and the state-goal channels go to the
    state-action tower, the goal pose to the goal tower."""
    cols = lambda *ks: [i for k in ks if k in lay for i in range(*lay[k])]
    out = {"state": cols("q", "dq", "ee_pos", "ee_rot", "prev_action", "ref_lead", "err_pos", "kp_err",
                         "zoom_pos_0", "zoom_pos_1", "zoom_ori_0", "zoom_ori_1", "rho_best"),
           "goal": cols("goal_pos", "goal_rot")}
    if sorted(out["state"] + out["goal"]) != list(range(lay["obs_dim"])):
        raise ValueError(f"the critic columns do not cover the {lay['obs_dim']}-D observation")
    return out


class _EnsembleTower(nn.Module):
    """FlashSAC critic trunk: embedder, residual blocks and RMS post-norm, ensembled over num_qs."""

    def __init__(self, num_qs: int, input_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        return self.post_norm(x)


class FlashSACLANCritic(nn.Module):
    """Drop-in replacement for FlashSACDoubleCritic: phi(s, a) and psi(g) towers, u = (phi - psi)^2, categorical head."""

    def __init__(self, num_blocks: int, action_dim: int, hidden_dim: int, num_bins: int, min_v: float, max_v: float,
                 cols: dict, goal_blocks: int = 1, num_qs: int = 2):
        super().__init__()
        self.num_qs = num_qs
        self.register_buffer("idx_state", torch.as_tensor(cols["state"], dtype=torch.long), persistent=False)
        self.register_buffer("idx_goal", torch.as_tensor(cols["goal"], dtype=torch.long), persistent=False)
        self.sa_tower = _EnsembleTower(num_qs, len(cols["state"]) + action_dim, hidden_dim, num_blocks)
        self.g_tower = _EnsembleTower(num_qs, len(cols["goal"]), hidden_dim, goal_blocks)
        self.predictor = EnsembleCategoricalValue(num_ensemble=num_qs, hidden_dim=hidden_dim, num_bins=num_bins,
                                                  min_v=min_v, max_v=max_v)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sa = torch.cat((observations.index_select(-1, self.idx_state), actions), dim=-1)
        g = observations.index_select(-1, self.idx_goal)
        expand = lambda x: x.unsqueeze(0).expand(self.num_qs, -1, -1)
        phi = self.sa_tower(expand(sa), training)
        psi = self.g_tower(expand(g), training)
        return self.predictor((phi - psi) ** 2, training)
