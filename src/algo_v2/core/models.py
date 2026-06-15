import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class DualBranchCNNExtractor(TorchModelV2, nn.Module):
    """
    Dual-Branch Hierarchical CNN with optional Transformer Self-Attention.

    State layout (14 indicators):  [cash(1)] [holdings(N)] [stock_0: close+14inds] … [stock_N-1: close+14inds]
    obs_dim = 1 + 16*N  →  N = (obs_dim - 1) // 16
    Features fed to CNN per stock = price + 14 indicators = 15
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        # SAC/TQC pass num_outputs=None; LSTMWrapper may pass a tuple — normalise to int
        if num_outputs is None:
            import numpy as np

            num_outputs = int(np.prod(action_space.shape)) * 2
        elif not isinstance(num_outputs, int):
            import math

            num_outputs = int(math.prod(num_outputs)) if hasattr(num_outputs, "__iter__") else int(num_outputs)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.use_attention = custom_config.get("use_attention", True)
        self.use_micro = custom_config.get("use_micro", True)
        self.use_macro = custom_config.get("use_macro", True)

        obs_dim = obs_space.shape[0]
        self.N = (obs_dim - 1) // 16  # number of stocks
        self.num_features = 15  # price + 14 indicators per stock
        self.embed_dim = 64

        self.input_norm = nn.LayerNorm(self.num_features)
        self.feature_projection = nn.Linear(self.num_features, self.embed_dim)

        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(self.embed_dim)

        fusion_input_dim = 1 + self.N  # cash scalar + holdings vector

        if self.use_micro:
            # Input (B, 1, N, 64) → Conv(1,5) → (B,16,N,60) → MaxPool(1,2) → (B,16,N,30)
            self.micro_flat_dim = 16 * self.N * 30
            self.micro_branch = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 5)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Flatten(),
                nn.LayerNorm(self.micro_flat_dim),
            )
            fusion_input_dim += self.micro_flat_dim
        else:
            self.micro_flat_dim = 0

        if self.use_macro:
            # Input (B, 1, N, 64) → Conv(5,5) → (B,32,N-4,60) → MaxPool(2,2) → (B,32,(N-4)//2,30)
            macro_h = (self.N - 4) // 2
            self.macro_flat_dim = 32 * macro_h * 30
            self.macro_branch = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Flatten(),
                nn.LayerNorm(self.macro_flat_dim),
            )
            fusion_input_dim += self.macro_flat_dim
        else:
            self.macro_flat_dim = 0

        if not self.use_micro and not self.use_macro:
            self.fallback_flat_dim = self.N * self.embed_dim
            fusion_input_dim += self.fallback_flat_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.num_outputs),
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.num_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()

        cash = obs[:, 0:1]
        holdings = obs[:, 1 : 1 + self.N]
        market_data = obs[:, 1 + self.N :]

        seq_input = market_data.view(-1, self.N, self.num_features)
        seq_input = self.input_norm(seq_input)
        embedded_seq = self.feature_projection(seq_input)

        if self.use_attention:
            attended_seq, _ = self.attention(embedded_seq, embedded_seq, embedded_seq)
            embedded_seq = self.attn_norm(embedded_seq + attended_seq)

        market_matrix = embedded_seq.unsqueeze(1)

        to_fuse = [cash, holdings]
        if self.use_micro:
            to_fuse.append(self.micro_branch(market_matrix))
        if self.use_macro:
            to_fuse.append(self.macro_branch(market_matrix))
        if not self.use_micro and not self.use_macro:
            to_fuse.append(embedded_seq.view(-1, self.fallback_flat_dim))

        features = self.fusion(torch.cat(to_fuse, dim=1))
        self._cur_value = self.value_branch(features).squeeze(1)
        return features, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


class IndicatorAttnExtractor(TorchModelV2, nn.Module):
    """
    Two orthogonal attention layers — no CNN branches.

    1. Indicator-scale attention (temporal): 15 technical indicator types as tokens,
       embed_dim = N stocks.  Learns cross-scale relationships: "does short-term RSI
       confirm long-term SMA right now?"
    2. Spatial stock attention: 49 stocks as tokens after feature projection.

    Purely feedforward — no hidden state, no seq_lens.
    Compatible with SAC/TQC off-policy replay (each sample is self-contained).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        if num_outputs is None:
            import numpy as np

            num_outputs = int(np.prod(action_space.shape)) * 2
        elif not isinstance(num_outputs, int):
            import math

            num_outputs = int(math.prod(num_outputs)) if hasattr(num_outputs, "__iter__") else int(num_outputs)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = obs_space.shape[0]
        self.N = (obs_dim - 1) // 16
        self.num_features = 15

        self.input_norm = nn.LayerNorm(self.num_features)

        # Indicator-scale attention: 15 indicator types as tokens, N-dim embeddings
        # num_heads=7 divides embed_dim=49 evenly (7 per head)
        self.ind_attn = nn.MultiheadAttention(embed_dim=self.N, num_heads=7, batch_first=True)
        self.ind_attn_norm = nn.LayerNorm(self.N)

        self.feature_projection = nn.Linear(self.num_features, 64)

        # Spatial stock attention: 49 stock tokens, 64-dim embeddings
        self.stock_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.stock_attn_norm = nn.LayerNorm(64)

        fusion_input_dim = 1 + self.N + self.N * 64  # cash + holdings + market_flat
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.num_outputs),
        )
        self.value_branch = nn.Sequential(
            nn.Linear(self.num_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        cash = obs[:, 0:1]
        holdings = obs[:, 1 : 1 + self.N]
        market_data = obs[:, 1 + self.N :]

        seq_input = market_data.view(-1, self.N, self.num_features)
        seq_input = self.input_norm(seq_input)  # (B, N, 15)

        # Indicator-scale attention: treat 15 feature types as temporal tokens
        mkt_t = seq_input.transpose(1, 2)  # (B, 15, N)
        ind_attended, _ = self.ind_attn(mkt_t, mkt_t, mkt_t)
        seq_input = self.ind_attn_norm(mkt_t + ind_attended).transpose(1, 2)  # (B, N, 15)

        # Project to 64-dim then apply spatial stock attention
        projected = self.feature_projection(seq_input)  # (B, N, 64)
        stock_attended, _ = self.stock_attn(projected, projected, projected)
        projected = self.stock_attn_norm(projected + stock_attended)  # (B, N, 64)

        market_flat = projected.flatten(1)  # (B, N*64)
        features = self.fusion(torch.cat([cash, holdings, market_flat], dim=1))
        self._cur_value = self.value_branch(features).squeeze(1)
        return features, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


class TemporalAttnExtractor(DualBranchCNNExtractor):
    """
    DualBranchCNNExtractor (micro + macro CNN branches + spatial stock attention)
    extended with indicator-scale temporal attention applied before the CNN.

    Attention dimensions:
      - Temporal (new): 15 indicator time scales — which scales matter most now?
      - Spatial (inherited): 49 stocks — which stocks are correlated now?
      - CNN (inherited): local patterns across the stock × feature matrix

    Feedforward — no hidden state — works correctly with SAC/TQC replay buffers.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        # Ensure all branches and spatial attention are enabled (full god_tier)
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)
        self.ind_attn = nn.MultiheadAttention(embed_dim=self.N, num_heads=7, batch_first=True)
        self.ind_attn_norm = nn.LayerNorm(self.N)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        cash = obs[:, 0:1]
        holdings = obs[:, 1 : 1 + self.N]
        market_data = obs[:, 1 + self.N :]

        seq_input = market_data.view(-1, self.N, self.num_features)
        seq_input = self.input_norm(seq_input)  # (B, N, 15)

        # Indicator-scale attention (new temporal layer before CNN)
        mkt_t = seq_input.transpose(1, 2)  # (B, 15, N)
        ind_attended, _ = self.ind_attn(mkt_t, mkt_t, mkt_t)
        mkt_t = self.ind_attn_norm(mkt_t + ind_attended)
        seq_input = mkt_t.transpose(1, 2)  # (B, N, 15)

        # Feature projection + spatial stock attention (inherited from parent)
        embedded_seq = self.feature_projection(seq_input)  # (B, N, 64)
        if self.use_attention:
            attended_seq, _ = self.attention(embedded_seq, embedded_seq, embedded_seq)
            embedded_seq = self.attn_norm(embedded_seq + attended_seq)

        # CNN branches (inherited from parent)
        market_matrix = embedded_seq.unsqueeze(1)  # (B, 1, N, 64)
        to_fuse = [cash, holdings]
        if self.use_micro:
            to_fuse.append(self.micro_branch(market_matrix))
        if self.use_macro:
            to_fuse.append(self.macro_branch(market_matrix))
        if not self.use_micro and not self.use_macro:
            to_fuse.append(embedded_seq.view(-1, self.fallback_flat_dim))

        features = self.fusion(torch.cat(to_fuse, dim=1))
        self._cur_value = self.value_branch(features).squeeze(1)
        return features, state


def register_models() -> None:
    """
    Register the custom RLlib model architectures under their canonical names.

    The string names here are the single source of truth for the model registry
    and MUST stay in sync with the ``custom_model`` values in
    ``algo_v2.config.ARCH_MAP`` and the algo-specific overrides. Call this once
    at startup in any process that builds RLlib trainers (training, validation,
    tuning, live execution) before constructing a config that references them.
    """
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model("dual_branch_cnn", DualBranchCNNExtractor)
    ModelCatalog.register_custom_model("indicator_attn", IndicatorAttnExtractor)
    ModelCatalog.register_custom_model("temporal_attn", TemporalAttnExtractor)
