"""
AASIST with SSL Frontend.
Replaces SincConv+ResBlock encoder with SSL model (WavLM/XLS-R/HuBERT).
Graph attention backend identical to original AASIST.
Used for ablation S2, S3, S4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AASIST import (
    GraphAttentionLayer,
    HtrgGraphAttentionLayer,
    GraphPool,
)
from models.ssl_frontend import SSLFrontend, MockSSLFrontend


class Model(nn.Module):
    """AASIST with SSL frontend.
    
    Config keys:
        ssl_mode: 'single' | 'mock'
        ssl_model: HuggingFace model name (for 'single')
        ssl_freeze: 'full' | 'partial' | 'none'
        ssl_output_dim: output dim of SSL projection (default: 128)
        ssl_cache_dir: path to pre-staged SSL weights
        mock_ssl_dim: hidden dim for mock mode (default: 64)
    """

    def __init__(self, d_args: dict):
        super().__init__()
        self.d_args = d_args

        ssl_mode = d_args.get("ssl_mode", "single")
        ssl_output_dim = d_args.get("ssl_output_dim", 128)

        # ---- SSL Frontend ----
        if ssl_mode == "mock":
            mock_dim = d_args.get("mock_ssl_dim", 64)
            nb_samp = d_args.get("nb_samp", 64600)
            self.ssl_frontend = MockSSLFrontend(
                input_samples=nb_samp, output_dim=ssl_output_dim,
                mock_dim=mock_dim)
        elif ssl_mode == "single":
            self.ssl_frontend = SSLFrontend(
                ssl_model_name=d_args.get("ssl_model", "microsoft/wavlm-large"),
                output_dim=ssl_output_dim,
                freeze_mode=d_args.get("ssl_freeze", "full"),
                unfreeze_top_n=d_args.get("ssl_unfreeze_top_n", 4),
                ssl_cache_dir=d_args.get("ssl_cache_dir", None))
        else:
            raise ValueError(f"Unknown ssl_mode: {ssl_mode}")

        # ---- Reshape SSL to spectral/temporal views ----
        self.spectral_nodes = d_args.get("spectral_nodes", 23)
        feat_per_node = ssl_output_dim // self.spectral_nodes
        self.feat_per_node = feat_per_node

        gat_dims = d_args["gat_dims"]

        self.proj_S = nn.Linear(feat_per_node, gat_dims[0])
        self.proj_T = nn.Linear(ssl_output_dim, gat_dims[0])

        # ---- Graph Attention (same as AASIST) ----
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.pos_S = nn.Parameter(
            torch.randn(1, self.spectral_nodes, gat_dims[0]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            gat_dims[0], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(
            gat_dims[0], gat_dims[0], temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # ---- Output ----
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: torch.Tensor, Freq_aug: bool = False):
        # ---- SSL Feature Extraction ----
        ssl_out = self.ssl_frontend(x)  # (B, T, D)

        # ---- Spectral Graph Nodes ----
        B, T, D = ssl_out.shape
        e_S_raw, _ = torch.max(ssl_out, dim=1)  # (B, D)
        trim_D = self.spectral_nodes * self.feat_per_node
        e_S_raw = e_S_raw[:, :trim_D]
        e_S = e_S_raw.view(B, self.spectral_nodes, self.feat_per_node)
        e_S = self.proj_S(e_S) + self.pos_S

        # ---- Temporal Graph Nodes ----
        e_T = self.proj_T(ssl_out)  # (B, T, gat_dim)

        # ---- GAT ----
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # ---- Heterogeneous cross-attention (dual branch) ----
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # ---- Dropout + Max fusion ----
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # ---- Readout ----
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output
