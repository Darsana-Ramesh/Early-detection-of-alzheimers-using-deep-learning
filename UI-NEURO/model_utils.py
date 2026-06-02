import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv, AttentionalAggregation, BatchNorm, GraphSizeNorm

class MultiplexCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = np.sqrt(in_channels)

    def forward(self, p, g, w):
        x = torch.stack([p, g, w], dim=1)
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        fused_out = torch.matmul(attn_weights, v).mean(dim=1)
        return fused_out, attn_weights

class NeuroVistaGNN(torch.nn.Module):
    def __init__(self, node_in=32, vector_in=1024, out_channels=4):
        super().__init__()
        self.gat_p = GATv2Conv(node_in, 64, heads=8, concat=True)
        self.gat_g = GATv2Conv(node_in, 64, heads=8, concat=True)
        self.gat_w = GATv2Conv(node_in, 64, heads=8, concat=True)
        self.cross_modal_fusion = MultiplexCrossAttention(512)
        self.gnorm, self.bn = GraphSizeNorm(), BatchNorm(512)
        self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(512, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(512 + vector_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

    def forward(self, data, return_attn=False):
        p_x, (p_edge, p_alpha) = self.gat_p(data.x, data.edge_index, return_attention_weights=True)
        g_x, (g_edge, g_alpha) = self.gat_g(data.x, data.edge_granger, return_attention_weights=True)
        w_x, (w_edge, w_alpha) = self.gat_w(data.x, data.edge_wavelet, return_attention_weights=True)

        p, g, w = F.elu(p_x), F.elu(g_x), F.elu(w_x)
        f, modality_weights = self.cross_modal_fusion(p, g, w)
        f = self.bn(self.gnorm(f, data.batch))
        
        logits = self.classifier(torch.cat([self.pool(f, data.batch), data.extra_feat], dim=1))

        if return_attn:
            return logits, {
                "node_attn": p_alpha.mean(dim=1).detach().cpu().numpy(),
                "modality_weights": modality_weights.mean(dim=0).detach().cpu().numpy(),
                "edge_index": p_edge.detach().cpu().numpy()
            }
        return logits