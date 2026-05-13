"""
SSL Frontend for voice anti-spoofing.
Provides WavLM/XLS-R/HuBERT feature extraction with trainable
weighted layer aggregation + MockSSLFrontend for CPU smoke tests.
"""
import torch
import torch.nn as nn


class WeightedLayerSum(nn.Module):
    """Learnable weighted sum over transformer layer outputs."""
    def __init__(self, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)

    def forward(self, hidden_states: list) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)  # (n_layers, B, T, D)
        normed_weights = torch.softmax(self.weights, dim=0)
        normed_weights = normed_weights.view(-1, 1, 1, 1)
        return (stacked * normed_weights).sum(dim=0)  # (B, T, D)


class SSLFrontend(nn.Module):
    """Self-supervised learning frontend using HuggingFace models.
    
    Args:
        ssl_model_name: HuggingFace model identifier
        output_dim: Output feature dimension
        freeze_mode: 'full' | 'partial' | 'none'
        unfreeze_top_n: Layers to unfreeze in 'partial' mode
        ssl_cache_dir: Local cache for pre-staged models (Lanta)
    """
    MODEL_INFO = {
        "microsoft/wavlm-large": {"dim": 1024, "layers": 24},
        "microsoft/wavlm-base": {"dim": 768, "layers": 12},
        "facebook/wav2vec2-xls-r-300m": {"dim": 1024, "layers": 24},
        "facebook/hubert-large-ll60k": {"dim": 1024, "layers": 24},
        "facebook/hubert-base-ls960": {"dim": 768, "layers": 12},
    }

    def __init__(self, ssl_model_name="microsoft/wavlm-large",
                 output_dim=64, freeze_mode="full", unfreeze_top_n=4,
                 ssl_cache_dir=None):
        super().__init__()
        self.ssl_model_name = ssl_model_name
        self.freeze_mode = freeze_mode
        self.unfreeze_top_n = unfreeze_top_n

        info = self.MODEL_INFO.get(ssl_model_name, {"dim": 1024, "layers": 24})
        self.ssl_dim = info["dim"]
        self.n_layers = info["layers"]

        self.ssl_model = self._load_ssl_model(ssl_model_name, ssl_cache_dir)
        self.layer_sum = WeightedLayerSum(self.n_layers + 1)
        self.output_proj = nn.Linear(self.ssl_dim, output_dim)
        self._apply_freeze()

    def _load_ssl_model(self, model_name, cache_dir=None):
        from transformers import AutoModel
        kwargs = {"output_hidden_states": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        return AutoModel.from_pretrained(model_name, **kwargs)

    def _apply_freeze(self):
        if self.freeze_mode == "full":
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        elif self.freeze_mode == "partial":
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            if hasattr(self.ssl_model, "encoder") and \
               hasattr(self.ssl_model.encoder, "layers"):
                layers = self.ssl_model.encoder.layers
                for layer in layers[-self.unfreeze_top_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        # 'none' = all trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ssl_out = self.ssl_model(x, output_hidden_states=True)
        hidden_states = ssl_out.hidden_states
        aggregated = self.layer_sum(list(hidden_states))
        return self.output_proj(aggregated)

    def get_output_dim(self):
        return self.output_proj.out_features

    def get_downsample_factor(self):
        return 320


class MockSSLFrontend(nn.Module):
    """Lightweight mock for CPU smoke testing. No model download needed."""
    def __init__(self, input_samples=64600, output_dim=64, mock_dim=64,
                 downsample_factor=320):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.output_dim = output_dim
        self.conv = nn.Conv1d(1, mock_dim, kernel_size=downsample_factor,
                              stride=downsample_factor)
        self.proj = nn.Linear(mock_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.unsqueeze(1))
        z = z.transpose(1, 2)
        z = self.norm(self.proj(z))
        return z

    def get_output_dim(self):
        return self.output_dim

    def get_downsample_factor(self):
        return self.downsample_factor
