import torch
import torch.nn as nn


class LabelEmbedder(nn.Module):
    """
    Label Embedding for the Diffusion Models
    It can be used as Classifier-Free Guidance(CFG)
    """

    def __init__(self, num_classes: int, embed_dim: int, drop_prob: float = 0.0):
        super().__init__()

        self.num_classes = num_classes  # Use num_classes as NULL label for CFG
        self.embed_dim = embed_dim
        self.drop_prob = drop_prob
        self.use_cfg = drop_prob > 0.0

        self.embedding_table = nn.Embedding(num_classes + int(self.use_cfg), embed_dim)

    def drop_labels(self, labels: torch.Tensor):
        if not self.use_cfg:
            return labels

        mask = torch.rand(labels.shape, device=labels.device) < self.drop_prob
        dropped_labels = labels.masked_fill(mask, self.num_classes)
        return dropped_labels

    def forward(self, labels: torch.Tensor, is_training: bool = True):
        if is_training and self.use_cfg:
            labels = self.drop_labels(labels)

        embeddings = self.embedding_table(labels)

        return embeddings


class TimeEmbedder(nn.Module):
    def __init__(self, freq_embed_dim: int, embed_dim: int):
        super().__init__()

        self.freq_embed_dim = freq_embed_dim
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def get_sinusoidal_time_embeddings(timesteps: torch.Tensor, embedding_dim: int, base: float = 10000.0):
        half_dim = embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * (
            torch.log(torch.tensor(base)) / half_dim
        )
        sinusoidal_input = timesteps[:, None].float() * torch.exp(exponent[None, :])
        embeddings = torch.cat([torch.sin(sinusoidal_input), torch.cos(sinusoidal_input)], dim=-1)

        if embedding_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)

        return embeddings

    def forward(self, timesteps: torch.Tensor, base: float = 10000.0):
        freq_embeddings = self.get_sinusoidal_time_embeddings(timesteps, self.freq_embed_dim, base)
        time_embeddings = self.mlp(freq_embeddings)

        return time_embeddings
