import torch


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_freqs,
        max_freq_log2,
        include_input=True,
        input_dims=3,
        log_sampling=True,
        periodic_fns=(torch.sin, torch.cos),
        **_,
    ):
        super().__init__()
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.input_dims
        out_dim = d if self.include_input else 0
        out_dim += d * len(self.periodic_fns) * self.num_freqs
        self.out_dim = out_dim

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs
        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def output_dim(self):
        return self.out_dim

    def forward(self, inputs):
        parts = [inputs] if self.include_input else []
        assert isinstance(self.freq_bands, torch.Tensor)
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                parts.append(p_fn(inputs * freq))
        return torch.cat(parts, -1)


class ColorOpacityNet(torch.nn.Module):
    """Normalized scalar [0,1] → R,G,B,A [0,1]."""

    def __init__(
        self, n_hidden_neurons: int = 64, embedf: int = 6, n_hidden_layers=4
    ) -> None:
        super().__init__()
        # Embedder
        self.embedder = Embedder(embedf, embedf - 1, input_dims=1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.embedder.output_dim(), n_hidden_neurons),
            torch.nn.LayerNorm(n_hidden_neurons),
            torch.nn.ReLU(inplace=True),
            *[
                layer
                for _ in range(n_hidden_layers)
                for layer in [
                    torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
                    torch.nn.LayerNorm(n_hidden_neurons),
                    torch.nn.ReLU(inplace=True),
                ]
            ],
            torch.nn.Linear(n_hidden_neurons, 4),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embedder(x))


class GradientOpacityNet(torch.nn.Module):
    """Normalized gradient magnitude [0,1] → A [0,1]."""

    def __init__(
        self, n_hidden_neurons: int = 64, embedf: int = 6, n_hidden_layers=4
    ) -> None:
        super().__init__()
        self.embedder = Embedder(embedf, embedf - 1, input_dims=1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.embedder.output_dim(), n_hidden_neurons),
            torch.nn.LayerNorm(n_hidden_neurons),
            torch.nn.ReLU(inplace=True),
            *[
                layer
                for _ in range(n_hidden_layers)
                for layer in [
                    torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
                    torch.nn.LayerNorm(n_hidden_neurons),
                    torch.nn.ReLU(inplace=True),
                ]
            ],
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embedder(x))


class TransferFunctionNet(torch.nn.Module):
    """(scalar_norm, grad_norm) → R,G,B,A where A = A_color * A_gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.color_opacity_net = ColorOpacityNet()
        self.gradient_opacity_net = GradientOpacityNet()

    def forward(self, scalars_and_gradients: torch.Tensor) -> torch.Tensor:
        # scalars_and_gradients: [N, H*W, 2]
        rgba = self.color_opacity_net(scalars_and_gradients[..., 0:1])  # [N, H*W, 4]
        grad_a = self.gradient_opacity_net(
            scalars_and_gradients[..., 1:2]
        )  # [N, H*W, 1]
        return torch.cat([rgba, grad_a], dim=-1)  # [N, H*W, 5]
