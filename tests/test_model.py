import torch

from color_transfer_function_designer.lib.model import (
    ColorOpacityNet,
    Embedder,
    GradientOpacityNet,
    TransferFunctionNet,
)


def test_embedder_default_output_dim_and_forward():
    embedder = Embedder(num_freqs=6, max_freq_log2=5, input_dims=1)
    # include_input (1) + sin/cos (2) * num_freqs (6) * input_dims (1) = 13
    assert embedder.output_dim() == 1 + 2 * 6 * 1
    x = torch.linspace(0.0, 1.0, 8).reshape(-1, 1)
    out = embedder(x)
    assert out.shape == (8, embedder.output_dim())


def test_embedder_linear_sampling_and_no_input():
    # log_sampling=False exercises the else-branch in create_embedding_fn.
    embedder = Embedder(
        num_freqs=4,
        max_freq_log2=3,
        input_dims=1,
        include_input=False,
        log_sampling=False,
    )
    # no include_input: 0 + 2 * 4 * 1 = 8
    assert embedder.output_dim() == 2 * 4 * 1
    out = embedder(torch.zeros(5, 1))
    assert out.shape == (5, embedder.output_dim())


def test_color_opacity_net_forward_shape_and_range():
    net = ColorOpacityNet()
    out = net(torch.rand(16, 1))
    assert out.shape == (16, 4)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_gradient_opacity_net_forward_shape_and_range():
    net = GradientOpacityNet()
    out = net(torch.rand(16, 1))
    assert out.shape == (16, 1)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_transfer_function_net_forward_2d():
    # Flat sample points (N, 2) -> (N, 5), the path used by lut_from_network.
    net = TransferFunctionNet()
    out = net(torch.rand(16, 2))
    assert out.shape == (16, 5)


def test_transfer_function_net_forward_image_batch():
    # Channel-first image batch (N, 2, H, W) -> (N, 5, H, W), the training path
    # whose output feeds the SSIM GaussianBlur.
    net = TransferFunctionNet()
    out = net(torch.rand(2, 2, 7, 5))
    assert out.shape == (2, 5, 7, 5)
