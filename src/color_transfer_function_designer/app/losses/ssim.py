import torch
import torchvision.transforms as T

_VARIANCE_FLOOR = 1e-8


def _signed_pow(base: torch.Tensor, exponent: float) -> torch.Tensor:
    return base.sign() * base.abs().clamp(min=_VARIANCE_FLOOR) ** exponent


def _compute_ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    k1: float,
    k2: float,
    max_channel_value: float,
    gaussian_blur: T.GaussianBlur,
) -> torch.Tensor:
    c1 = (k1 * max_channel_value) ** 2
    c2 = (k2 * max_channel_value) ** 2
    c3 = 0.5 * c2
    mu_x, mu_y = gaussian_blur(x), gaussian_blur(y)
    mu_x_sq, mu_y_sq = mu_x * mu_x, mu_y * mu_y
    sgm_x_sq = (gaussian_blur(x**2) - mu_x_sq).clamp(min=0.0)
    sgm_y_sq = (gaussian_blur(y**2) - mu_y_sq).clamp(min=0.0)
    sgm_x = torch.sqrt(sgm_x_sq + _VARIANCE_FLOOR)
    sgm_y = torch.sqrt(sgm_y_sq + _VARIANCE_FLOOR)
    sgm_xy = gaussian_blur(x * y) - mu_x * mu_y

    luminance = ((2 * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1)) ** alpha
    contrast = ((2 * sgm_x * sgm_y + c2) / (sgm_x_sq + sgm_y_sq + c2)) ** beta
    structure = _signed_pow((sgm_xy + c3) / (sgm_x * sgm_y + c3), gamma)
    return luminance * contrast * structure


class SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        max_channel_value: float = 255.0,
        gaussian_window_size: tuple[int, int] = (11, 11),
        gaussian_sigma: tuple[float, float] = (1.5, 1.5),
    ):
        super().__init__()
        self.gaussian_blur = T.GaussianBlur(
            kernel_size=gaussian_window_size, sigma=gaussian_sigma
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_window_size = gaussian_window_size
        self.k1 = k1
        self.k2 = k2
        self.max_channel_value = max_channel_value

    def forward(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        ssim_map = _compute_ssim_map(
            input,
            output,
            self.alpha,
            self.beta,
            self.gamma,
            self.k1,
            self.k2,
            self.max_channel_value,
            self.gaussian_blur,
        )
        # SSIM close to 1.0 means input is structurally similar to output
        # The objective is to penalize the network for dissimilar outputs so
        # subtract the ssim score from 1.0
        return 1.0 - ssim_map.mean()
