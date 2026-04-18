import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"
    
class CosineCutoff(nn.Module):
    """
    Cosine cutoff function for smooth edge weighting
    """
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
        
    def forward(self, distances):
        """
        Args:
            distances: [E] - edge lengths
        Returns:
            [E] - cutoff weights in [0, 1]
        """
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).to(cutoffs.dtype)
        return cutoffs
    
class ExpNormalSmearing(nn.Module):
    """
    Exponential normal smearing with cosine cutoff
    More expressive than Gaussian smearing with trainable parameters
    """
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        """Initialize means and betas in exponential space"""
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def forward(self, distances):
        """
        Args:
            distances: [E] - edge lengths
        Returns:
            [E, num_rbf] - radial basis functions
        """
        distances = distances.unsqueeze(-1)
        
        # Exponential transformation
        exp_dist = torch.exp(-self.alpha * distances)
        
        # RBF with trainable means and betas
        rbf = torch.exp(-self.betas * (exp_dist - self.means) ** 2)
        
        # Apply smooth cutoff
        cutoff_weights = self.cutoff_fn(distances.squeeze(-1)).unsqueeze(-1)
        
        return rbf * cutoff_weights
    
class GaussianSmearing(nn.Module):
    """
    Gaussian smearing for edge lengths (legacy, kept for compatibility)
    """
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        
        self.width = (stop - start) / (num_gaussians - 1)
    
    def forward(self, distances):
        """
        Args:
            distances: [E] - edge lengths
        Returns:
            [E, num_gaussians]
        """
        distances = distances.unsqueeze(-1)
        return torch.exp(-0.5 * ((distances - self.offset) / self.width) ** 2)

class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )

class RadialMLP(torch.nn.Module):
    """
    Construct a radial MLP (Linear → LayerNorm → SiLU) stack
    given a list of channel sizes, following ESEN / FairChem.
    """

    def __init__(self, channels_list) -> None:
        super().__init__()

        modules = []
        in_channels = channels_list[0]

        for idx, out_channels in enumerate(channels_list[1:], start=1):
            modules.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            in_channels = out_channels
            if idx < len(channels_list) - 1:
                modules.append(torch.nn.LayerNorm(out_channels))
                modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)
        self.hs = channels_list

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)