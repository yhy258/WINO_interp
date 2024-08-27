import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.types import Device


def complex_to_real(val, device):
    val = torch.tensor([val.real, val.imag]).to(device)
    val = val[None, :, None, None]
    return val

def inverse_minmaxscaler(data, minval, maxval, device=None, complex=True):
    if complex :
        # minval = complex_to_real(minval, device)
        # maxval = complex_to_real(maxval, device)
        minval = minval.to(device)
        maxval = maxval.to(device)
    return data * (maxval - minval) + minval


def relative_error(x, y, p, reduction="mean"):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    diff_norms = torch.linalg.norm(x - y, p, 1)
    y_norms = torch.linalg.norm(y, p, 1)
    if reduction == "mean":
        return torch.mean(diff_norms / y_norms)
    elif reduction == "none":
        return diff_norms / y_norms
    else:
        raise ValueError("Only reductions 'mean' and 'none' supported!")


class NMSE(torch.nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    @torch.jit.ignore
    def forward(self, x, y):
        return relative_error(x, y, self.p, self.reduction)

class NMAE(torch.nn.Module):
    def __init__(self, p=1, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    @torch.jit.ignore
    def forward(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction=='mean':
            return torch.mean(diff_norms / y_norms)
        else:
            return diff_norms
    
class NRSE(torch.nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    @torch.jit.ignore
    def forward(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        return torch.mean(diff_norms / y_norms)

class ComplexMSELoss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Complex MSE between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        # factors = torch.linspace(0.1, 1, x.size(-1), device=x.device).view(1,1,1,-1)
        # return F.mse_loss(torch.view_as_real(x.mul(factors)), torch.view_as_real(target.mul(factors)))
        if self.norm:
            diff = torch.view_as_real(x - target)
            return (
                diff.square()
                .sum(dim=[1, 2, 3, 4])
                .div(torch.view_as_real(target).square().sum(dim=[1, 2, 3, 4]))
                .mean()
            )
        return F.mse_loss(torch.view_as_real(x), torch.view_as_real(target))


class ComplexL1Loss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Complex L1 loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        if self.norm:
            diff = torch.view_as_real(x - target)
            return diff.norm(p=1, dim=[1, 2, 3, 4]).div(torch.view_as_real(target).norm(p=1, dim=[1, 2, 3, 4])).mean()
        return F.l1_loss(torch.view_as_real(x), torch.view_as_real(target))


class SpectralMatchingLoss(torch.nn.Module):
    def __init__(self, norm=False, criterion_mode='mse'):
        super().__init__()
        self.norm = norm
        if criterion_mode == 'mae':
            self.criterion = ComplexL1Loss(norm=norm)
        else:
            self.criterion = ComplexMSELoss(norm=norm)
            
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        # complex로 표현이 가능한가?
        x_fft = torch.fft.fft2(x[:, 0, :, :] + 1j*x[:, 1, :, :], norm='ortho')[:, None, :, :] # BS, H, W
        target_fft = torch.fft.fft2(target[:, 0, :, :] + 1j*target[:, 1, :, :], norm='ortho')[:, None, :, :]
        return self.criterion(x_fft, target_fft)
        

        