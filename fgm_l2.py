from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGM_L2:
    """
    Fast Gradient Method using L2 distance.
    Parameters
    ==========
    eps : float
        Epsilon to multiply the attack noise
    image_constraints : tuple
        Bounds of the images. Default: (0, 1)
    """

    def __init__(self,
                 eps: float,
                 image_constraints: Tuple[float, float] = (0, 1)) -> None:
        self.eps = eps

        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]

        self.criterion = F.cross_entropy

    def attack(self, model: nn.Module, inputs: torch.Tensor,
               labels: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the given inputs.
        Parameters
        ==========
        model : nn.Module
            Model to attack
        inputs : torch.Tensor
            Batch of images to generate adv for
        labels : torch.Tensor
            True labels in case of untargeted, target in case of targeted
        targeted : bool
            Whether to perform a targeted attack or not
        """
        multiplier = -1 if targeted else 1
        delta = torch.zeros_like(inputs, requires_grad=True)

        logits = model(inputs + delta)
        loss = self.criterion(logits, labels)
        grad = torch.autograd.grad(loss, delta)[0]

        adv = inputs + multiplier * self.eps * grad / (grad.view(grad.size(0), -1).norm(2, 1)).view(-1, 1, 1, 1)
        adv = torch.clamp(adv, self.boxmin, self.boxmax)

        return adv.detach()
