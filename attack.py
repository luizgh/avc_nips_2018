import torch
import numpy as np
from typing import Callable, Iterable, Optional


def attack(model: Callable,
           s_models: Iterable[torch.nn.Module],
           attacks,
           image: np.ndarray,
           label: int,
           targeted: bool,
           device: Optional[torch.device]):
    """ Run a black-box attack on 'model', using surrogate models 's_models'
        and attackers 'attacks'

        For each surrogate model and attacker, we create an attack against the
        surrogate model, and use the resulting direction to create an attack
        against the defense ('model'). This is done with a binary search along
        this direction.

    Parameters
    ----------
    model : Callable
        The model under attack. Should be a function that takes an image
        x (H x W x C) with pixels from [0, 255] and returns the label (int)
    s_models : List of torch models
        The surrogate models. Each model should be a PyTorch nn.Module, that
        takes an input x (B x H x W x C) with pixels from [0, 1], and returns
        the pre-softmax activations (logits).
    attacks : List of attack functions
        List of attacks. Each attack should have a method as follows:
            attack(model, inputs, labels, targeted) -> adv_image
    image : np.ndarray
        An image (H x W x C) with pixels ranging from [0, 255]
    label : int
        The true label (if targeted=True) or target label (if targeted=False)
    targeted : bool
        Wheter to run untargeted or a targeted attack
    device: torch.device
        Which device to use for the attacks

    Returns
    -------
    np.ndarray:
        The best adversarial image found against 'model'. None if no
        adversarial is found.

    """
    adversarial = None
    best_norm = np.linalg.norm(np.maximum(255 - image, image))

    original_label = model(image)
    if not targeted and original_label != label:
        # Image is already adversarial
        return image
    if targeted and original_label == label:
        # Image is already adversarial
        return image

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_image = torch.tensor(image).float().div(255).permute(2, 0, 1)
    t_image = t_image.unsqueeze(0).to(device)
    t_label = torch.tensor(label, device=device).unsqueeze(0)

    for s_m in s_models:
        for attack in attacks:
            adv_img = attack.attack(s_m, t_image, t_label, targeted).squeeze(0)
            delta = adv_img.permute(1, 2, 0).cpu().numpy() * 255 - image
            delta = np.round(delta)
            norm = np.linalg.norm(delta)

            if norm > 0:
                # Run bound search
                lower, upper, found = bound_search(model, image,
                                                   label, delta,
                                                   targeted=targeted)
                if found:
                    norm = np.linalg.norm(upper)
                    if norm < best_norm:
                        adversarial = upper + image
                        best_norm = norm

                # Run binary search
                upper_, found_ = binary_search(model, image, label,
                                               lower, upper, steps=10,
                                               targeted=targeted)

                if found_:
                    norm = np.linalg.norm(upper_)
                    if norm < best_norm:
                        adversarial = upper_ + image
                        best_norm = norm


    return adversarial


def bound_search(model, image, label, delta, alpha=1, iters=9, targeted=False):
    """ Coarse search for the decision boundary in direction delta """
    def out_of_region(delta):
        # Returns whether or not image+delta is outside the desired region
        # (e.g. inside the class boundary for untargeted, outside the target
        # class for targeted)
        if targeted:
            return model(image + delta) != label
        else:
            return model(image + delta) == label

    if out_of_region(delta):
        # increase the noise
        lower = delta
        upper = np.clip(image + np.round(delta * (1 + alpha)), 0, 255) - image

        for _ in range(iters):
            if out_of_region(upper):
                lower = upper
                adv = image + np.round(upper * (1 + alpha))
                upper = np.clip(adv, 0, 255) - image
            else:
                return lower, upper, True
    else:
        # inside the region of interest. Decrease the noise
        upper = delta
        lower = np.clip(image + np.round(delta / (1 + alpha)), 0, 255) - image

        for _ in range(iters):
            if not out_of_region(lower):
                upper = lower
                adv = image + np.round(lower / (1 + alpha))
                lower = np.clip(adv, 0, 255) - image
            else:
                return lower, upper, True

    return np.zeros_like(delta), np.maximum(1 - image, image), False


def binary_search(model, image, label, lower, upper, steps=10, targeted=False):
    """ Binary search for the decision boundary in direction delta """
    def out_of_region(delta):
        # returns whether or not image+delta is outside the desired region
        # (e.g. inside the class boundary for untargeted, outside the target
        # class for targeted)
        if targeted:
            return model(image + delta) != label
        else:
            return model(image + delta) == label

    found = False
    for _ in range(steps):
        middle = np.round((lower + upper) / 2)
        middle = np.clip(image + middle, 0, 255) - image
        if out_of_region(middle):
            lower = middle
        else:
            upper = middle
            found = True

    return upper, found
