import torch
import argparse
from fast_adv.attacks import DDN
from fgm_l2 import FGM_L2
from attack import attack
from utils import NormalizedModel
from models import resnet18, resnext50_32x4d
from scipy.misc import imread, imsave
import numpy as np


parser = argparse.ArgumentParser('Attack example')
parser.add_argument('--model-path', '--m', required=True)
parser.add_argument('--surrogate-model-path', '--sm', required=True)

args = parser.parse_args()

image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)

# Load image
img = imread('data/img.png')
t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)
label = 97  # sandal

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model under attack:
m = resnet18()
model = NormalizedModel(m, image_mean, image_std)
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)
model.eval().to(device)


# Simulate a black-box model (i.e. returns only predictions, no gradient):
def black_box_model(img):
    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
    t_img = t_img.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t_img).argmax()

# Sanity check: image correctly labeled:
t_img = t_img.to(device)
assert model(t_img).argmax() == label
assert black_box_model(img) == label


# Load surrogate model
smodel = resnext50_32x4d()
smodel = NormalizedModel(smodel, image_mean, image_std)
state_dict = torch.load(args.surrogate_model_path)
smodel.load_state_dict(state_dict)
smodel.eval().to(device)

# Sanity check: image correctly labeled by surrogate classifier:
assert smodel(t_img).argmax() == label

surrogate_models = [smodel]
attacks = [
    DDN(100, device=device),
    FGM_L2(1)
]

adv = attack(black_box_model, surrogate_models, attacks,
             img, label, targeted=False, device=device)

pred_on_adv = black_box_model(adv)

print('True label: {}; Prediction on the adversarial: {}'.format(label,
                                                                 pred_on_adv))

# Compute l2 norm in range [0, 1]
l2_norm = np.linalg.norm(((adv - img) / 255))
print('L2 norm of the attack: {:.4f}'.format(l2_norm))
print('Saving adversarial image to "data/adv.png"')

imsave('data/adv.png', adv)
