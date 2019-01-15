import foolbox
import torch

from foolbox import zoo

from models import resnext50_32x4d
from utils import NormalizedModel


def create():
    url = 'https://storage.googleapis.com/luizgh-datasets/avc_models/resnext50_32x4d_ddn.pt'
    weights_path = zoo.fetch_weights(url)
    state_dict = torch.load(weights_path)

    image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)
    m = resnext50_32x4d()
    model = NormalizedModel(model=m, mean=image_mean, std=image_std)
    model.load_state_dict(state_dict)
    model.eval()

    fmodel = foolbox.models.PyTorchModel(model, (0, 1), num_classes=200)

    return fmodel

if __name__ == '__main__':
    import numpy as np
    from scipy.misc import imread

    fmodel = create()

    x = imread('data/img.png').transpose((2, 0, 1)).astype(np.float32) / 255 # label is 97 (sandal)

    logits = fmodel.predictions(x)
    print('Logits', logits)
    print('Predicted label:', logits.argmax())