from .layers import *
from math import floor

__all__ = ['ResNeXt', 'resnext29', 'resnext29_1x64d', 'resnext29_32x4d',
           'resnext38', 'resnext38_1x64d', 'resnext38_32x4d',
           'resnext50', 'resnext50_32x4d',
           'resnext101', 'resnext101_32x4d', 'resnext101_64x4d']

model_locations = {
    'resnext50_32x4d': 'resnext50_32x4d_imagenet.pth',
    # https://storage.googleapis.com/luizgh-datasets/avc_models/resnext50_32x4d_imagenet.pth
}


class Bottleneck(nn.Module):
    '''Grouped convolution block.'''
    expansion = 4

    def __init__(self, in_planes, planes, cardinality=32, bottleneck_width=4, stride=1, norm_layer=nn.BatchNorm2d,
                 num_groups=None, activ=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        group_width = cardinality * floor(planes * bottleneck_width / 64)

        self.convs = nn.Sequential(
            conv(in_planes, group_width, kernel_size=1, bias=False,
                 norm_layer=norm_layer, num_groups=num_groups, activ=activ),
            conv(group_width, group_width, kernel_size=3, stride=stride, bias=False, groups=cardinality,
                 norm_layer=norm_layer, num_groups=num_groups, activ=activ),
            conv(group_width, planes * self.expansion, kernel_size=1, bias=False,
                 norm_layer=norm_layer, num_groups=num_groups, activ=None)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                     norm_layer=norm_layer, num_groups=num_groups, activ=None)
            )

        self.activ = activ(1) if activ == nn.PReLU else activ

    def forward(self, x):
        out = self.convs(x)
        out += self.shortcut(x)
        out = self.activ(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=1000,
                 norm_layer=nn.BatchNorm2d, num_groups=None, activ=nn.ReLU(inplace=True)):
        super(ResNeXt, self).__init__()

        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.init = conv(3, self.in_planes, kernel_size=7, stride=1, padding=3, bias=False,
                         norm_layer=norm_layer, num_groups=num_groups, activ=activ)

        self.blocks = nn.Sequential(
            self._make_layer(64, num_blocks[0], 1, norm_layer=norm_layer, num_groups=num_groups),
            self._make_layer(128, num_blocks[1], 2, norm_layer=norm_layer, num_groups=num_groups),
            self._make_layer(256, num_blocks[2], 2, norm_layer=norm_layer, num_groups=num_groups),
            self._make_layer(512, num_blocks[3], 2, norm_layer=norm_layer, num_groups=num_groups)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_planes, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride, norm_layer=nn.BatchNorm2d, num_groups=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, self.cardinality, self.bottleneck_width, stride,
                                     norm_layer=norm_layer, num_groups=num_groups))
            self.in_planes = Bottleneck.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


_class_mapper = {
    0: 285, 1: 758, 2: 890, 3: 765, 4: 951, 5: 30, 6: 430, 7: 972, 8: 967, 9: 731, 10: 704, 11: 235, 12: 532, 13: 323,
    14: 294, 15: 779, 16: 963, 17: 338, 18: 879, 19: 687, 20: 683, 21: 645, 22: 1, 23: 964, 24: 604, 25: 978, 26: 508,
    27: 354, 28: 928, 29: 677, 30: 811, 31: 474, 32: 372, 33: 113, 34: 973, 35: 146, 36: 815, 37: 329, 38: 414, 39: 208,
    40: 932, 41: 145, 42: 325, 43: 76, 44: 387, 45: 737, 46: 954, 47: 806, 48: 314, 49: 744, 50: 440, 51: 315, 52: 565,
    53: 924, 54: 761, 55: 25, 56: 975, 57: 367, 58: 707, 59: 568, 60: 970, 61: 619, 62: 862, 63: 398, 64: 675, 65: 427,
    66: 281, 67: 99, 68: 105, 69: 466, 70: 485, 71: 849, 72: 448, 73: 353, 74: 400, 75: 866, 76: 301, 77: 655, 78: 207,
    79: 873, 80: 471, 81: 678, 82: 808, 83: 570, 84: 470, 85: 526, 86: 567, 87: 309, 88: 525, 89: 123, 90: 734, 91: 605,
    92: 747, 93: 437, 94: 107, 95: 909, 96: 739, 97: 774, 98: 720, 99: 467, 100: 114, 101: 341, 102: 286, 103: 134,
    104: 887, 105: 319, 106: 480, 107: 947, 108: 612, 109: 900, 110: 492, 111: 801, 112: 837, 113: 308, 114: 627,
    115: 128, 116: 149, 117: 817, 118: 899, 119: 839, 120: 345, 121: 929, 122: 877, 123: 496, 124: 462, 125: 71,
    126: 910, 127: 716, 128: 768, 129: 786, 130: 821, 131: 283, 132: 760, 133: 425, 134: 411, 135: 187, 136: 842,
    137: 826, 138: 621, 139: 853, 140: 562, 141: 75, 142: 445, 143: 923, 144: 850, 145: 424, 146: 509, 147: 436,
    148: 781, 149: 950, 150: 557, 151: 122, 152: 874, 153: 542, 154: 543, 155: 458, 156: 457, 157: 511, 158: 349,
    159: 365, 160: 50, 161: 79, 162: 845, 163: 573, 164: 109, 165: 115, 166: 500, 167: 935, 168: 888, 169: 652,
    170: 957, 171: 488, 172: 614, 173: 917, 174: 69, 175: 347, 176: 733, 177: 61, 178: 735, 179: 435, 180: 311,
    181: 313, 182: 151, 183: 32, 184: 291, 185: 406, 186: 682, 187: 438, 188: 945, 189: 421, 190: 463, 191: 635,
    192: 962, 193: 576, 194: 267, 195: 988, 196: 625, 197: 447, 198: 938, 199: 386
}


def transfer_fc(source):
    """Transfer weights and bias from source FCL trained on imagenet to a tinyimgenet FCL"""
    fc = nn.Linear(source.weight.size(1), 200)
    for i in _class_mapper.keys():
        fc.weight[i].copy_(source.weight[_class_mapper[i]])
        fc.bias[i].copy_(source.bias[_class_mapper[i]])
    fc.weight.detach_().requires_grad_(True)
    fc.bias.detach_().requires_grad_(True)
    return fc


def resnext29(cardinality, basewidth, **kwargs):
    """Constructs a ResNeXt-20 model."""
    return ResNeXt([2, 2, 2, 2], cardinality, basewidth, num_classes=200, **kwargs)


def resnext29_1x64d(**kwargs):
    return resnext29(1, 64, **kwargs)


def resnext29_32x4d(**kwargs):
    return resnext29(32, 4, **kwargs)


def resnext38(cardinality, basewidth, **kwargs):
    """Constructs a ResNeXt-38 model."""
    return ResNeXt([3, 3, 3, 3], cardinality, basewidth, num_classes=200, **kwargs)


def resnext38_1x64d(**kwargs):
    return resnext38(1, 64, **kwargs)


def resnext38_32x4d(**kwargs):
    return resnext38(32, 4, **kwargs)


def resnext50(cardinality, basewidth, **kwargs):
    """Constructs a ResNeXt-50 model."""
    return ResNeXt([3, 4, 6, 3], cardinality, basewidth, **kwargs)


def resnext50_32x4d(pretrained=False, **kwargs):
    model = resnext50(32, 4, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_locations['resnext50_32x4d']))
    fc = transfer_fc(model.fc)
    model.fc = fc
    return model


def resnext101(cardinality, basewidth, **kwargs):
    """Constructs a ResNeXt-101 model."""
    return ResNeXt([3, 4, 23, 3], cardinality, basewidth, **kwargs)


def resnext101_32x4d(pretrained=False, **kwargs):
    return resnext101(32, 4, **kwargs)


def resnext101_64x4d(pretrained=False, **kwargs):
    return resnext101(64, 4, **kwargs)
