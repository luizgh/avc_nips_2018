import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def bn(ni, init_zero=False):
    m = nn.BatchNorm2d(ni, momentum=0.01)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = bn(planes * self.expansion)  # , init_zero=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = bn(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # return self.fc(self.drop(x))
        return self.fc(x)


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


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    fc = transfer_fc(model.fc)
    model.fc = fc
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    fc = transfer_fc(model.fc)
    model.fc = fc
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    fc = transfer_fc(model.fc)
    model.fc = fc
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    fc = transfer_fc(model.fc)
    model.fc = fc
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model. """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    fc = transfer_fc(model.fc)
    model.fc = fc
    return model
