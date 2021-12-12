import torch.nn as nn
import math
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from configs.config import config
import random
import os
import numpy as np

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_path = '$root/resnet18_pretrained/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    return model

def upsample_deterministic(x,upscale):
    return x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale).reshape(x.size(0), x.size(1), x.size(2)*upscale, x.size(3)*upscale)

class FeatExtractor(nn.Module):
    def __init__(self, pretrained=config.pretrained):
        super(FeatExtractor, self).__init__()
        model_resnet = resnet18(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        layer1_out = self.layer1(out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)

        layer2_out_up = upsample_deterministic(layer2_out, 2)
        layer3_out_up = upsample_deterministic(layer3_out, 4)

        layer_cat = torch.cat([layer1_out, layer2_out_up, layer3_out_up], dim=1)
        return layer3_out, layer_cat

class FeatEmbedder(nn.Module):
    def __init__(self):
        super(FeatEmbedder, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, input, params=None):
        if params == None:
            out = self.layer4(input)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            feature = self.bottleneck(out)
            cls_out = self.classifier(feature)
        else:
            out = F.conv2d(
                input,
                params['layer4.0.conv1.weight'],
                stride=2,
                padding=1)
            out = F.batch_norm(
                out,
                params['layer4.0.bn1.running_mean'],
                params['layer4.0.bn1.running_var'],
                params['layer4.0.bn1.weight'],
                params['layer4.0.bn1.bias'],
                training=True)
            out = F.relu(out, inplace=True)
            out = F.conv2d(
                out,
                params['layer4.0.conv2.weight'],
                stride=1,
                padding=1)
            out = F.batch_norm(
                out,
                params['layer4.0.bn2.running_mean'],
                params['layer4.0.bn2.running_var'],
                params['layer4.0.bn2.weight'],
                params['layer4.0.bn2.bias'],
                training=True)
            residual = F.conv2d(
                input,
                params['layer4.0.downsample.0.weight'],
                stride=2)
            residual = F.batch_norm(
                residual,
                params['layer4.0.downsample.1.running_mean'],
                params['layer4.0.downsample.1.running_var'],
                params['layer4.0.downsample.1.weight'],
                params['layer4.0.downsample.1.bias'],
                training=True)
            out += residual
            out = F.relu(out, inplace=True)

            residual = out
            out = F.conv2d(
                out,
                params['layer4.1.conv1.weight'],
                stride=1,
                padding=1)
            out = F.batch_norm(
                out,
                params['layer4.1.bn1.running_mean'],
                params['layer4.1.bn1.running_var'],
                params['layer4.1.bn1.weight'],
                params['layer4.1.bn1.bias'],
                training=True)
            out = F.relu(out, inplace=True)
            out = F.conv2d(
                out,
                params['layer4.1.conv2.weight'],
                stride=1,
                padding=1)
            out = F.batch_norm(
                out,
                params['layer4.1.bn2.running_mean'],
                params['layer4.1.bn2.running_var'],
                params['layer4.1.bn2.weight'],
                params['layer4.1.bn2.bias'],
                training=True)
            out += residual
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out = F.linear(out, params['bottleneck.0.weight'], params['bottleneck.0.bias'])
            out = F.relu(out, inplace=True)
            feature = F.dropout(out, 0.5, training=True)
            cls_out = F.linear(feature, params['classifier.weight'], params['classifier.bias'])

        return feature, cls_out

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

class DepthEstmator(nn.Module):
    def __init__(self):
        super(DepthEstmator, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(448, 224),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),

            conv3x3(224, 112),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),

            conv3x3(112, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.depth_cls = nn.Linear(256, 2)

    def forward(self, input, params=None):
        if params == None:
            depth_map = self.conv(input)
            depth_small = F.adaptive_avg_pool2d(depth_map, 16)
            depth_feature = depth_small.view(depth_small.size(0), -1)
            depth_cls_out = self.depth_cls(depth_feature)
        else:
            out = F.conv2d(
                input,
                params['conv.0.weight'],
                padding=1)
            out = F.batch_norm(
                out,
                params['conv.1.running_mean'],
                params['conv.1.running_var'],
                params['conv.1.weight'],
                params['conv.1.bias'],
                training=True)
            out = F.relu(out, inplace=True)
            out = F.conv2d(
                out,
                params['conv.3.weight'],
                padding=1)
            out = F.batch_norm(
                out,
                params['conv.4.running_mean'],
                params['conv.4.running_var'],
                params['conv.4.weight'],
                params['conv.4.bias'],
                training=True)
            out = F.relu(out, inplace=True)
            out = F.conv2d(
                out,
                params['conv.6.weight'],
                padding=1)
            out = F.batch_norm(
                out,
                params['conv.7.running_mean'],
                params['conv.7.running_var'],
                params['conv.7.weight'],
                params['conv.7.bias'],
                training=True)
            depth_map = F.relu(out, inplace=True)
            depth_small = F.adaptive_avg_pool2d(depth_map, 16)
            depth_feature = depth_small.view(depth_small.size(0), -1)
            depth_cls_out = F.linear(depth_feature, params['depth_cls.weight'], params['depth_cls.bias'])
        return depth_map, depth_cls_out

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

class DGFANet(nn.Module):
    def __init__(self):
        super(DGFANet, self).__init__()
        self.feature_generator = FeatExtractor()
        self.feature_embedder = FeatEmbedder()
        self.depth_estimator = DepthEstmator()

    def forward(self, input, parameters_cls=None, parameters_depth=None):
        out, out_cat = self.feature_generator(input)
        feature, cls_out = self.feature_embedder(out, parameters_cls)
        depth_map, depth_cls_out = self.depth_estimator(out_cat, parameters_depth)
        return feature, cls_out, depth_map, depth_cls_out
    
    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

if __name__ == '__main__':
    x = Variable(torch.ones(5, 3, 256, 256))
    model = DGFANet()