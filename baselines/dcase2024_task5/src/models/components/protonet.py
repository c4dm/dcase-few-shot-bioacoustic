import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import logging
from pathlib import Path
import re

# from embedding_propagation import EmbeddingPropagation


def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
        )

    def forward(self, x):
        (num_samples, seq_len, mel_bins) = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)
        x = self.encoder(x)
        x = nn.MaxPool2d(2)(x)

        return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1, with_bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=with_bias,  # TODO here I change the bias
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        with_bias=False,
        non_linearity="leaky_relu",
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, with_bias=with_bias)
        self.bn1 = nn.BatchNorm2d(planes)

        if non_linearity == "leaky_relu":
            self.relu = nn.LeakyReLU(0.1)
        else:
            self.relu = nn.ReLU()

        self.conv2 = conv3x3(planes, planes, with_bias=with_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, with_bias=with_bias)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x
        # import ipdb; ipdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        features,
        block=BasicBlock,
        keep_prob=1.0,
        avg_pool=True,
        drop_rate=0.1,
        linear_drop_rate=0.5,
        dropblock_size=5,
    ):
        drop_rate = features.drop_rate
        with_bias = features.with_bias

        self.inplanes = 1
        super(ResNet, self).__init__()
        self.linear_drop_rate = linear_drop_rate
        self.layer1 = self._make_layer(block, 64, stride=2, features=features)
        self.layer2 = self._make_layer(block, 128, stride=2, features=features)
        self.layer3 = self._make_layer(
            block,
            64,
            stride=2,
            drop_block=True,
            block_size=dropblock_size,
            features=features,
        )
        self.layer4 = self._make_layer(
            block,
            64,
            stride=2,
            drop_block=True,
            block_size=dropblock_size,
            features=features,
        )
        # if avg_pool:
        #     self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.features = features
        self.keep_avg_pool = avg_pool
        # self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.pool_avg = nn.AdaptiveAvgPool2d(
            (
                features.time_max_pool_dim,
                int(features.embedding_dim / (features.time_max_pool_dim * 64)),
            )
        )  # try max pooling
        self.pool_max = nn.AdaptiveMaxPool2d(
            (
                features.time_max_pool_dim,
                int(features.embedding_dim / (features.time_max_pool_dim * 64)),
            )
        )  # try max pooling
        # self.ep = EmbeddingPropagation()

        # dim = int(features.embedding_dim/(4*64)) * 4 * 64

        # self.mapping = nn.Sequential(
        #     nn.Linear(dim, dim*2),
        #     nn.ReLU(),
        #     nn.Dropout(self.linear_drop_rate),
        #     nn.Linear(dim*2, dim*2),
        #     nn.ReLU(),
        #     nn.Dropout(self.linear_drop_rate),
        #     nn.Linear(dim*2, dim),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=features.non_linearity
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""
        state_dict = torch.load(weight_file, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(
                r"(^layer1\.|\.layer1\.|^layer2\.|\.layer2\.|^layer3\.|\.layer3\.|^layer4\.|\.layer4\.)",
                k,
            )
            if m is None:
                continue
            new_k = k[m.start() :]
            new_k = new_k[1:] if new_k[0] == "." else new_k
            weights[new_k] = state_dict[k]
        self.load_state_dict(weights)
        self.eval()
        logging.info(
            f"Using audio embbeding network pretrained weight: {Path(weight_file).name}"
        )
        return self

    def _make_layer(
        self, block, planes, stride=1, drop_block=False, block_size=1, features=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                features.drop_rate,
                drop_block,
                block_size,
                features.with_bias,
                features.non_linearity,
            )
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        (num_samples, seq_len, mel_bins) = x.shape

        # for p in self.layer1.parameters():
        #     p.requires_grad = False
        # self.layer1.eval()
        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.features.layer_4:
            x = self.layer4(x)
        x = self.pool_avg(x)

        x = x.view(x.size(0), -1)
        # x = self.mapping(x)
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tqdm import tqdm

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    conf = OmegaConf.load("/vol/research/dcase2022/project/hhlab/configs/train.yaml")

    model = ResNet(conf.features)
    print(model)
    print(get_n_params(model))
    input = torch.randn((3, 17, 128))
    print(model(input).size())
