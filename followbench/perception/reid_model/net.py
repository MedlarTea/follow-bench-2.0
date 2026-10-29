"""Lightweight ResNet-style person ReID backbone (ported from follow/).

Forward returns L2-normalised 512-d embeddings when ``reid=True``; otherwise it
runs the classifier head trained on Market-1501 (751 IDs). We always use the
embedding path here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, is_downsample: bool = False) -> None:
        super().__init__()
        self.is_downsample = is_downsample
        stride = 2 if is_downsample else 1
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out),
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.is_downsample = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def _make_layers(c_in: int, c_out: int, repeat_times: int, is_downsample: bool = False) -> nn.Sequential:
    blocks = [_BasicBlock(c_in, c_out, is_downsample=is_downsample)]
    for _ in range(1, repeat_times):
        blocks.append(_BasicBlock(c_out, c_out))
    return nn.Sequential(*blocks)


class ReIDNet(nn.Module):
    """Compact backbone consumed by ``PersonReIDExtractor``."""

    def __init__(self, num_classes: int = 751, reid: bool = True) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = _make_layers(64, 64, 2, False)
        self.layer2 = _make_layers(64, 128, 2, True)
        self.layer3 = _make_layers(128, 256, 2, True)
        self.layer4 = _make_layers(256, 512, 2, True)
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        if self.reid:
            return x.div(x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
        return self.classifier(x)
