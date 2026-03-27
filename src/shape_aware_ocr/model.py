from __future__ import annotations

import torch
import torch.nn as nn


class LPRNetTorch(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int | tuple[int, int] = 1, padding: int = 1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def depthwise_separable(in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1)):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            conv_block(3, 64),
            depthwise_separable(64, 128, stride=(2, 2)),
            depthwise_separable(128, 128),
            depthwise_separable(128, 192, stride=(2, 1)),
            depthwise_separable(192, 192),
            depthwise_separable(192, 256),
            depthwise_separable(256, 256),
        )

        temporal_layers = []
        for dilation in (1, 2, 4, 8, 16):
            temporal_layers.extend(
                [
                    nn.Conv1d(256, 256, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                ]
            )
        self.temporal = nn.Sequential(*temporal_layers)
        self.logits = nn.Conv1d(256, num_classes, kernel_size=1, padding=0, bias=True)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.mean(dim=2)
        x = self.temporal(x)
        x = self.logits(x)
        return x.permute(2, 0, 1).contiguous()


class ShapeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
