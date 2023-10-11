import torch.nn as nn
import numpy as np
import torch
from typing import Tuple, List, Union, Callable, Type, Iterator
from enum import Enum, unique
import torch.optim.optimizer

class MultiMeshConv(nn.Module):

    def __init__(self, number_features: Union[Tuple[int, ...], List[int]]):
        super(MultiMeshConv, self).__init__()
        layers = [
          nn.Sequential(*
                        [SingleMeshConv(number_features[i], number_features[i + 1], i==0)] +
                        ([nn.InstanceNorm1d(number_features[i + 1])]) +
                        [nn.LeakyReLU(0.2, inplace=True)]
          ) for i in range(len(number_features) - 2)
        ] + [SingleMeshConv(number_features[-2], number_features[-1], False)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, gfmm):
        for layer in self.layers:
            x = layer((x, gfmm))
        return x

class SingleMeshConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, is_first):
        super(SingleMeshConv, self).__init__()
        self.first = is_first
        if is_first:
            self.conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.conv = nn.Conv1d(in_channels * 2, out_channels, 1)

    def forward(self, mesh) -> torch.Tensor:
        # TODO: Need to replace gfmm with mesh indexing of face neighbors
        x, gfmm = mesh
        n_faces = x.shape[-1]  # 1, in_fe, f
        if not self.first:
            x_a = x[:, :, gfmm]
            x_b = x.view(1, -1, 1, n_faces).expand_as(x_a)
            x = torch.cat((x_a, x_b), 1)
        else:
            x = x.view(1, 3, -1, n_faces).permute(0, 2, 1, 3)
        x = x.reshape(1, -1, n_faces * 3)
        x = self.conv(x)
        x = x.view(1, -1, 3, n_faces)
        x = x.max(2)[0]
        return x