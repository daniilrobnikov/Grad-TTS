import torch
import torch.nn as nn
import torch.nn.functional as F
from model.scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
)


class ResUnit(nn.Module):
    def __init__(self, dim, dim_out):
        super(ResUnit, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(nn.Conv2d(dim, dim_out, 3, padding=1))
        self.layer.append(nn.BatchNorm2d(dim_out))
        self.layer.append(ActivationBalancer(dim_out, channel_dim=1))
        self.layer.append(DoubleSwish())

    def forward(self, x):
        output = x
        for layer in self.layer:
            output = layer(output)
        return output


class TrunkBranch(nn.Module):
    def __init__(self, dim, dim_out, t=2):
        super(TrunkBranch, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(ResUnit(dim, dim_out))
        for _ in range(t - 1):
            self.layer.append(ResUnit(dim_out, dim_out))

    def forward(self, x, mask):
        output = x * mask
        for layer in self.layer:
            output = layer(output)
            output = output * mask
        return output


class MaskBranch(nn.Module):
    def __init__(self, dim, dim_out, r=1):
        super(MaskBranch, self).__init__()

        self.pre_layer = nn.ModuleList()
        self.pre_layer.append(nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False))
        self.pre_layer.append(ResUnit(dim, dim_out))
        for _ in range(r - 1):
            self.layer.append(ResUnit(dim_out, dim_out))

        self.layer = nn.ModuleList()
        self.layer.append(nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False))
        for _ in range(2 * r):
            self.layer.append(ResUnit(dim_out, dim_out))

        self.post_layer = nn.ModuleList()
        for _ in range(r):
            self.post_layer.append(ResUnit(dim_out, dim_out))

        self.conv1 = nn.Conv2d(dim_out, dim_out, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask):
        output = x * mask
        for layer in self.pre_layer:
            output = layer(output)

        for layer in self.layer:
            output = layer(output)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

        for layer in self.post_layer:
            output = layer(output)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.softmax(output)
        return output * mask


class AttentionModule(nn.Module):
    def __init__(self, dim, dim_out, p=1, t=2, r=1):
        super(AttentionModule, self).__init__()

        self.input_units = nn.ModuleList()
        for _ in range(p):
            self.input_units.append(ResUnit(dim, dim))

        self.trunk_branch = TrunkBranch(dim, dim_out, t)
        self.mask_branch = MaskBranch(dim, dim_out, r)

        self.output_units = nn.ModuleList()
        for _ in range(p):
            self.output_units.append(ResUnit(dim_out, dim_out))

    def forward(self, x, mask):
        # Input units
        for layer in self.input_units:
            x = layer(x * mask)

        # Trunk branch
        output_trunk = self.trunk_branch(x, mask)
        # Mask branch
        output_mask = self.mask_branch(x, mask)
        # Combine both
        output = (1 + output_mask) * output_trunk

        # Output units
        for layer in self.input_units:
            x = layer(x * mask)
        return output * mask


class RANBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim):
        super(RANBlock, self).__init__()

        self.attention_module = AttentionModule(dim, dim_out)
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.attention_module(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        output = h + self.res_conv(x * mask)
        return output * mask
