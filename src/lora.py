"""Low-rank adaptation (LoRA) for the linear layers of KryptoNet-40.

Usage:
    from kryptonet import load_kryptonet
    from lora import add_lora, count_trainable
    model = add_lora(load_kryptonet("kryptonet40.pt"), r=8, layers=(1, 2, 3, 4))
    count_trainable(model)    # only the A and B matrices of W_1, ..., W_4 are trained
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LINEAR = (0, 2, 4, 6, 8)  # positions of W_1, ..., W_5 in the nn.Sequential built by kryptonet.py


class LoRALinear(nn.Module):
    """h = W x + b + (alpha / r) B A x, with W and b frozen and only A (r x d_in) and B (d_out x r) trained.

    B starts at zero, so the layer computes exactly what the pretrained layer did until training moves B.
    By default the update is evaluated as (x A^T) B^T, so B A is never formed; merged=True evaluates
    x (W + (alpha / r) B A)^T instead, to compare the cost of the two orders.
    """

    def __init__(self, base, r, alpha=8.0, merged=False):
        super().__init__()
        self.base, self.r, self.scale, self.merged = base, r, alpha / r, merged
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        # as in peft; Hu et al. (2022) describe a Gaussian A
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B starts at zero
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        if self.merged:
            return F.linear(x, self.base.weight + self.scale * self.B @ self.A, self.base.bias)
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


def add_lora(model, r, layers=(1, 2, 3, 4), alpha=8.0, train_head=False, merged=False):
    """A copy of KryptoNet with LoRA on W_l for l in `layers` (numbered 1 to 5 as in the paper), all else frozen."""
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for l in layers:
        model[LINEAR[l - 1]] = LoRALinear(model[LINEAR[l - 1]], r, alpha, merged)
    if train_head:
        for p in model[LINEAR[-1]].parameters():
            p.requires_grad_(True)
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
