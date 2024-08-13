import timm
model = timm.create_model("xception", pretrained=True)
import torch.nn as nn
import torch.optim as optim
import torch

original_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias
)

with torch.no_grad():
    model.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)