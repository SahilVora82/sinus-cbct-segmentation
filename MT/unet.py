import torch
import torch.nn as nn

from typing import List


# https://github.com/ludvb/batchrenorm
class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.running_mean: torch.Tensor
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.running_std: torch.Tensor
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.num_batches_tracked: torch.Tensor
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0) 
                batch_std = z.std(0, unbiased=False) + self.eps
            else:
                batch_mean = x.mean(dims)
                batch_std = x.std(dims, unbiased=False) + self.eps

            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            ).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x

class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")


class UNet(nn.Module):
    @staticmethod
    def DoubleConv3d(input_channels, inter_channels, output_channels, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv3d(input_channels, inter_channels, kernel_size, padding=padding, bias=False),
            BatchRenorm3d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(inter_channels, output_channels, kernel_size, padding=padding, bias=False),
            BatchRenorm3d(output_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    @staticmethod
    def ConvTranspose3d(input_channels, output_channels, kernel_size, stride):
        return nn.ConvTranspose3d(input_channels, output_channels, kernel_size, stride)
    
    # Example channels: [48, 64, 128, 196]
    def __init__(self, channels: List[int]):
        super().__init__()
        
        input_channels = 1
        output_channels = 3
        
        self.conv_down_1 = self.DoubleConv3d(input_channels, channels[0], channels[0], 5, 2)
        self.pool_1 = nn.MaxPool3d(2)
        self.conv_down_2 = self.DoubleConv3d(channels[0], channels[1], channels[1], 3, 1)
        self.pool_2 = nn.MaxPool3d(2)
        self.conv_down_3 = self.DoubleConv3d(channels[1], channels[1], channels[2], 3, 1)
        self.pool_3 = nn.MaxPool3d(2)

        self.conv_down_4 = self.DoubleConv3d(channels[2], channels[2], channels[3], 3, 1)
        
        self.conv_transpose_1 = self.ConvTranspose3d(channels[3], channels[3], 2, 2)
        self.conv_up_1 = self.DoubleConv3d(channels[2] + channels[3], channels[2], channels[2], 3, 1)
        self.conv_transpose_2 = self.ConvTranspose3d(channels[2], channels[2], 2, 2)
        self.conv_up_2 = self.DoubleConv3d(channels[1] + channels[2], channels[1], channels[1], 3, 1)
        self.conv_transpose_3 = self.ConvTranspose3d(channels[1], channels[1], 2, 2)
        self.conv_up_3 = self.DoubleConv3d(channels[0] + channels[1], channels[0], channels[0], 3, 1)
        
        self.conv_out = nn.Conv3d(channels[0], output_channels, 3, 1, 1)
        
        with torch.no_grad():
            for module in self.modules():
                if(isinstance(module, nn.Linear)):
                    weight = module.weight
                    weight.normal_(0, (1 / weight.size(1)) ** 0.5)
                    if(module.bias != None):
                        module.bias.fill_(0)
                if(isinstance(module, nn.Conv3d)):
                    weight = module.weight
                    weight.normal_(0, (2 / (weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4)))**0.5)
                    if(module.bias != None):
                        module.bias.fill_(0)
                if(isinstance(module, nn.ConvTranspose3d)):
                    weight = module.weight
                    weight.normal_(0, (2 / (weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4)))**0.5)
                    if(module.bias != None):
                        module.bias.fill_(0)
    
    def forward(self, x):
        x_conv_down_1 = self.conv_down_1(x) # 224 -> 224
        x_conv_down_2 = self.conv_down_2(self.pool_1(x_conv_down_1)) # 224 -> 112
        x_conv_down_3 = self.conv_down_3(self.pool_2(x_conv_down_2)) # 112 -> 56
        
        x_conv_transpose_1 = self.conv_transpose_1(self.conv_down_4(self.pool_3(x_conv_down_3))) # 56 -> 28 -> 56
        x_conv_transpose_2 = self.conv_transpose_2(self.conv_up_1(torch.cat([x_conv_down_3, x_conv_transpose_1], dim=1))) # 56 -> 112
        x_conv_transpose_3 = self.conv_transpose_3(self.conv_up_2(torch.cat([x_conv_down_2, x_conv_transpose_2], dim=1))) # 112 -> 224
        return self.conv_out(self.conv_up_3(torch.cat([x_conv_down_1, x_conv_transpose_3], dim=1))) # 224 -> 224