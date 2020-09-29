import torch
import torch.nn as nn
import math
from mmcv.cnn import constant_init, kaiming_init

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class SCALayer(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1, w_ratio=4, pool='att', fusion='channel_conv1d_add'):
        super(SCALayer, self).__init__()

        assert pool in ['avg', 'att']
        assert fusion in ['channel_add', 'channel_mul', 'channel_conv1d_add', 'channel_conv1d_mul']

        self.in_channel = in_channel
        self.mid_channel = in_channel // w_ratio
        #
        t = int(math.log(in_channel, 2)+b)//gamma
        self.k_size = t if t%2 else t+1
        self.pool = pool
        self.fusion = fusion

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(in_channel, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' == fusion or 'channel_mul' == fusion:
            self.channel_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1),
                nn.LayerNorm([self.mid_channel, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_channel, self.in_channel, kernel_size=1)
            )
            print('\nfusion:', self.fusion)
        elif 'channel_conv1d_add' == fusion or 'channel_conv1d_mul' == fusion:
            self.channel_conv1d = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
            print('\nfusion:', fusion)
        else:
            raise Exception("Error fusion type!")

        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.fusion == 'channel_mul' or self.fusion == 'channel_add':
            last_zero_init(self.channel_conv)


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.fusion == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_conv(context))
            out = x * channel_mul_term
        elif self.fusion == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_conv(context)
            out = x + channel_add_term
        elif self.fusion == 'channel_conv1d_mul':
            # [N, C, 1, 1]
            channel_conv1d_tearm = torch.sigmoid(self.channel_conv1d(context.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
            out = x * channel_conv1d_tearm.expand_as(x)
        else:
            channel_conv1d_tearm = self.channel_conv1d(context.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = x + channel_conv1d_tearm.expand_as(x)

        return out


if __name__ == "__main__":
    x = torch.rand(4,64,128,128)
    scalayer = SCALayer(64, fusion='channel_mul')

    y = scalayer(x)
    print(y.size())