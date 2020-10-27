import torch
import torch.nn as nn
import torchvision


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, width, fusion_types='channel_add', tln=0):
        super(SpatialBlock, self).__init__()
        # check if the fusion_type parameter is correct
        assert fusion_types in ['channel_add', 'channel_mul']
        # correct, fusion_types is the way of x_t, c_t, m_t fusion
        self.fusion_types = fusion_types

        self.in_channels = in_channels
        self.planes = in_channels // 8
        self.width = width

        # spatial attention / context modeling
        self.conv_pool = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

        # transform
        if tln:
            self.trans = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, self.width, self.width]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        else:
            self.trans = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

    def forward(self, x_t, c_t, m_t):
        if self.fusion_types == 'channel_add':
            x_c_m_t = x_t + c_t + m_t
        else:
            x_c_m_t = x_t * c_t * m_t

        # spatial attention
        avgout = torch.mean(x_c_m_t, dim=1, keepdim=True)
        maxout, _ = torch.max(x_c_m_t, dim=1, keepdim=True)
        avg_max = torch.cat([avgout, maxout], dim=1)
        context_mask = self.conv_pool(avg_max)
        context_mask = self.sigmoid(context_mask)
        context = context_mask * x_c_m_t

        # transform
        mask = self.trans(context)
        return mask

if __name__ == '__main__':
    model = SpatialBlock(16)
    print(model)

    x = torch.randn(1, 16, 64, 64)
    out = model(x, x, x)
    print(out.shape)
