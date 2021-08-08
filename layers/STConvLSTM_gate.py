__author__ = 'shuxin'

import torch
import torch.nn as nn


class AttnBlock(nn.Module):
    def __init__(self, in_channels, height, width, fusion_types='channel_add'):
        super(AttnBlock, self).__init__()
        # check if the fusion_type parameter is correct
        assert fusion_types in ['channel_add', 'channel_mul']
        # correct, fusion_types is the way of x_t, c_t, m_t fusion
        self.fusion_types = fusion_types

        self.in_channels = in_channels
        self.width = width
        self.height = height

        # mix the average & maximum feature
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.LayerNorm([1, self.height, self.width]),
            nn.ReLU(inplace=True)
        )

        # spatial attention / context modeling
        self.linear = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax(dim=1)

        # transform
        self.trans = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

    def forward(self, x_t, c_t, m_t):
        if self.fusion_types == 'channel_add':
            x_c_m_t = x_t + c_t + m_t
        else:
            x_c_m_t = x_t * c_t * m_t

        # choose the average & obvious feature
        avgout = torch.mean(x_c_m_t, dim=1, keepdim=True)
        maxout, _ = torch.max(x_c_m_t, dim=1, keepdim=True)
        avg_max = torch.cat([avgout, maxout], dim=1)
        feature_map = self.conv_pool(avg_max)

        # calculate attention
        query_context = self.linear(feature_map)
        context = self.softmax(query_context) * x_c_m_t
        # transform
        mask = self.trans(context)
        return mask


class STConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width,
                 filter_size, stride, layer_norm, attn_fusion, forget_bias=1.0):
        super(STConvLSTMCell, self).__init__()
        """Initialize the Causal LSTM cell.
        Args:
            in_channel: number of channels for input tensor.
            num_hidden: number of units for output tensor.
            width: the width of image, default width = height
            filter_size: int tuple that's the height and width of the filter.
            stride: the length of step
            layer_norm: whether to apply tensor layer normalization
            attn_fusion: the fusion way of x_t, c_t and m_t, "channel_add" or "channel_mul"
        """

        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.padding = filter_size // 2
        self._forget_bias = forget_bias

        # attention block
        self.attnBlock = AttnBlock(num_hidden, height, width, attn_fusion)

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 6, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 6, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 2, height, width])
            )
        else:
            self.conv_x = nn.Conv2d(in_channel, num_hidden * 6, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

            self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

            self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)
            self.conv_m = nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, height, width))
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, height, width))
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, height, width))

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        i_x, f_x, g_x, o_x, i_x_prime, f_x_prime = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)
        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        m_concat = self.conv_m(m_t)
        i_m, f_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_m_prime = torch.sigmoid(i_x_prime + i_m)
        f_m_prime = torch.sigmoid(f_x_prime + f_m)

        m_mask = self.attnBlock(x_t, c_new, m_t)
        m_new = f_m_prime * m_t + i_m_prime * torch.tanh(m_mask)
        mem = torch.cat((c_new, m_new), 1)
        o_m = self.conv_o(mem)

        o_t = torch.sigmoid(o_x + o_h + o_m + self.Wco * c_new)

        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


if __name__ == '__main__':
    #  Test attention block
    attn = AttnBlock(16, 64, 64)
    print(attn)
    x = torch.randn(1, 16, 64, 64)
    out = attn(x, x, x)
    print(out.shape)

    #  Test STConvLSTMCell
    cell = STConvLSTMCell(in_channel=16, num_hidden=64, height=16, width=16, filter_size=3,
                          stride=1, layer_norm=1, attn_fusion="channel_add")
    print(cell)
    x = torch.randn(1, 16, 64, 64)
    out = STConvLSTMCell(x, x, x, x)
    print(out.shape)
