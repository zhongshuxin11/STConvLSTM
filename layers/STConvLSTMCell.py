import numpy as np
import torch
import torch.nn as nn
from layers.SpatialBlock import SpatialBlock


# c is convolution
class STLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm,
                 sp_fusion, sp_tln, forget_bias=1.0):
        super(STLSTMCell, self).__init__()
        """Initialize the Causal LSTM cell.
        Args:
            in_channel: number of channels for input tensor.
            num_hidden: number of units for output tensor.
            width: the width of image, default width = height
            filter_size: int tuple that's the height and width of the filter.
            stride: the length of step
            forget_bias: float, The bias added to forget gates.
            layer_norm: whether to apply tensor layer normalization
        """

        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.padding = filter_size // 2
        self._forget_bias = forget_bias
        # spatial block
        self.spatialBlock = SpatialBlock(num_hidden, width, sp_fusion, sp_tln)
        
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 2, width, width])
            )
            self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        else:
            # nn.Conv2d: in_channels, out_channels, kernel_size, stride, padding
            self.conv_x = nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

            self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

            self.conv_c = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

            self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden * 2, kernel_size=filter_size,
                                    stride=stride, padding=self.padding)

            self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t).cuda()
        h_concat = self.conv_h(h_t).cuda()
        c_concat = self.conv_c(c_t).cuda()

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        m_mask = self.spatialBlock(x_t, c_new, m_t)
        m_new = m_t * m_mask

        mem = torch.cat((c_new, m_new), 1)
        c_m_concat = self.conv_o(mem).cuda()
        o_c, o_m = torch.split(c_m_concat, self.num_hidden, dim=1)

        o_t = torch.tanh(o_x + o_h + o_c + o_m)  # on paper that is no o_h
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

