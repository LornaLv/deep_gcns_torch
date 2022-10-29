import torch
# import torch.nn as nn
# import copy
try:
    from .gcn_revop import InvertibleModuleWrapper
except:
    from gcn_revop import InvertibleModuleWrapper

class GroupAdditiveCoupling(torch.nn.Module):
    # split_dim:拆分维度，沿指定维度进行拆分
    def __init__(self, Fms, split_dim=-1, group=2):
        super(GroupAdditiveCoupling, self).__init__()

        self.Fms = Fms
        self.split_dim = split_dim
        self.group = group

    def forward(self, x, edge_index, *args):
        # torch.chunk 在给定维度(轴)上对输入张量进行分块，分为两组
        # xs表示的是分组X1、X2，即有xs[1]=X1,xs[2]=X2
        xs = torch.chunk(x, self.group, dim=self.split_dim)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
        args_chunks = list(zip(*chunked_args))
        # y_in对应于X0′
        y_in = 2 * sum(xs[1:])
        # y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            # y对应于Xi′
            # xs[i]加系数
            # y = 2 * xs[i] + Fmd
            y = xs[i] + Fmd
            y_in = y
            ys.append(y)

        out = torch.cat(ys, dim=self.split_dim)

        return out

    def inverse(self, y, edge_index, *args):
        ys = torch.chunk(y, self.group, dim=self.split_dim)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
        args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.group-1, -1, -1):
            if i != 0:
                y_in = ys[i-1]
            else:
                y_in = sum(xs)/2
                # y_in = sum(xs)

            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            # x = (ys[i] - Fmd)/2
            x = ys[i] - Fmd
            xs.append(x)

        x = torch.cat(xs[::-1], dim=self.split_dim)

        return x
