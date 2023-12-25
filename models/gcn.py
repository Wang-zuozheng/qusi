import math
import numpy as np
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_feature(nn.Module):
    def __init__(self, in_features, mid_features):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, mid_features)
        self.gc2 = GraphConvolution(mid_features, mid_features)
        self.gc3 = GraphConvolution(mid_features, in_features)
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)    
    
    def forward(self, text_features, gcn_relation):
        text_features = self.gc1(text_features, gcn_relation.cuda())
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, gcn_relation.cuda())
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, gcn_relation.cuda())
        return text_features