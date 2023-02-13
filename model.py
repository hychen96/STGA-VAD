import torch
from torchsummary import summary

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from graph_layers import GraphConvolution
import numpy as np



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, is_concat=True, dropout_rate=0.6, leaky_relu_negative_slope=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.is_concat = is_concat
        self.dropout_rate = dropout_rate
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

        if self.is_concat:
            assert self.out_features % self.n_heads == 0
            self.n_hidden = self.out_features // self.n_heads
        else:
            self.n_hidden = self.out_features

        self.linear = nn.Linear(self.in_features, self.n_hidden*self.n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden*2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, h, adj_mat):
        n_nodes = h.shape[0]

        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        # e = e.masked_fill(adj_mat == 0, float('-inf'))
        adj = adj_mat.permute(2,0,1).repeat(self.n_heads,1,1).permute(1,2,0)
        e = e.mul(adj)
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('ijh,jhf->ihf', a, g)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)



class GAT(nn.Module):
    def __init__(self, in_features, n_hidden=64, n_classes=1, n_heads=8, dropout_rate=0.6):
        super(GAT, self).__init__()

        self.in_features = in_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.layer1 = GraphAttentionLayer(self.in_features, self.n_hidden, self.n_heads, is_concat=True, dropout_rate=self.dropout_rate)
        self.activation = nn.ELU()

        self.output = GraphAttentionLayer(self.n_hidden, self.n_classes, 1, is_concat=False, dropout_rate=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)



    def forward(self, x, adj_mat):
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.output(x, adj_mat)
        x = torch.softmax(x, dim=0)


        return x


class STGA(nn.Module):
    def __init__(self, nfeat, nclass, dropout_rate=0.6):
        super(STGA, self).__init__()
        # original layers
        self.fc1 = nn.Linear(nfeat, 512)
        self.fc2 = nn.Linear(512, 128)
        # Graph Convolution
        self.gc1 = GraphConvolution(128, 32)  # nn.Linear(128, 32)
        self.gc2 = GraphConvolution(32, nclass)
        self.gc3 = GraphConvolution(128, 32)  # nn.Linear(128, 32)
        self.gc4 = GraphConvolution(32, nclass)  # nn.Linear(128, 32)

        self.dropout_rate = dropout_rate

        self.GAT1 = GAT(in_features=nfeat)
        self.GAT2 = GAT(in_features=nfeat)

    def forward(self, x, a1): #adj=n*n  x=n*4096
        # assert (x.shape[0] == 1)
        # L2-normalization

        x_att = x
        x = x / torch.norm(x, p=2, dim=-1).reshape(-1, 1)

        # original layers
        x = F.relu(self.fc1(x))  #x=n*512
        x = F.dropout(x, self.dropout_rate, training=True)
        x = F.relu(self.fc2(x))  #x=n*128
        x = F.dropout(x, self.dropout_rate, training=True)

        #Temporal Graph
        a1 = a1 + torch.eye(a1.shape[0]).cuda()

        d_inv_sqrt1 = torch.diag(torch.pow(torch.sum(a1, dim=1), -0.5))

        att_1 = self.GAT1(x_att, a1.unsqueeze(-1))

        adj_hat1 = d_inv_sqrt1.matmul(a1).matmul(d_inv_sqrt1)

        x1 = F.relu(self.gc1(x, adj_hat1)) #x1=n*32

        x1 = self.gc2(x1, adj_hat1) #x1=n*1

        x1 = torch.sigmoid(x1)


        # Spatial Graph
        # a2 = torch.from_numpy(cos(x)).cuda()
        a2 = x.matmul(x.t())
        # b = a2.max(dim=1,keepdim=True)
        a2 = torch.exp(a2 - a2.max(dim=1,keepdim=True)[0])

        a2 = a2 + torch.eye(a2.shape[0]).cuda()
        d_inv_sqrt2 = torch.diag(torch.pow(torch.sum(a2, dim=1), -0.5))

        att_2 = self.GAT2(x_att, a2.unsqueeze(-1))

        adj_hat2 = d_inv_sqrt2.matmul(a2).matmul(d_inv_sqrt2)

        x2 = F.relu(self.gc3(x, adj_hat2))

        x2 = self.gc4(x2, adj_hat2)

        x2 = torch.sigmoid(x2)


        # return (x1 + x2) / 2.0
        return (att_1+att_2+x1+x2)/4.0


