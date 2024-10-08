"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch as th
import torch.nn as nn
# from dgl.nn import GraphConv
from .graphconv_edge_weight import GraphConvEdgeWeight as GraphConv
import numpy as np
from sklearn.manifold import TSNE

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=normalization))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=normalization))
        # output layer
        # self.layers.append(GraphConv(n_hidden*2, n_classes, norm=normalization))
        self.fc = nn.Linear(n_hidden+768, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    # def forward(self, features, g,g_gaze, edge_weight):
    #     h = features
    #     for i, layer in enumerate(self.layers):
    #         if i != 0:
    #             h = self.dropout(h)
    #
    #         h = layer(g, h, edge_weights=edge_weight)
    #
    #     return h

    def forward(self, features, g,g_gaze, edge_weight,gaze_weight,cls_feats):
        h = features
        h_gaze = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
                h_gaze = self.dropout(h_gaze)
            # if i < len(self.layers)-1:
            h = layer(g, h, edge_weights=edge_weight)
            h_gaze = layer(g_gaze,h_gaze,edge_weights=gaze_weight)
            # else:
        # print(h.shape, h_gaze.shape)
        h_final = torch.cat((h,h_gaze),dim=1)
        # print(h.shape, h_gaze.shape,h_final.shape)
        # print('cls_feats:', features.shape)

        # h_final = torch.cat((h_final,features),dim=1)
        # h_final = torch.cat((h, features), dim=1)
        h_final = torch.cat((features, h_gaze), dim=1)
        # print('shape:',h_final.shape)
        h_final1 = self.fc(h_final)
        # gcn_pred = th.nn.Softmax(dim=1)(h_final1)

        # x = h_final1.cuda().data.cpu().numpy()


        # cls_feats.transpose(1, 2)
                #
                # print(h_final.shape)
                # h_final = torch.cat((h_final,cls_feats),dim=1)
                # h_final = layer(g,h_final)
        return h_final1#,h_final