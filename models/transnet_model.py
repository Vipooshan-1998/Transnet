import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, global_max_pool, GATv2Conv, SAGPooling
from torch_geometric.nn.norm import InstanceNorm
import copy
import sys


class Stagnet(nn.Module):
    def __init__(self, input_dim=4096, embedding_dim=128, img_feat_dim=2304, num_classes=2):
        super(Stagnet, self).__init__()

        self.num_heads = 1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # process the object graph features
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)
        
        # GNN for encoding the object-level graph
        self.gc1_spatial = GATv2Conv(
            embedding_dim * 2 + embedding_dim // 2, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=1  
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
        
        self.gc1_temporal = GATv2Conv(
            embedding_dim * 2 + embedding_dim // 2, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=1 
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2)
        
        self.pool = SAGPooling(embedding_dim, ratio=0.8)

        # SlowFast features processing
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        # Added LSTM for temporal sequence processing
        self.temporal_lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # Frame-level graph
        self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim // 2, heads=self.num_heads)
        self.gc2_norm1 = InstanceNorm(embedding_dim // 2)
        self.gc2_i3d = GATv2Conv(embedding_dim * 2,  embedding_dim // 2, heads=self.num_heads)
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

        self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
        
        """
        Inputs: 
        x - object-level graph nodes' feature matrix 
        edge_index - spatial graph connectivity for object-level graph 
        img_feat - frame SlowFast features 
        video_adj_list - Graph connectivity for frame-level graph
        edge_embeddings - Edge features for the object-level graph
        temporal_adj_list - temporal graph connectivity for object-level graph 
        temporal_wdge_w - edge weights for frame-level graph 
        batch_vec - vector for graph pooling the object-level graph

        Returns: 
        logits_mc - Final logits 
        probs_mc - Final probabilities
        """
		
		# process object graph features
        x_feat = self.x_fc(x[:, :self.input_dim])
        x_feat = self.relu(self.x_bn1(x_feat))
        x_label = self.obj_l_fc(x[:, self.input_dim:])
        x_label = self.relu(self.obj_l_bn1(x_label))
        x = torch.cat((x_feat, x_label), 1)

        # Get graph embedding for object-level graph
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
        ))
        
        # Temporal processing
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
        ))
        
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # Process SlowFast feature with temporal modeling
        img_feat = self.img_fc(img_feat)

		# LSTM processing 
        img_feat = img_feat.unsqueeze(0)  
        img_feat, (_, _) = self.temporal_lstm(img_feat)
        img_feat = img_feat.squeeze(0)

        # Get frame embedding for all nodes in frame-level graph
        frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
        frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
        frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
        frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
        logits_mc = self.classify_fc2(frame_embed_sg)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


