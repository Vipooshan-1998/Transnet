import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    SAGPooling,
    global_max_pool,
    InstanceNorm
)
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpaceTempGoG_detr_dad(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(SpaceTempGoG_detr_dad, self).__init__()

        self.num_heads = 4
        self.num_layers = 2
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)   # 2048 -> 256
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)         # 300 -> 64
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,  # 256+64=320
            out_channels=embedding_dim // 2,                      # 64
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Cross-graph attention
        # -----------------------
        self.gc_cross = TransformerConv(
            in_channels=(embedding_dim // 2 * self.num_heads) * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc_cross_norm = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim // 2 * self.num_heads, ratio=0.8)

        # -----------------------
        # Transformer for pooled graph embeddings
        # -----------------------
        encoder_layer_graph = TransformerEncoderLayer(
            d_model=embedding_dim // 2 * self.num_heads,
            nhead=self.num_heads,
            batch_first=True
        )
        self.graph_transformer = TransformerEncoder(encoder_layer_graph, num_layers=self.num_layers)

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)  # 2048 -> 256
        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=self.num_heads,
            batch_first=True
        )
        self.temporal_transformer = TransformerEncoder(encoder_layer_img, num_layers=self.num_layers)

        # Attention features -> Transformer
        self.atten_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        encoder_layer_atten = TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=self.num_heads,
            batch_first=True
        )
        self.temporal_transformer_atten = TransformerEncoder(encoder_layer_atten, num_layers=self.num_layers)

        # -----------------------
        # Frame-level graph encoding
        # -----------------------
        self.gc2_sg = TransformerConv(
            in_channels=embedding_dim // 2 * self.num_heads,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc2_i3d = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Apply TransformerConv to attention features as well
        self.gc2_atten = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc2_norm3 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = (embedding_dim // 2 * self.num_heads) + \
                     (embedding_dim // 2 * self.num_heads) + \
                     (embedding_dim * 2) + \
                     (embedding_dim // 2 * self.num_heads)  # graph_transformer output

        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, atten_feat, video_adj_list, edge_embeddings,
                temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)  # (N, 320)

        # Spatial graph
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # Temporal graph
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Cross-graph attention
        # -----------------------
        n_embed_cross = torch.cat((n_embed_spatial, n_embed_temporal), dim=-1)
        n_embed_cross = self.relu(self.gc_cross_norm(self.gc_cross(n_embed_cross, edge_index)))

        # Graph pooling
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed_cross, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # Transformer on pooled graph embeddings
        # -----------------------
        g_embed_trans = g_embed.unsqueeze(0)
        g_embed_trans = self.graph_transformer(g_embed_trans)
        g_embed_trans = g_embed_trans.squeeze(0)

        # -----------------------
        # I3D feature processing
        # -----------------------
        img_feat_orig = self.img_fc(img_feat).unsqueeze(0)
        img_feat_orig = self.temporal_transformer(img_feat_orig).squeeze(0)
        frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

        # -----------------------
        # Attention feature processing with TransformerConv
        # -----------------------
        atten_feat_fc = self.atten_fc(atten_feat).unsqueeze(0)
        atten_feat_trans = self.temporal_transformer_atten(atten_feat_fc).squeeze(0)
        frame_embed_atten = self.relu(self.gc2_norm3(self.gc2_atten(atten_feat_trans, video_adj_list)))

        # -----------------------
        # Pool object graph embeddings for sg
        # -----------------------
        frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))

        # -----------------------
        # Concatenate all features
        # -----------------------
        frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, frame_embed_atten, g_embed_trans), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
        logits_mc = self.classify_fc2(frame_embed_)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc
