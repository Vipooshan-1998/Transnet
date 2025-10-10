import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, SAGPooling, global_max_pool, InstanceNorm
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
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
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

        # Graph pooling
        self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D and Attention-SlowFast features -> Transformers
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        self.atten_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        # Image Transformer
        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=self.num_layers)

        # Attention Transformer
        encoder_layer_atten = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_atten = TransformerEncoder(encoder_layer_atten, num_layers=self.num_layers)

        # Global graph Transformer
        encoder_layer_graph = TransformerEncoderLayer(
            d_model=embedding_dim * self.num_heads * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_graph = TransformerEncoder(encoder_layer_graph, num_layers=self.num_layers)

        # Frame-level GraphConv after pooling
        self.gc2_sg = TransformerConv(
            in_channels=embedding_dim * self.num_heads * 2,
            out_channels=embedding_dim,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc2_norm1 = InstanceNorm(embedding_dim * self.num_heads)

        # GraphConv for img features
        self.gc2_i3d = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc2_norm2 = InstanceNorm(embedding_dim * self.num_heads)

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * 3 + embedding_dim * self.num_heads * 2  # g_embed_out + frame_embed_sg + frame_embed_img + optionally atten
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, atten_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec, video_adj_list):

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # Spatial graph
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)))

        # Temporal graph
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)))

        # -----------------------
        # Global pooling + Transformer
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)
        g_embed_trans = self.temporal_transformer_graph(g_embed.unsqueeze(0)).squeeze(0)

        # -----------------------
        # Image feature processing: Transformer -> GraphConv
        # -----------------------
        img_feat_trans = self.temporal_transformer_img(self.img_fc(img_feat).unsqueeze(0)).squeeze(0)
        frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_trans, video_adj_list)))

        # -----------------------
        # Frame-level graph features -> GraphConv -> Transformer
        # -----------------------
        frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
        frame_embed_sg = self.temporal_transformer_graph(frame_embed_sg.unsqueeze(0)).squeeze(0)

        # -----------------------
        # Attention SlowFast feature processing
        # -----------------------
        atten_feat_seq = self.temporal_transformer_atten(self.atten_fc(atten_feat).unsqueeze(0)).squeeze(0)

        # -----------------------
        # Concatenate all outputs
        # -----------------------
        fused_feat = torch.cat((g_embed_trans, frame_embed_sg, frame_embed_img, atten_feat_seq), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc

