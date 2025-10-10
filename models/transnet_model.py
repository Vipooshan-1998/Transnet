import torch
import torch.nn as nn
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

        self.num_heads = 1
        self.num_layers = 1
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

        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        encoder_layer_atten = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )

        self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=self.num_layers)
        self.temporal_transformer_atten = TransformerEncoder(encoder_layer_atten, num_layers=self.num_layers)

        # -----------------------
        # LSTMs (num_layers=1, hidden_size = input_size)
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * self.num_heads,
            hidden_size=embedding_dim * self.num_heads,
            num_layers=1,
            batch_first=True
        )
        self.temporal_lstm_img = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )
        self.temporal_lstm_atten = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * self.num_heads + embedding_dim * 2 + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, atten_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # Spatial graph
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # Temporal graph
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # Concat + pooling
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec).unsqueeze(0)  # (1, batch, feat)

        # -----------------------
        # LSTM over pooled graph features
        # -----------------------
        lstm_out_graph, _ = self.temporal_lstm_graph(g_embed)
        lstm_out_graph = lstm_out_graph[:, -1, :]  # last timestep

        # -----------------------
        # I3D feature processing
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = self.temporal_transformer_img(img_feat_proj)

        img_feat_proj = img_feat_proj.unsqueeze(0)  # add batch dim
        lstm_out_img, _ = self.temporal_lstm_img(img_feat_proj)
        lstm_out_img = lstm_out_img[:, -1, :]

        # -----------------------
        # Attention SlowFast feature processing
        # -----------------------
        atten_feat_proj = self.atten_fc(atten_feat)
        atten_feat_proj = self.temporal_transformer_atten(atten_feat_proj)

        atten_feat_proj = atten_feat_proj.unsqueeze(0)
        lstm_out_atten, _ = self.temporal_lstm_atten(atten_feat_proj)
        lstm_out_atten = lstm_out_atten[:, -1, :]

        # -----------------------
        # Concatenate all LSTM outputs
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, lstm_out_img, lstm_out_atten), dim=-1)

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc
