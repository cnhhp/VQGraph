from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from vq import VectorQuantize
import dgl

from models.token_predictor import attach_token_predictor_to_encoder, maybe_add_token_loss


class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type,
        codebook_size,
        lamb_edge,
        lamb_node,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.norms = nn.ModuleList()
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        self.decoder_1 = nn.Linear(input_dim, input_dim)
        self.decoder_2 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.vq = VectorQuantize(
            dim=input_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=0.25,
            use_cosine_sim=True,
        )
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
        self.token_predictor: Optional[nn.Linear] = None
        self.lambda_token: float = 0.0
        self.token_pred_tau: float = 1.0
        self.token_target_tau: float = 0.15

    def attach_token_predictor(
        self,
        vocab_size: int,
        tokenbook_embeddings: torch.Tensor,
        cfg: Union[object, dict],
    ) -> None:
        attach_token_predictor_to_encoder(self, vocab_size, tokenbook_embeddings, cfg)

    def forward(self, g, feats, text_emb: Optional[torch.Tensor] = None):
        h = feats
        adj = g.adjacency_matrix().to_dense().to(feats.device)
        h_list = []
        h = self.graph_layer_1(g, h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)
        h_list.append(h)
        if text_emb is not None:
            quantized, _, commit_loss, dist, codebook = self.vq(h, text_emb=text_emb)
        else:
            quantized, _, commit_loss, dist, codebook = self.vq(h)
        quantized_edge = self.decoder_1(quantized)
        quantized_node = self.decoder_2(quantized)

        feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)
        adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        adj_quantized = (adj_quantized - adj_quantized.min()) / (
            adj_quantized.max() - adj_quantized.min()
        )
        edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))

        dist = torch.squeeze(dist)
        h_list.append(quantized)
        h = self.graph_layer_2(g, quantized_edge)
        h_list.append(h)
        h = self.linear(h)
        loss = feature_rec_loss + edge_rec_loss + commit_loss

        token_loss_weighted, token_loss_raw = maybe_add_token_loss(
            self, quantized, text_emb
        )
        loss = loss + token_loss_weighted

        return h_list, h, loss, dist, codebook, token_loss_raw


class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type,
        codebook_size,
        lamb_edge,
        lamb_node,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.norms = nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        self.decoder_1 = nn.Linear(input_dim, input_dim)
        self.decoder_2 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.codebook_size = codebook_size
        self.vq = VectorQuantize(
            dim=input_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=0.25,
            use_cosine_sim=True,
        )
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
        self.token_predictor: Optional[nn.Linear] = None
        self.lambda_token: float = 0.0
        self.token_pred_tau: float = 1.0
        self.token_target_tau: float = 0.15

    def attach_token_predictor(
        self,
        vocab_size: int,
        tokenbook_embeddings: torch.Tensor,
        cfg: Union[object, dict],
    ) -> None:
        attach_token_predictor_to_encoder(self, vocab_size, tokenbook_embeddings, cfg)

    def forward(self, blocks, feats, text_emb: Optional[torch.Tensor] = None):
        h = feats
        g = dgl.DGLGraph().to(h.device)
        g.add_nodes(h.shape[0])
        blocks = [blk.int() for blk in blocks]
        for block in blocks:
            src, dst = block.all_edges()
            src = src.type(torch.int64)
            dst = dst.type(torch.int64)
            g.add_edges(src, dst)
            g.add_edges(dst, src)
        adj = g.adjacency_matrix().to_dense().to(feats.device)
        h_list = []
        h = self.graph_layer_1(g, h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)
        h_list.append(h)
        if text_emb is not None:
            quantized, _, commit_loss, dist, codebook = self.vq(h, text_emb=text_emb)
        else:
            quantized, _, commit_loss, dist, codebook = self.vq(h)
        quantized_edge = self.decoder_1(quantized)
        quantized_node = self.decoder_2(quantized)

        feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)
        adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        adj_quantized = (adj_quantized - adj_quantized.min()) / (
            adj_quantized.max() - adj_quantized.min()
        )
        edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))

        dist = torch.squeeze(dist)
        h_list.append(quantized)
        h = self.graph_layer_2(g, quantized_edge)
        h_list.append(h)
        h = self.linear(h)
        loss = feature_rec_loss + edge_rec_loss + commit_loss

        token_loss_weighted, token_loss_raw = maybe_add_token_loss(
            self, quantized, text_emb
        )
        loss = loss + token_loss_weighted

        h = h[: blocks[-1].num_dst_nodes()]
        return h_list, h, loss, dist, codebook, token_loss_raw

    def inference(self, dataloader, feats, text_emb: Optional[torch.Tensor] = None):
        device = feats.device
        dist_all = torch.zeros(feats.shape[0], self.codebook_size, device=device)
        y = torch.zeros(feats.shape[0], self.output_dim, device=device)
        for input_nodes, output_nodes, blocks in dataloader:
            g = dgl.DGLGraph().to(feats.device)
            g.add_nodes(input_nodes.shape[0])
            block = blocks[0].int().to(device)
            src, dst = block.all_edges()
            src = src.type(torch.int64)
            dst = dst.type(torch.int64)
            g.add_edges(src, dst)
            g.add_edges(dst, src)
            adj = g.adjacency_matrix().to_dense().to(feats.device)
            h_list = []
            h = feats[input_nodes]
            h = self.graph_layer_1(g, h)
            if self.norm_type != "none":
                h = self.norms[0](h)
            h = self.dropout(h)
            h_list.append(h)
            batch_text = None
            if text_emb is not None:
                batch_text = text_emb[input_nodes]
                quantized, _, commit_loss, dist, codebook = self.vq(
                    h, text_emb=batch_text
                )
            else:
                quantized, _, commit_loss, dist, codebook = self.vq(h)
            dist = torch.squeeze(dist)
            dist_all[input_nodes] = dist
            quantized_edge = self.decoder_1(quantized)
            quantized_node = self.decoder_2(quantized)

            feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)
            adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
            adj_quantized = (adj_quantized - adj_quantized.min()) / (
                adj_quantized.max() - adj_quantized.min()
            )
            edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))
            h = self.graph_layer_2(g, quantized_edge)
            h_list.append(h)
            h = self.linear(h)
            loss = feature_rec_loss + edge_rec_loss + commit_loss
            h = h[: block.num_dst_nodes()]
            y[output_nodes] = h

        return h_list, y, loss, dist_all, codebook, None


class Model(nn.Module):
    """Wrapper for GCN / SAGE teacher models with VQ codebook."""

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"],
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"],
            ).to(conf["device"])
        else:
            raise ValueError(
                f"Unsupported teacher model: {conf['model_name']}. Use GCN or SAGE."
            )

    def attach_token_predictor(
        self,
        vocab_size: int,
        tokenbook_embeddings: torch.Tensor,
        cfg: Union[object, dict],
    ) -> None:
        self.encoder.attach_token_predictor(vocab_size, tokenbook_embeddings, cfg)

    def forward(self, data, feats, text_emb: Optional[torch.Tensor] = None):
        return self.encoder(data, feats, text_emb=text_emb)

    def inference(self, data, feats, text_emb: Optional[torch.Tensor] = None):
        if "SAGE" in self.model_name:
            return self.encoder.inference(data, feats, text_emb=text_emb)
        out = self.forward(data, feats, text_emb=text_emb)
        if len(out) == 5:
            return out + (None,)
        return out
