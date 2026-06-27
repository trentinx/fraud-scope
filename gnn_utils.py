"""Modèles GNN et fonctions d'entraînement/évaluation pour la détection de fraude."""

import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import recall_score, average_precision_score, f1_score


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * heads, 2, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gnn(model, data, y, train_mask, epochs=100, lr=0.01):
    """Entraîne un GNN avec class weights pour compenser le déséquilibre."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    n_pos = y[train_mask].sum().item()
    n_neg = (train_mask.sum() - n_pos).item()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)])

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=weight)
        loss.backward()
        optimizer.step()

    return model


def evaluate_gnn(model, data, y, test_mask):
    """Évalue un GNN et retourne recall, AUPRC, F1 et temps d'inférence."""
    model.eval()
    with torch.no_grad():
        start = time.time()
        out = model(data.x, data.edge_index)
        inference_time = time.time() - start

        proba = F.softmax(out[test_mask], dim=1)[:, 1].numpy()
        pred = out[test_mask].argmax(dim=1).numpy()
        true = y[test_mask].numpy()

    return {
        "recall": recall_score(true, pred),
        "auprc": average_precision_score(true, proba),
        "f1": f1_score(true, pred),
        "inference_ms": round(inference_time * 1000, 1),
    }
