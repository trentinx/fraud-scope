"""Fonctions utilitaires pour l'analyse de graphe sur IEEE Fraud Detection."""

import pandas as pd
import numpy as np
import networkx as nx


def build_bipartite_graph(df: pd.DataFrame, uid_col: str = "uid") -> nx.Graph:
    """Construit un graphe biparti clients–marchands à partir d'un DataFrame de transactions.

    Noeuds clients : identifiés par uid_col.
    Noeuds marchands : identifiés par ProductCD (préfixés "M_").
    Arêtes : une par transaction, avec montant, label fraude et TransactionID.
    """
    G = nx.Graph()

    for prod in df["ProductCD"].unique():
        G.add_node(f"M_{prod}", node_type="merchant", label=prod)

    for _, row in df.iterrows():
        client = row[uid_col]
        merchant = f"M_{row['ProductCD']}"

        if client not in G:
            G.add_node(client, node_type="client")

        G.add_edge(client, merchant,
                   amount=row["TransactionAmt"],
                   is_fraud=int(row["isFraud"]),
                   tx_id=row["TransactionID"])

    return G


def extract_graph_features(df: pd.DataFrame, G: nx.Graph, uid_col: str = "uid") -> pd.DataFrame:
    """Extrait les features graphe par client.

    - graph_degree : nombre de transactions du client
    - graph_merchants_7d : nombre max de marchands distincts sur une fenêtre de 7 jours
    - graph_betweenness : centralité betweenness (calculée sur le graphe fourni)
    """
    degree_df = df.groupby(uid_col).size().rename("graph_degree")

    day = (df["TransactionDT"] / 86400).astype(int)
    merchants_7d = (
        df.assign(day=day)
        .groupby(uid_col)
        .apply(
            lambda g: g.groupby("day")["ProductCD"].nunique()
            .rolling(7, min_periods=1).max().iloc[-1],
            include_groups=False,
        )
        .rename("graph_merchants_7d")
    )

    betweenness = nx.betweenness_centrality(G)
    betweenness_df = pd.Series(
        {node: val for node, val in betweenness.items()
         if G.nodes[node].get("node_type") == "client"},
        name="graph_betweenness",
    )

    graph_features = pd.concat([degree_df, merchants_7d], axis=1)
    graph_features = graph_features.join(betweenness_df, how="left")
    graph_features["graph_betweenness"] = graph_features["graph_betweenness"].fillna(0)

    return graph_features
