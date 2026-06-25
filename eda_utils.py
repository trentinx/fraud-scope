"""Fonctions utilitaires pour l'analyse exploratoire du dataset IEEE Fraud Detection."""

import pandas as pd
import numpy as np


def load_train_data(data_dir: str = "data") -> pd.DataFrame:
    """Charge et fusionne train_transaction + train_identity."""
    tx = pd.read_csv(f"{data_dir}/train_transaction.csv")
    identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    df = tx.merge(identity, on="TransactionID", how="left")
    return df


def check_client_proxy(df: pd.DataFrame, key_cols: list[str], check_col: str = "P_emaildomain") -> dict:
    """Évalue si une combinaison de colonnes identifie un client unique.

    Crée une clé à partir de *key_cols*, puis mesure combien de valeurs
    distinctes de *check_col* existent par groupe.  Si un groupe possède
    beaucoup de valeurs différentes, la clé mélange probablement plusieurs
    clients.

    Retourne un dict résumé prêt à être affiché.
    """
    key = df[key_cols].fillna("NA").astype(str).agg("_".join, axis=1)
    n_groups = key.nunique()
    check_per_group = df.groupby(key)[check_col].nunique()

    n_single = (check_per_group <= 1).sum()
    n_multi = (check_per_group > 1).sum()
    tx_per_group = key.value_counts()

    return {
        "key_cols": " + ".join(key_cols),
        "nb_groupes": n_groups,
        "1_seul_email": n_single,
        "2+_emails": n_multi,
        "pct_collision": round(n_multi / n_groups * 100, 1),
        "median_tx": tx_per_group.median(),
        "max_tx": tx_per_group.max(),
    }


def get_cols_to_drop(corr_matrix: pd.DataFrame, corr_target: pd.Series, threshold: float) -> set:
    """Retourne les colonnes à supprimer pour un seuil de corrélation donné.

    Pour chaque paire avec |corr| > threshold, supprime la colonne
    la moins corrélée avec la target.
    """
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and abs(val) > threshold:
                if idx not in to_drop and col not in to_drop:
                    if corr_target.get(idx, 0) >= corr_target.get(col, 0):
                        to_drop.add(col)
                    else:
                        to_drop.add(idx)
    return to_drop


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes temporelles à partir de TransactionDT (secondes)."""
    df = df.copy()
    df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day"] = (df["TransactionDT"] / 86400).astype(int)
    df["day_of_week"] = df["day"] % 7
    return df
