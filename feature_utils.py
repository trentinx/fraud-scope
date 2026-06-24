"""Fonctions de feature engineering temporel pour IEEE Fraud Detection."""

import pandas as pd
import numpy as np

CLIENT_COL = "uid"


def _build_uid(df: pd.DataFrame) -> pd.Series:
    """Construit un identifiant client à partir de card1, addr1 et D1.

    D1 représente le nombre de jours depuis la première transaction du client.
    En soustrayant D1 du jour courant, on obtient le jour exact de la première
    transaction, ce qui permet de distinguer des clients partageant la même
    carte et la même adresse mais ayant des historiques différents.
    """
    first_day = (df["TransactionDT"] / 86400).astype(int) - df["D1"]
    return (
        df["card1"].astype(str) + "_"
        + df["addr1"].fillna("NA").astype(str) + "_"
        + first_day.fillna("NA").astype(str)
    )


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Trie par temps, construit le UID client et crée un index DateTime unique."""
    df = df.sort_values("TransactionDT").reset_index(drop=True).copy()
    df[CLIENT_COL] = _build_uid(df)
    df["TransactionDateTime"] = pd.to_datetime(df["TransactionDT"], unit="s", origin="2026-01-01")
    # Ajout de nanosecondes pour rendre chaque timestamp unique (requis par reindex)
    dup_offset = df.groupby("TransactionDateTime").cumcount()
    df["_unique_dt"] = df["TransactionDateTime"] + pd.to_timedelta(dup_offset, unit="ns")
    return df


# --- Features de vélocité ---

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features de vélocité par client (uid), vectorisé.

    Pour chaque transaction, calcule sur les transactions précédentes du même client :
    - nb de transactions dans les 1h, 24h, 7j
    - montant cumulé dans les 1h, 24h, 7j
    """
    df_idx = df.set_index("_unique_dt")
    grouped = df_idx.groupby(CLIENT_COL)["TransactionAmt"]

    windows = {"1h": "1h", "24h": "24h", "7d": "7d"}

    for suffix, window_size in windows.items():
        rolling = grouped.rolling(window_size, closed="left")
        df[f"tx_count_{suffix}"] = rolling.count().droplevel(0).reindex(df_idx.index).values
        df[f"tx_amount_{suffix}"] = rolling.sum().droplevel(0).reindex(df_idx.index).values

    velocity_cols = [c for c in df.columns if c.startswith("tx_")]
    df[velocity_cols] = df[velocity_cols].fillna(0)

    return df.copy()


# --- Features de déviation comportementale ---

def add_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features de déviation comportementale, vectorisé.

    - ratio montant_actuel / montant_moyen_7j du client
    - booléen 'nouveau marchand' (ProductCD jamais vu par ce client)
    - booléen 'nouvel appareil' (DeviceInfo jamais vu par ce client)
    """
    # Ratio montant actuel / moyenne 7j
    df_idx = df.set_index("_unique_dt")
    mean_7d = (
        df_idx.groupby(CLIENT_COL)["TransactionAmt"]
        .rolling("7d", closed="left")
        .mean()
        .droplevel(0)
        .reindex(df_idx.index)
        .values
    )
    df["amount_ratio_7d"] = df["TransactionAmt"] / mean_7d
    df["amount_ratio_7d"] = df["amount_ratio_7d"].fillna(1.0).replace([np.inf, -np.inf], 1.0)

    # Nouveau marchand
    df["is_new_merchant"] = (df.groupby([CLIENT_COL, "ProductCD"]).cumcount() == 0).astype(int)

    # Nouvel appareil
    is_device_notna = df["DeviceInfo"].notna()
    df["is_new_device"] = 0
    df.loc[is_device_notna, "is_new_device"] = (
        df[is_device_notna].groupby([CLIENT_COL, "DeviceInfo"]).cumcount() == 0
    ).astype(int)

    return df.copy()
