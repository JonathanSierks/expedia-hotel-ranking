"""
Pointwise Factorization Machine fuer das Expedia Hotel Ranking -- simpel gehalten.

Nutzt die Library `torchfm` (pip install torchfm). Bewusst auf einen REDUZIERTEN
Feature-Satz beschraenkt: die selbst gebauten Ranking-Features plus ein paar
Basis-Features, die sich gut eignen.

Wichtig zu torchfm:
  * Das FM-Modell erwartet ausschliesslich kategoriale Felder: Input ist ein
    Long-Tensor (batch, num_fields), jeder Eintrag ein Index in eine
    Embedding-Tabelle. Deshalb werden ALLE Features zu Integer-Codes --
    kategoriale per Mapping, kontinuierliche per Quantil-Bucketing.
  * Der forward-Output hat schon ein Sigmoid -> Loss = binary_cross_entropy.
  * Encoder werden NUR auf Train gefittet und dann auf Val angewendet
    (gleiche Bin-Grenzen, gleiche Kategorie-Codes) -- sonst leakt Val-Info.

--------------------------------------------------------------------------
Nutzung direkt im Notebook (du hast df_train_balanced / df_val schon):

    from pointwise_fm import fit_encoders, transform, train_fm, ndcg_at_k

    encoders, field_dims = fit_encoders(df_train_balanced, n_bins=32)
    X_tr, _ = transform(df_train_balanced, encoders)
    X_va, _ = transform(df_val, encoders)
    model, best = train_fm(X_tr, df_train_balanced["target"].to_numpy("float32"),
                           X_va, df_val, field_dims)

Oder als Skript auf der rohen CSV (macht Split + Downsampling selbst):

    python pointwise_fm.py --data training_set_VU_DM.csv
    python pointwise_fm.py --data training_set_VU_DM.csv --quick
--------------------------------------------------------------------------
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from torchfm.model.fm import FactorizationMachineModel


# ----------------------------------------------------------------------
# Reduzierter Feature-Satz:  Spaltenname -> "cat" (kategorial) | "num" (kontinuierlich)
# Auskommentieren / ergaenzen nach Bedarf.
# ----------------------------------------------------------------------
FEATURES = {
    # --- selbst gebaute Ranking-/Gruppen-Features ---
    "price_rank_in_group":       "num",
    "price_group_median_ratio":  "num",
    "price_group_mean_diff":     "num",
    "star_rank":                 "num",
    "price_diff_rank":           "num",
    "price_vs_historical_diff":  "num",
    "prop_log_price_usd":        "num",
    "location_score1_bucket":    "cat",
    "location_score2_bucket":    "cat",
    "prop_id_encoded":           "cat",
    "srch_destination_id_encoded": "cat",
    "srch_room_count_encoded":   "cat",
    # --- ein paar gut geeignete Basis-Features ---
    "prop_starrating":           "cat",
    "prop_review_score":         "num",
    "prop_brand_bool":           "cat",
    "promotion_flag":            "cat",
    "prop_log_historical_price": "num",
    "srch_length_of_stay":       "num",
    "srch_booking_window":       "num",
    "srch_adults_count":         "num",
    "srch_saturday_night_bool":  "cat",
}
GROUP_COL = "srch_id"
RELEVANCE_COL = "relevance"   # gradierte Relevanz, nur fuer NDCG


# ----------------------------------------------------------------------
# Encoder: auf Train fitten, auf Train UND Val anwenden
# ----------------------------------------------------------------------
def fit_encoders(df_train: pd.DataFrame, n_bins: int = 32):
    """Lernt pro Feature entweder ein Kategorie-Mapping oder Quantil-Bin-Grenzen.
    Code 0 ist immer reserviert fuer NaN bzw. unbekannte Werte."""
    encoders = {}
    for col, kind in FEATURES.items():
        s = df_train[col]
        if kind == "cat":
            cats = pd.unique(s.dropna())
            mapping = {v: i + 1 for i, v in enumerate(cats)}     # 0 = NaN/unbekannt
            encoders[col] = ("cat", mapping)
        else:
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.nanquantile(s.to_numpy(dtype="float64"), qs))
            encoders[col] = ("num", edges)
    return encoders, _field_dims(encoders)


def _field_dims(encoders):
    dims = []
    for kind, enc in encoders.values():
        if kind == "cat":
            dims.append(len(enc) + 1)              # +1 fuer Code 0
        else:
            dims.append((len(enc) - 1) + 1)        # #Bins + NaN-Bucket
    return dims


def transform(df: pd.DataFrame, encoders):
    """DataFrame -> (X int-codes [n_rows, n_fields], field_dims)."""
    cols = []
    for col, (kind, enc) in encoders.items():
        s = df[col]
        if kind == "cat":
            codes = s.map(enc).fillna(0).astype("int64").to_numpy()
        else:
            edges = enc
            # pd.cut mit den Train-Bin-Grenzen; ausserhalb der Spanne -> NaN -> 0
            binned = pd.cut(s, bins=edges, labels=False, include_lowest=True)
            codes = pd.to_numeric(binned, errors="coerce").to_numpy()
            codes = np.where(np.isnan(codes), -1, codes).astype("int64") + 1
        cols.append(codes)
    X = np.column_stack(cols).astype("int64")
    return X, _field_dims(encoders)


# ----------------------------------------------------------------------
# NDCG@k, gruppiert nach srch_id  (auf df_val mit INTAKTEN Gruppen rechnen!)
# ----------------------------------------------------------------------
def ndcg_at_k(relevance, scores, groups, k=5):
    d = pd.DataFrame({"g": groups, "rel": relevance, "score": scores})
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for _, sub in d.groupby("g", sort=False):
        gains = 2.0 ** sub["rel"].to_numpy() - 1.0
        order = np.argsort(-sub["score"].to_numpy(), kind="stable")
        dcg = (gains[order][:k] * discounts[:len(order)][:k]).sum()
        idcg = (np.sort(gains)[::-1][:k] * discounts[:len(gains)][:k]).sum()
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out))


# ----------------------------------------------------------------------
# Training: pointwise BCE + Early Stopping auf Val-NDCG
# ----------------------------------------------------------------------
def train_fm(X_tr, y_tr, X_va, df_val, field_dims,
             embed_dim=16, batch_size=4096, epochs=20, patience=3,
             lr=1e-3, l2=1e-5, ndcg_k=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rel_va = df_val[RELEVANCE_COL].to_numpy(dtype="float64")
    grp_va = df_val[GROUP_COL].to_numpy()

    model = FactorizationMachineModel(field_dims, embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)  # weight_decay = L2 auf Embeddings

    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.array(X_tr, copy=True)),
                      torch.from_numpy(np.array(y_tr, copy=True))),
        batch_size=batch_size, shuffle=True)
    Xva_t = torch.from_numpy(np.array(X_va, copy=True)).to(device)

    best_ndcg, best_state, waited = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)                                  # Sigmoid ist schon drin
            loss = F.binary_cross_entropy(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            scores = model(Xva_t).cpu().numpy()
        ndcg = ndcg_at_k(rel_va, scores, grp_va, k=ndcg_k)
        print(f"  Epoche {epoch:2d} | train loss {loss.item():.4f} | Val NDCG@{ndcg_k} {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, waited = ndcg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  Early Stopping in Epoche {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Bestes Val-NDCG@{ndcg_k}: {best_ndcg:.4f}")
    return model, best_ndcg


# ----------------------------------------------------------------------
# Standalone-Pfad: rohe CSV -> Split -> Downsampling -> FM
# (im Notebook ueberspringst du das und rufst die Funktionen oben direkt auf)
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Pointwise FM fuer Expedia (torchfm).")
    p.add_argument("--data", required=True, help="Pfad zur rohen training_set_VU_DM.csv")
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--neg-ratio", type=int, default=5, help="Negative:Positive nach Downsampling")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="Mini-Lauf zum Testen")
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.quick:
        args.nrows = args.nrows or 80_000
        args.epochs = min(args.epochs, 3)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.data, nrows=args.nrows)
    print(f"Geladen: {len(df):,} Zeilen")

    # relevance-Label, falls noch nicht vorhanden
    if RELEVANCE_COL not in df.columns:
        df[RELEVANCE_COL] = np.where(df["booking_bool"] == 1, 5,
                            np.where(df["click_bool"] == 1, 1, 0))

    # Split nach srch_id (ganze Suchen, nicht einzelne Zeilen)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(df, groups=df[GROUP_COL]))
    df_train, df_val = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    # gemeinsames binaeres Target: geklickt ODER gebucht
    for d in (df_train, df_val):
        d["target"] = ((d["click_bool"] == 1) | (d["booking_bool"] == 1)).astype("float32")

    # Downsampling der Negativen NUR im Trainingsset
    pos = df_train[df_train["target"] == 1]
    neg = df_train[df_train["target"] == 0].sample(n=len(pos) * args.neg_ratio,
                                                   random_state=args.seed)
    df_train_bal = pd.concat([pos, neg]).sample(frac=1, random_state=args.seed)
    print(f"Train nach Downsampling: {len(df_train_bal):,} Zeilen "
          f"({df_train_bal['target'].mean() * 100:.1f} % positiv)")
    print(f"Val (intakt): {len(df_val):,} Zeilen, {df_val[GROUP_COL].nunique():,} Suchen\n")

    # Encoder auf (Downsampling-)Train fitten, auf beide anwenden
    encoders, field_dims = fit_encoders(df_train_bal, n_bins=args.n_bins)
    X_tr, _ = transform(df_train_bal, encoders)
    X_va, _ = transform(df_val, encoders)
    print(f"{len(field_dims)} Felder | Embedding-Tabellen gesamt: {sum(field_dims):,} Zeilen\n")

    print("--- Training Pointwise FM ---")
    train_fm(X_tr, df_train_bal["target"].to_numpy("float32"),
             X_va, df_val, field_dims,
             embed_dim=args.embed_dim, epochs=args.epochs)


if __name__ == "__main__":
    main()
